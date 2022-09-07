"""Geometric Graph Transformer"""
from typing import Optional, Tuple

import torch.nn.functional as F  # noqa
from einops import rearrange, repeat  # noqa
from torch import nn, Tensor
import torch

from protein_learning.common.helpers import exists
from protein_learning.network.common.rigids import Rigids
from protein_learning.network.modules.node_block import NodeUpdateBlock
from protein_learning.network.modules.pair_block import PairUpdateBlock
from protein_learning.network.modules.net_config import GeomGTConfig
from protein_learning.network.modules.fape import FAPELoss
from protein_learning.network.common.constants import RIGID_SCALE


class GraphTransformer(nn.Module):  # noqa
    """Graph Transformer"""

    def __init__(
            self,
            config: GeomGTConfig
    ):

        super(GraphTransformer, self).__init__()
        c = self.config = config
        self.depth, self.use_ipa = c.depth, c.use_ipa

        # shared rigid update accross all layers
        self.to_rigid_update = nn.Sequential(
            nn.LayerNorm(c.node_dim),
            nn.Linear(c.node_dim, 6)
        ) if c.use_ipa else None

        self.share_weights = c.share_weights
        if c.share_weights and c.use_ipa:
            self.fape_aux = FAPELoss(clamp_prob=0.9, scale=10)

        depth = 1 if c.share_weights else c.depth
        # Node Updates
        self.node_updates = nn.ModuleList([NodeUpdateBlock(
            node_dim=c.node_dim,
            pair_dim=c.pair_dim,
            use_ipa=c.use_ipa,
            **c.node_update_kwargs,
        ) for _ in range(depth)])

        # Pair Updates
        self.pair_updates = nn.ModuleList([PairUpdateBlock(
            node_dim=c.node_dim,
            pair_dim=c.pair_dim,
            **c.pair_update_kwargs,
        ) for _ in range(depth)]) if exists(c.pair_update_kwargs) else [None] * depth

    def update_rigids(self, feats: Tensor, rigids: Rigids, msk: Optional[Tensor] = None) -> Optional[Rigids]:
        """Update rigid transformations"""

        scale = 0.1 if self.config.scale_rigid_update else 1
        if exists(self.to_rigid_update):
            update = self.to_rigid_update(feats) * scale
            if exists(msk):
                msk = rearrange(msk, "i->() i ()")
                update = torch.masked_fill(update, msk, 0)
            quaternion_update, translation_update = update.chunk(2, dim=-1)
            quaternion_update = F.pad(quaternion_update, (1, 0), value=1.)  # noqa
            return rigids.compose(Rigids(quaternion_update, translation_update))
        return None

    def forward(
            self,
            node_feats: Tensor,
            pair_feats: Optional[Tensor],
            rigids: Optional[Rigids] = None,
            true_rigids: Optional[Rigids] = None,
            res_mask: Optional[Tensor] = None,
            pair_mask: Optional[Tensor] = None,
            compute_aux: Optional[bool] = True,
            rigid_update_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Rigids], Optional[Tensor]]:
        """
        :param node_feats: node features (b,n,d_node)
        :param pair_feats: pair features (b,n,n,d_pair)
        :param rigids: rigids to use for IPA (Identity is used o.w.)
        :param true_rigids: native protein rigids for aux. loss
        (applicable only if weights are shared)
        :param res_mask: residue mask (b,n)
        :param pair_mask: pair mask (b,n,n)
        :param compute_aux: whether to compute auxilliary FAPE loss between iterations
        :param rigid_update_mask: described in "flexible coordinates" section of paper,
        used if coordinates are to be treated as part of the input - in which case,
        the true_rigids parameter should not be none (o.w. this is skipped)

        :return:
            (1) scalar features
            (2) pair features
            (3) rigids (optional)
            (4) auxilliary loss (optional) - computed only if weights are shared
            and true rigids are supplied
        """
        assert exists(true_rigids) or not self.share_weights
        node_feats, pair_feats, device = node_feats, pair_feats, node_feats.device
        b, n, *_ = node_feats.shape
        assert node_feats.ndim == 3 and pair_feats.ndim == 4, f"scalar and pair feats must have batch dimension!"

        if exists(rigid_update_mask) and not exists(true_rigids):
            print("[WARNING] rigid update mask was passed, but rigids are missing from input!")

        if self.use_ipa:
            # if no initial rigids passed in, start from identity
            rigids = rigids if exists(rigids) else \
                Rigids.IdentityRigid(leading_shape=(b, n), device=device)  # noqa
            rigids = rigids.scale(1 / RIGID_SCALE)

        if exists(rigid_update_mask) and exists(true_rigids):
            msk = rigid_update_mask
            centroid = torch.mean(true_rigids.translations[0, msk], dim=0, keepdim=True)
            rigids.translations[0, ~msk] = centroid * torch.ones_like(rigids.translations[0, ~msk])
            trans, quats = true_rigids.translations[0, msk], true_rigids.quaternions[0, msk]
            map(lambda x: x.detach().clone() * (1 / RIGID_SCALE), (trans, quats))
            rigids.translations[0, msk], rigids.quaternions[0, msk] = trans, quats

        node_updates, pair_updates, aux_loss = self.node_updates, self.pair_updates, None

        aux_loss = 0 if self.share_weights and compute_aux else None
        if self.share_weights:
            node_updates = [node_updates[0]] * self.depth
            pair_updates = [pair_updates[0]] * self.depth

        forward_kwargs = dict(node_feats=node_feats, pair_feats=pair_feats, mask=pair_mask)
        for node_update, pair_update in zip(node_updates, pair_updates):
            # update rigids
            rigids = rigids.detach_rot() if self.use_ipa else None
            # update node
            node_feats = node_update(rigids=rigids, **forward_kwargs)
            forward_kwargs["node_feats"] = node_feats
            # update pair
            pair_feats = pair_update(**forward_kwargs) if exists(pair_update) else None
            forward_kwargs["pair_feats"] = pair_feats

            rigids = self.update_rigids(node_feats, rigids, msk=rigid_update_mask)
            if self.share_weights and self.use_ipa and compute_aux:
                rigids = rigids.scale(RIGID_SCALE)
                aux_loss = self.aux_loss(
                    res_mask=res_mask,
                    pred_rigids=rigids,
                    true_rigids=true_rigids,
                ) / self.depth + aux_loss
                rigids = rigids.scale(1 / RIGID_SCALE)

        rigids = rigids.scale(RIGID_SCALE) if exists(rigids) else None
        return node_feats, pair_feats, rigids, aux_loss

    def aux_loss(
            self,
            res_mask: Optional[Tensor],
            pred_rigids: Rigids,
            true_rigids: Rigids,
    ):
        """Auxiliary loss when IPA layers have shared weights"""
        pred_ca = pred_rigids.translations
        native_ca = true_rigids.translations
        assert pred_ca.shape == native_ca.shape, \
            f"{pred_ca.shape, native_ca.shape}"
        pred_ca, native_ca = map(
            lambda x: rearrange(x, "b n c -> b n () c"),
            (pred_ca, native_ca)
        )
        assert pred_ca.ndim == 4, f"{pred_ca.shape}"
        if exists(res_mask):
            assert res_mask.ndim == 2, f"{res_mask.shape}"
            res_mask = res_mask.unsqueeze(-1)
        return self.fape_aux.forward(
            pred_coords=pred_ca,
            true_coords=native_ca.detach(),
            pred_rigids=pred_rigids,
            true_rigids=true_rigids.detach_all(),
            coord_mask=res_mask,
            reduce=True,
        )

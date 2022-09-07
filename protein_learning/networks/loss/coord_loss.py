"""Coordinate Based Loss Functions"""
import random
from typing import Optional

import torch
from einops import rearrange, repeat  # noqa
from torch import nn, Tensor

from protein_learning.common.global_constants import get_logger
from protein_learning.common.helpers import exists, safe_norm
from protein_learning.common.rigids import Rigids

logger = get_logger(__name__)

flatten_coords = lambda crds: rearrange(crds, "b n a c-> b (n a) c")
to_res_rel_coords = lambda x, y: rearrange(x, "b n a c -> b n a () c") - rearrange(y, "b n a c -> b n () a c")
BB_LDDT_THRESHOLDS, SC_LDDT_THRESHOLDS = [0.5, 1, 2, 4], [0.25, 0.5, 1, 2]
to_rel_dev_coords = lambda x: rearrange(x, "b n a ... -> b (n a) () ...") - rearrange(x, "b n a ... -> b () (n a) ...")
max_value = lambda x: torch.finfo(x.dtype).max  # noqa


def per_residue_mean(x: Tensor, mask: Tensor) -> Tensor:
    """Takes a (masked) mean over the atom axis for each residue,
    then takes a mean over the residue axis.
    :param x: shape (b,n,a,*) -> (batch, res, atom, *) (must have 4 dimensions)
    :param mask: shape (b,n,a) -> (batch, res, atom)
    mask[b,i,a] indicates if the atom is present for a given residue.
    :return : mean of (masaked) per-residue means
    """
    if mask is None:
        return torch.mean(x)
    assert x.ndim == 4
    atoms_per_res = mask.sum(dim=-1)
    retain_mask = atoms_per_res > 0
    x = x.masked_fill(~mask.unsqueeze(-1), value=0.)
    per_res_mean = x.sum(dim=(-1, -2)) / torch.clamp_min(atoms_per_res, 1.)
    return torch.sum(per_res_mean / torch.sum(retain_mask, dim=-1, keepdim=True))


class FAPELoss(nn.Module):
    """FAPE Loss"""

    def __init__(
            self,
            d_clamp: float = 10,
            eps: float = 1e-8,
            scale: float = 10,
            clamp_prob: float = 0.9,
    ):
        """Clamped FAPE loss

        :param d_clamp: maximum distance value allowed for loss - all values larger will be clamped to d_clamp
        :param eps: tolerance factor for computing norms (so gradient is defined in sqrt)
        :param scale: (Inverse) Amount to scale loss by (usually equal tho d_clamp)
        :param clamp_prob: Probability with which loss values are clamped [0,1]. If this
        value is not 1, then a (1-clamp_prob) fraction of samples will not have clamping applied to
        distance deviations
        """
        super(FAPELoss, self).__init__()
        self._d_clamp, self.eps, self.scale = d_clamp, eps, scale
        self.clamp_prob = clamp_prob
        self.clamp = lambda diffs: torch.clamp_max(
            diffs, self._d_clamp if random.uniform(0, 1) < self.clamp_prob else max_value(diffs)
        )

    def forward(
            self,
            pred_coords: Tensor,
            true_coords: Tensor,
            pred_rigids: Optional[Rigids] = None,
            true_rigids: Optional[Rigids] = None,
            coord_mask: Optional[Tensor] = None,
            reduce: bool = True,
            clamp_val: Optional[float] = None,
            mask_fill: float = 0,
    ) -> Tensor:
        """Compute FAPE Loss

        :param pred_coords: tensor of shape (b,n,a,3)
        :param true_coords: tensor of shape (b,n,a,3)
        :param pred_rigids: (Optional) predicted rigid transformation
        :param true_rigids: (Optional) rigid transformation computed on native structure.
        If missing, the transformation is computed assuming true_points[:,:,:3]
        are N,CA, and C coordinates
        :param coord_mask: (Optional) tensor of shape (b,n,a) indicating which atom coordinates
        to compute loss for
        :param reduce: whether to output mean of loss (True), or return loss for each input coordinate
        :param : clamp_val: value to use for clamping (optional)

        :return: FAPE loss between predicted and true coordinates
        """
        b, n, a = true_coords.shape[:3]
        assert pred_coords.ndim == true_coords.ndim == 4

        true_rigids = true_rigids if exists(true_rigids) else Rigids.RigidFromBackbone(true_coords)
        pred_rigids = pred_rigids if exists(pred_rigids) else Rigids.RigidFromBackbone(pred_coords)

        pred_coords, true_coords = map(lambda x: repeat(x, "b n a c -> b () (n a) c"),
                                       (pred_coords, true_coords))

        # rotate and translate coordinates into local frame
        true_coords_local = true_rigids.apply_inverse(true_coords).detach()
        pred_coords_local = pred_rigids.apply_inverse(pred_coords)

        # compute deviations of predicted and actual coords.
        unclamped_diffs = safe_norm(pred_coords_local - true_coords_local, dim=-1, eps=self.eps)
        diffs = torch.clamp_max(unclamped_diffs, clamp_val) if exists(clamp_val) \
            else self.clamp(unclamped_diffs)

        if exists(coord_mask):
            residue_mask = torch.any(coord_mask, dim=-1, keepdim=True)
            coord_mask = repeat(coord_mask, "b n a -> b m (n a)", m=n)
            coord_mask = coord_mask.masked_fill(~residue_mask, False)

        if reduce:
            return (1 / self.scale) * torch.mean(diffs[coord_mask] if exists(coord_mask) else diffs)

        return (1 / self.scale) * (diffs.masked_fill(~coord_mask, mask_fill) if exists(coord_mask) else diffs)

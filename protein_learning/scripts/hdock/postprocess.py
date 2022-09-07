import sys
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['OPENBLAS_NUM_THREADS'] = '2'
os.environ['MKL_NUM_THREADS'] = '2'
os.environ['OMP_NUM_THREADS'] = '2'
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import subprocess
from functools import partial
from multiprocessing.pool import Pool
import torch
from biopandas.pdb import PandasPdb  # noqa
import time


def exists(x):
    return x is not None


def default(x, y):
    return x if exists(x) else y


def join(*args):
    return os.path.join(*args)


def mkdir(*args):
    x = join(*args)
    os.makedirs(x, exist_ok=True)
    return x


def is_file(*args):
    return os.path.isfile(join(*args))


def is_dir(*args):
    return not is_file(*args)


def is_pdb(x):
    return x.endswith(".pdb")


def copy(from_path, to_path):
    subprocess.call(f"cp {from_path} {to_path}", shell=True)


def has_chain(ppdb, chain):
    chain_mask = np.array(ppdb.df["ATOM"]["chain_id"] == chain)
    return np.any(chain_mask)


def get_ptn_coords(pdb_path, atom_ty="CA", chain=None, ppdb=None):
    ppdb = ppdb if exists(ppdb) else PandasPdb().read_pdb(pdb_path)
    atom_mask = ppdb.df["ATOM"]["atom_name"] == atom_ty
    crds = []
    if chain is not None:
        chain_mask = np.array(ppdb.df["ATOM"]["chain_id"] == chain)
        atom_mask = np.logical_and(atom_mask, chain_mask)
    for crd in "x_coord y_coord z_coord".split():
        crds.append(np.array(ppdb.df["ATOM"][crd][atom_mask]))
    crds = list(map(lambda x: x[:, None], crds))
    return np.concatenate(crds, axis=-1)


def get_ppdb(path):
    return PandasPdb().read_pdb(path)


def get_res_ids_from_atom_ids(pdb_path, atom_ids, ppdb=None):
    ppdb = ppdb if exists(ppdb) else PandasPdb().read_pdb(pdb_path)
    return [ppdb.df["ATOM"]["residue_number"][x - 1] for x in atom_ids]


def get_rec_lig_paths(rec_lig_root, target):
    rec = join(rec_lig_root, target + "_rec.pdb")
    lig = join(rec_lig_root, target + "_lig.pdb")
    return rec, lig


def _get_contacts(x, y, thresh=10):
    x, y = map(torch.tensor, (x, y))
    dists = torch.cdist(x, y).numpy()
    dists[dists > thresh] = thresh + 1
    dists[dists <= thresh] = 0
    return np.where(dists == 0)


def get_contact_mat(x, y, thresh=10):
    x, y = map(torch.tensor, (x, y))
    dists = torch.cdist(x, y).numpy()
    dists[dists <= thresh] = 1
    dists[dists > 1.01] = 0
    return dists


def count_contacts(contacts, rec_coords, lig_crds, thresh=14):
    return sum([1 if np.linalg.norm(lig_crds[lc] - rec_coords[rc]) < thresh else 0 for (rc, lc) in contacts])


def get_interface_vecs(x, y, contacts, thresh=10):
    contacts = torch.tensor(contacts)
    i1 = torch.sum(contacts, dim=1)
    i2 = torch.sum(contacts, dim=0)
    assert i1.shape[0] == x.shape[0]
    i1, i2 = map(lambda z: z.bool().float(), (i1, i2))
    assert torch.max(i1) < 1.001
    return i1.numpy(), i2.numpy()


def get_model_int_n_contact_score(rec_coords, lig_path, contact_list, rec_int_list=None, lig_int_list=None, thresh=14):
    lig_ppdb = PandasPdb().read_pdb(lig_path)
    lig_coords = get_ptn_coords(lig_path, ppdb=lig_ppdb)
    score = count_contacts(contact_list, rec_coords, lig_coords, thresh=thresh)
    rec_int, lig_int = None, None
    int_score = 0
    if exists(lig_int_list) or exists(rec_int_list):
        int_score = 1e6
        contacts = get_contact_mat(rec_coords, lig_coords)
        rec_int, lig_int = get_interface_vecs(rec_coords, lig_coords, contacts, thresh=thresh)
    if exists(lig_int_list):
        int_score = min(sum([lig_int[i] for i in lig_int_list]), int_score)
    if exists(rec_int_list):
        int_score = min(sum([rec_int[i] for i in rec_int_list]), int_score)
    return score + int_score


def get_contacts_from_structure(complex_dir, tgt, n_con):
    # Only works for antibodies!
    path = join(complex_dir, f"{tgt}_complex.pdb")
    if not os.path.exists(path):
        return []
    ppdb = get_ppdb(path)
    rec_crds = []
    for c in "H L".split():
        if has_chain(ppdb, c):
            rec_crds.append(get_ptn_coords(None, ppdb=ppdb, chain=c))
    rec_crds = np.concatenate(rec_crds, axis=0)
    lig_crds = get_ptn_coords(None, ppdb=ppdb, chain="A")
    c_rec, c_lig = _get_contacts(rec_crds, lig_crds)
    n_con = min(n_con, len(c_rec))
    choices = np.random.choice(np.arange(len(c_rec)), size=n_con, replace=False)
    c_rec, c_lig = c_rec[choices], c_lig[choices]
    return list(zip(c_rec, c_lig))


def get_contacts_from_file(contact_file, pdb_root, tgt, is_hdock=True):
    if not os.path.exists(contact_file):
        return []
    rppdb, lppdb = map(lambda x: get_ppdb(x), get_rec_lig_paths(pdb_root, tgt))
    rcs, lcs = [], []
    with open(contact_file, "r+") as f:
        for x in f:
            i, j, d = x.strip().split()
            if is_hdock:
                i, j = map(lambda x: x.split(":")[0], (i, j))
            rcs.append(int(i))
            lcs.append(int(j))
    if not is_hdock:
        rcs = get_res_ids_from_atom_ids(None, rcs, ppdb=rppdb)
        lcs = get_res_ids_from_atom_ids(None, lcs, ppdb=lppdb)
    return list(zip(rcs, lcs))


def get_interface_from_file(tgt_dir, tgt, pdb_root):
    rppdb, lppdb = map(lambda x: get_ppdb(x), get_rec_lig_paths(pdb_root, tgt))
    is_ab_ag = has_chain(lppdb, "A")
    r_coords = get_ptn_coords(None, chain="H" if is_ab_ag else "R", ppdb=rppdb)
    r_int_file, l_int_file = map(lambda x: join(tgt_dir, x), ("rec_active_site.txt", "lig_active_site.txt"))
    r_int, l_int = [], []
    if os.path.exists(r_int_file):
        with open(r_int_file, "r+") as f:
            ridx_n_chain = [x.strip().split(":") for x in f]
        for (idx, chain) in ridx_n_chain:
            if chain == "L":
                r_int.append(int(idx) + len(r_coords))
            else:
                r_int.append(int(idx))
    if os.path.exists(l_int_file):
        with open(l_int_file, "r+") as f:
            lidx_n_chain = [x.strip().split(":") for x in f]
        for (idx, chain) in lidx_n_chain:
            l_int.append(int(idx))

    return r_int, l_int


def make_models_hdock(fldr, max_models=500):
    try:
        os.chdir(fldr)
        subprocess.call(f"~/HDOCKlite-v1.1/createpl Hdock.out model.pdb -models -nmax {max_models} &", shell=True)
    except:
        print(f"failed to make models for {os.path.basename(fldr)}")


def get_best_model_hdock(tgt, data_root, max_models, thresh=12, n_con=None, complex_dir=None):
    try:
        tgt_fldr = os.path.join(data_root, tgt)
        pdb_root = tgt_fldr
        rppdb, lppdb = map(lambda x: get_ppdb(x), get_rec_lig_paths(pdb_root, tgt))
        contact_file = os.path.join(tgt_fldr, "contacts.txt")
        if os.path.exists(contact_file):
            contacts = get_contacts_from_file(contact_file, pdb_root, tgt, is_hdock=True)
        elif exists(complex_dir) and exists(n_con):
            print("loading structure contacts")
            try:
                contacts = get_contacts_from_structure(complex_dir, tgt, n_con)
            except Exception as e:
                print(f"failed to load contacts {e}")
        else:
            contacts = []
        r_int, l_int = get_interface_from_file(tgt_fldr, tgt, tgt_fldr)
        rec_coords = get_ptn_coords(None, ppdb=rppdb)
        best_score, best_model_idx, max_score = 0, 0, len(contacts) + min(len(l_int), len(r_int))
        print(f"[INFO] beginning scoring for tgt {tgt}")
        for i in range(max_models):
            try:
                lig_path = os.path.join(tgt_fldr, f"model_{i + 1}.pdb")
                if not os.path.exists(lig_path):
                    continue
                score = get_model_int_n_contact_score(
                    rec_coords, lig_path, contact_list=contacts, rec_int_list=r_int, lig_int_list=l_int, thresh=thresh
                )
                if score == max_score:
                    best_model_idx, best_score = i, score
                    break
                if score > best_score:
                    best_model_idx, best_score = i, score
            except Exception as e:
                print(f"failing best model - got exception {e}")
                pass
        best_model_path = os.path.join(tgt_fldr, f"model_{best_model_idx + 1}.pdb")
        print(f"best model for target : {tgt} = {best_model_path}, "
              f"best score : {best_score}, max_score = {max_score}, contacts:{contacts}")
        return tgt, best_model_path
    except:
        print(f"failed to get model for {tgt}")
        return None, None


def copy_models(tgt_n_model_paths, out_root, result_root):
    for tgt, path in tgt_n_model_paths:
        try:
            pdb_dir = os.path.join(result_root, tgt)
            lig_path = os.path.join(out_root, f"{tgt}_lig.pdb")
            cmd = f"cp {path} {lig_path} &"
            subprocess.call(cmd, shell=True)

            rec_path_to = os.path.join(out_root, f"{tgt}_rec.pdb")
            rec_path_from = os.path.join(pdb_dir, f"{tgt}_rec.pdb")
            cmd = f"cp {rec_path_from} {rec_path_to} &"
            subprocess.call(cmd, shell=True)
        except:
            print(f"failed to copy model for {tgt}, skipping")


def remove_models(targets, result_root):
    for tgt in targets:
        try:
            tgt_dir = os.path.join(result_root, tgt)
            os.chdir(tgt_dir)
            cmd = f"rm -r model*.pdb &"
            subprocess.call(cmd, shell=True)
        except:
            print(f"failed to remove models for {tgt}, skipping")


def recopy_rec_lig(result_dir, pdb_dir, tgts):
    for tgt in tgts:
        try:
            rec, lig = map(lambda x: f"{tgt}_{x}.pdb", "rec lig".split())
            paths_to = map(lambda x: join(result_dir, tgt, x), (rec, lig))
            paths_from = map(lambda x: join(pdb_dir, x), (rec, lig))
            for (x, y) in zip(paths_from, paths_to):
                copy(x, y)
        except:
            print(f"failed to copy {tgt}")
            pass


if __name__ == "__main__":
    parser = ArgumentParser(description="Build HDOCK pdb files",  # noqa
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--pdb_root',
        help="directory to load pdbs from",
    )

    parser.add_argument(
        '--out_root',
        help="root directory to write results to - must have one folder per target"
    )
    parser.add_argument(
        '--n_threads',
        default=10,
        type=int,
    )
    parser.add_argument(
        '--max_models',
        default=500,
        type=int,
    )
    parser.add_argument(
        "--result_root",
        help="make directory structure in out root from pdb_dir targets"
    )
    parser.add_argument(
        "--complex_dir",
        default=None,
    )
    parser.add_argument(
        "--n_con",
        default=None,
        type=int,
    )
    parser.add_argument("--max_tgts", default=1000, type=int)

    args = parser.parse_args(sys.argv[1:])
    print(f"[INFO] Args: {args}")
    os.makedirs(args.out_root, exist_ok=True)
    N = args.max_tgts
    fn = partial(
        get_best_model_hdock,
        data_root=args.result_root,
        max_models=args.max_models,
        n_con=args.n_con,
        complex_dir=args.complex_dir,
    )
    tgts = [x for x in os.listdir(args.result_root) if is_dir(args.result_root, x) and len(x) == 4]
    recopy_rec_lig(args.result_root, args.pdb_root, tgts)

    for i, tgt in enumerate(tgts[:N]):
        if (i + 1) % 10 == 0:
            time.sleep(10)
        make_models_hdock(join(args.result_root, tgt), max_models=args.max_models)
        time.sleep(10)
    print("[INFO] processing models")
    if args.n_threads > 1:
        with Pool(args.n_threads) as p:
            tgt_n_model_paths = p.map(fn, tgts[:N])
    else:
        tgt_n_model_paths = [fn(t) for t in tgts[:N]]

    copy_models(tgt_n_model_paths, args.out_root, args.result_root)
    remove_models(tgts[:N], args.result_root)

import sys
import os
import subprocess
from collections import defaultdict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter


def split_complex(pdb_in, target_name):
    sub_pdbs = defaultdict(list)
    curr_pdb = 0
    with open(pdb_in, "r") as f_in:
        for line in f_in:
            sub_pdbs[curr_pdb].append(line)
            if "TER" in line:
                curr_pdb += 1
    lig_pdb_idx = 2
    for i in range(len(sub_pdbs)):
        if "lig" in sub_pdbs[i][0]:
            lig_pdb_idx = i
            break

    # write receptor
    rec_path = os.path.join(os.path.dirname(pdb_in), f"{target_name}_rec.pdb")
    with open(rec_path, "w+") as rec:
        for i in range(lig_pdb_idx):
            for line in sub_pdbs[i]:
                rec.write(line)

    # wrte ligand
    lig_path = os.path.join(os.path.dirname(pdb_in), f"{target_name}_lig.pdb")
    with open(lig_path, "w+") as lig:
        for line in sub_pdbs[lig_pdb_idx]:
            lig.write(line)


if __name__ == "__main__":
    parser = ArgumentParser(description="Build HDOCK pdb files",  # noqa
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--pdb_dir',
        help="directory to store pdbs in"
    )
    parser.add_argument(
        '--data_root',
        help="root directory with hdock results"
    )

    args = parser.parse_args(sys.argv[1:])
    root, pdb_dir = args.data_root, args.pdb_dir
    cmd = f"~/HDOCKlite-v1.1/createpl Hdock.out top1.pdb -nmax 1 -complex"
    os.makedirs(pdb_dir, exist_ok=True)

    for x in os.listdir(root):
        tmp = os.path.join(root, x)
        if os.path.isdir(tmp):
            if "Hdock.out" in os.listdir(tmp):
                os.chdir(tmp)
                subprocess.call(cmd, shell=True)
                # split the complex file so we can compare with dockQ
                split_complex(os.path.join(tmp, "top1.pdb"), x)
                # copy into target pdb dir
                lig_path_from = os.path.join(tmp, f"{x}_lig.pdb")
                rec_path_from = os.path.join(tmp, f"{x}_rec.pdb")
                lig_path_to = os.path.join(pdb_dir, f"{x}_lig.pdb")
                rec_path_to = os.path.join(pdb_dir, f"{x}_rec.pdb")
                subprocess.call(f"cp {lig_path_from} {lig_path_to}", shell=True)
                subprocess.call(f"cp {rec_path_from} {rec_path_to}", shell=True)

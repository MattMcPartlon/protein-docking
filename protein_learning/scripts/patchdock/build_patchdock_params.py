import os
import subprocess
from protein_learning.common.io.utils import load_npy
import numpy as np
import time
import sys

PATCHDOCK_PATH = "/home/mmcpartlon/patch_dock_download/PatchDock"
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

is_pdb = lambda x: x.endswith(".pdb")


def exists(x):
    return x is not None


def lig_name(ligand):
    return ligand.split("_")[0]


def _write_file(path, lines):
    with open(path, "w+") as f:
        for line in lines:
            f.write(line)


def add_restrs(tgt_dir, param_file):
    contact_file = os.path.join(tgt_dir, "contacts.txt")
    if os.path.exists(contact_file):
        lines = []
        with open(param_file, "r") as f:
            for line in f:
                if line.startswith("#distanceConstraint"):
                    lines.append(f"distanceConstraintFile ./contacts.txt\n")
                else:
                    lines.append(line)
        _write_file(param_file, lines)


def add_active_sites(tgt_dir, param_file, is_ab):
    lig_file = os.path.join(tgt_dir, "lig_active_site.txt")
    rec_file = os.path.join(tgt_dir, "rec_active_site.txt")
    lig_f_name = "lig_active_site.txt"
    rec_f_name = "rec_active_site.txt"
    if is_ab:  # patchdock swaps rec. lig for Ab-Ag dock
        lig_file, rec_file = rec_file, lig_file
        rec_f_name, lig_f_name = lig_f_name, rec_f_name
    if os.path.exists(rec_file) or os.path.exists(lig_file):
        with open(param_file, "r") as f:
            lines = [x for x in f]

        if os.path.exists(lig_file) and not is_ab:
            for i, line in enumerate(lines):
                if line.startswith("#ligandActiveSite"):
                    lines[i] = f"ligandActiveSite ./{lig_f_name}\n"

        if os.path.exists(rec_file):
            for i, line in enumerate(lines):
                if line.startswith("#receptorActiveSite"):
                    lines[i] = f"receptorActiveSite ./{rec_f_name}\n"

        _write_file(param_file, lines)


if __name__ == "__main__":

    parser = ArgumentParser(description="Build Patchdock files",  # noqa
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--pdb_dir',
        help="directory to load pdbs from"
    )
    parser.add_argument(
        '--out_root',
        help="root directory to write results to - must have one folder per target"
    )
    parser.add_argument(
        "--make_dirs_from_targets",
        action="store_true",
        help="make directory structure in out root from pdb_dir targets"
    )
    parser.add_argument(
        "--is_ab_ag",
        action="store_true",
        help="whether the target docking is between Ab and Ag"
    )

    args = parser.parse_args(sys.argv[1:])
    if args.make_dirs_from_targets:
        entries = set([x[:4] for x in os.listdir(args.pdb_dir) if is_pdb(x)])
    else:
        entries = set(
            [x for x in os.listdir(args.out_root) if
             os.path.isdir(os.path.join(args.out_root, x)) and len(x) == 4]
        )

    # make param files
    print(f"[INFO] gathered {len(entries)} targets")
    for i, tgt in enumerate(entries):
        try:
            receptor, ligand = map(lambda x: f"{tgt}_{x}", ("rec", "lig"))
            tgt_fldr = os.path.join(args.out_root, lig_name(ligand))
            os.makedirs(tgt_fldr, exist_ok=True)
            os.chdir(tgt_fldr)
            receptor, ligand = map(lambda y: os.path.join(args.pdb_dir, y + ".pdb"), (receptor, ligand))
            if not (os.path.exists(receptor) and os.path.exists(ligand)):
                print(f"skipping : {tgt}")
                continue
            # create param file in target folder
            cmd = f"{PATCHDOCK_PATH}/buildParams.pl {receptor} {ligand} 4.0 {'AA' if args.is_ab_ag else ''}"
            # print(f"running command : {cmd} \nFrom working directory {tgt_fldr}")
            print(f"{tgt}:{i}\nCalling: {cmd}")
            subprocess.call(cmd, shell=True)
        except:
            print(f"skipping {tgt}")

    time.sleep(3)
    # add active sites/contacts
    print("[INFO] Adding active sites and distance constraints")
    for i, tgt in enumerate(entries):
        try:
            tgt_dir = os.path.join(args.out_root, tgt)
            param_file = os.path.join(args.out_root, tgt, "params.txt")
            add_active_sites(tgt_dir, param_file, args.is_ab_ag)
            add_restrs(tgt_dir, param_file)
        except:
            print(f"skipping {tgt}")

    with open(os.path.join(args.out_root, "run_patchdock.txt"), "w+") as f:
        for x in os.listdir(args.out_root):
            try:
                if os.path.isdir(os.path.join(args.out_root, x)):
                    pdp = os.path.join("/home/mmcpartlon/patch_dock_download/PatchDock", "patch_dock.Linux")
                    fldr = os.path.join(args.out_root, x)
                    f.write(f"cd {fldr}\n")
                    f.write(f"nohup {pdp} ./params.txt ./{x}.txt &\n")
            except:
                print(f"skipping {x}")

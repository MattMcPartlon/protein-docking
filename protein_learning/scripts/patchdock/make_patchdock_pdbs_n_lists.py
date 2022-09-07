import os
import subprocess
import time
import sys

OUT_ROOT = "/mnt/local/mmcpartlon/patchdock/db5_interface_5/"
os.makedirs(OUT_ROOT, exist_ok=True)
PATCHDOCK_PATH = "/home/mmcpartlon/patch_dock_download/PatchDock"
PDB_ROOT = "/mnt/local/mmcpartlon/db5_dror_pdbs/bound"
TARGET_LIST = "/mnt/local/mmcpartlon/db5_dror_pdbs/test.list"
INTERFACE_DAT = "/home/mmcpartlon/ProteinLearningV2/protein_learning/scripts/data/db5_interface_res.npy"
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

# receptorPdb /mnt/local/mmcpartlon/db5_dror_pdbs/bound/1EXB_r.pdb
# ligandPdb /mnt/local/mmcpartlon/db5_dror_pdbs/bound/1EXB_l.pdb
def get_ligand_n_receptor_pdbs(param_file):
    lig, rec = None, None
    with open(param_file, "r") as f:
        for line in f:
            if line.strip().startswith("receptorPdb"):
                rec = line.strip().split(" ")[1]
            if line.strip().startswith("ligandPdb"):
                lig = line.strip().split(" ")[1]
    return lig, rec


def remove_receptor_from_pdb(trans_pdb_path, ligand_pdb_path):
    pdb_in_lines = []
    ligand_pdb_lines = []
    with open(trans_pdb_path,"r") as f:
        pdb_in_lines = [x for x in f]

    with open(ligand_pdb_path,"r+") as f:
        ligand_pdb_lines = [x for x in f]






if __name__ == "__main__":
    parser = ArgumentParser(description="get stats from pdbs",  # noqa
                            epilog='',
                            formatter_class=ArgumentDefaultsHelpFormatter
                            )
    parser.add_argument(
        '--data_root',
        help="patchdock data root directory - should contain a folder for each target",
    )
    parser.add_argument(
        '--pdb_dir',
        help="directory to store patchdock pdbs in",
    )
    list_entries = []
    args = parser.parse_args(sys.argv[1:])
    os.makedirs(args.pdb_dir, exist_ok=True)
    #os.makedirs(os.path.basename(args.list_path), exist_ok=True)
    # first translate all pred. ligands
    for tgt in os.listdir(args.data_root):
        fldr = os.path.join(args.data_root, tgt)
        if os.path.isdir(fldr):
            tgt_params = os.path.join(fldr, "params.txt")
            if os.path.exists(tgt_params):
                os.chdir(fldr)
                subprocess.call("rm -rf -r *1.pdb",shell=True)
                subprocess.call(f"{PATCHDOCK_PATH}/transOutput.pl ./{tgt}.txt 1 1 0", shell=True)
    time.sleep(20)
    # copy over to folder
    for tgt in os.listdir(args.data_root):
        fldr = os.path.join(args.data_root, tgt)
        if os.path.isdir(fldr):
            tgt_params = os.path.join(fldr, "params.txt")
            if os.path.exists(tgt_params):
                lig_path, rec_path = get_ligand_n_receptor_pdbs(tgt_params)
                # transform predicted ligand
                os.chdir(fldr)
                trans_ligand = f"{tgt}.txt.1.pdb"
                print(trans_ligand, os.path.exists(f"./{trans_ligand}"))
                lig_name = os.path.split(lig_path)[-1]
                rec_name = os.path.split(rec_path)[-1]
                subprocess.call(f"cp ./{trans_ligand} {args.pdb_dir}/{lig_name}", shell=True)
                subprocess.call(f"cp {rec_path} {args.pdb_dir}/{rec_name}", shell=True)
                list_entries.append([lig_name[:-4], rec_name[:-4]])


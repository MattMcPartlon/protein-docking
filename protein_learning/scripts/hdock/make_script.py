import sys
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

hd_cmd_path = "/home/mmcpartlon/HDOCKlite-v1.1/hdock"
is_pdb = lambda x: x.endswith(".pdb")


def add_restrs(tgt_dir, cmd):
    if os.path.exists(os.path.join(tgt_dir, "contacts.txt")):
        cmd += " -restr contacts.txt"
    return cmd


def add_active_sites(tgt_dir, cmd):
    if os.path.exists(os.path.join(tgt_dir, "lig_active_site.txt")):
        cmd += " -lsite lig_active_site.txt"
    if os.path.exists(os.path.join(tgt_dir, "rec_active_site.txt")):
        cmd += " -rsite rec_active_site.txt"
    return cmd


if __name__ == "__main__":
    parser = ArgumentParser(description="Build HDOCK pdb files",  # noqa
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

    args = parser.parse_args(sys.argv[1:])
    if args.make_dirs_from_targets:
        entries = set([x[:4] for x in os.listdir(args.pdb_dir) if is_pdb(x)])
    else:
        entries = set([x for x in os.listdir(args.out_root) if os.path.isdir(os.path.join(args.out_root,x))  and len(x) == 4])

    cmd_path = os.path.join(args.out_root, "run_hdock.sh")
    with open(cmd_path, "w+") as fout:
        fout.write(f"mkdir -p {args.out_root}\n")
        for tgt in entries:
            try:
                tgt_dir = os.path.join(args.out_root, tgt)
                fout.write(f"mkdir -p {tgt_dir}\n")
                fout.write(f"cd {tgt_dir}\n")
                fout.write(f"cp {os.path.join(args.pdb_dir,f'{tgt}_rec.pdb')} ./\n")
                fout.write(f"cp {os.path.join(args.pdb_dir,f'{tgt}_lig.pdb')} ./\n")
                run_cmd = f"nohup {hd_cmd_path} {tgt}_rec.pdb {tgt}_lig.pdb"
                run_cmd = add_active_sites(tgt_dir, run_cmd)
                run_cmd = add_restrs(tgt_dir, run_cmd)
                run_cmd += f" >{tgt}.log &\n"
                fout.write(run_cmd)
            except:
                print(f"skipping {tgt}")

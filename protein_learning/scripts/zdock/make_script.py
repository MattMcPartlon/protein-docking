import sys
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import subprocess

zdock_cmd_path = "/home/mmcpartlon/zdock3.0.2_linux_x64/zdock"
zd_root = "/home/mmcpartlon/zdock3.0.2_linux_x64/"
zdock_copy_paths = "create.pl create_lig uniCHARMM mark_sur".split()
zdock_copy_paths = [os.path.join(zd_root,x) for x in zdock_copy_paths]
is_pdb = lambda x: x.endswith(".pdb")


def copy_files(tgt_dir):
    cmd_str = ""
    for x in zdock_copy_paths:
        cmd_str+= f"cp {x} {tgt_dir}/ \n"
    return cmd_str


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
    os.makedirs(args.out_root,exist_ok=True)
    cmd_path = os.path.join(args.out_root, "run_zdock.sh")
    with open(cmd_path, "w+") as fout:
        fout.write(f"mkdir -p {args.out_root}\n")
        for tgt in entries:
            try:
                tgt_dir = os.path.join(args.out_root, tgt)
                fout.write(f"mkdir -p {tgt_dir}\n")
                fout.write(f"cd {tgt_dir}\n")
                fout.write(f"cp {os.path.join(args.pdb_dir,f'{tgt}_rec.pdb')} ./\n")
                fout.write(f"cp {os.path.join(args.pdb_dir,f'{tgt}_lig.pdb')} ./\n")
                fout.write(copy_files(tgt_dir))
                fout.write(f"./mark_sur {tgt}_rec.pdb {tgt}_rec_m.pdb\n")
                fout.write(f"./mark_sur {tgt}_lig.pdb {tgt}_lig_m.pdb\n")
                run_cmd = f"nohup {zdock_cmd_path} -R {tgt}_rec_m.pdb -L {tgt}_lig_m.pdb " \
                          f"-o {tgt}.out -N 10 >zdock_{tgt}.log &\n"
                fout.write(run_cmd)
            except:
                print(f"skipping {tgt}")

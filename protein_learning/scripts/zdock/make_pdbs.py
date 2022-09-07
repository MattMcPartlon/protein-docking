import sys
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import subprocess


def default(x, y):
    return x if x is not None else y


if __name__ == "__main__":
    parser = ArgumentParser(description="Build HDOCK pdb files",  # noqa
                            formatter_class=ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--result_dir',
        help="root directory of results (one folder per target)"
    )
    parser.add_argument(
        '--pdb_dir',
        help="directory to store pdbs in (result_dir/pred_pdbs is default)",
        default=None,
    )

    args = parser.parse_args(sys.argv[1:])
    result_dir = args.result_dir
    pdb_dir = default(args.pdb_dir, os.path.join(result_dir, "pred_pdbs"))
    os.makedirs(pdb_dir, exist_ok=True)
    targets = filter(lambda x: len(x) == 4 and os.path.isdir(os.path.join(result_dir, x)), os.listdir(result_dir))
    for tgt in targets:
        tgt_folder = os.path.join(result_dir, tgt)
        os.chdir(tgt_folder)
        if os.path.exists(os.path.join(tgt_folder, f"{tgt}.out")):
            subprocess.call(f"./create.pl {tgt}.out", shell=True)
            try:
                for i in range(9):
                    os.remove(f"./complex.{i + 2}.pdb")
            except:
                pass
            subprocess.call(f"cp ./complex.1.pdb {pdb_dir}/{tgt}_complex.pdb &", shell=True)

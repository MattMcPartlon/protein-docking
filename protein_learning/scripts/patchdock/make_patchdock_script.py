import os
import sys

OUT_ROOT = sys.argv[1]
PATCHDOCK_PATH = "/home/mmcpartlon/patch_dock_download/PatchDock"

for x in os.listdir(OUT_ROOT):
    if os.path.isdir(os.path.join(OUT_ROOT,x)):
        pdp = os.path.join("/home/mmcpartlon/patch_dock_download/PatchDock", "patch_dock.Linux")
        fldr = os.path.join(OUT_ROOT, x)
        print(f"nohup {pdp} {fldr}/params.txt {fldr}/{x}.txt &")

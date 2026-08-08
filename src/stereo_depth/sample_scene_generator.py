"""
Helper script to generate a standard Middlebury sample scene in data/sample_scene/ for unit tests and digest benchmarks.
"""

import os
import cv2
import numpy as np
from src.stereo_depth.pfm_io import write_pfm


def create_sample_middlebury_scene(output_dir="data/sample_scene"):
    os.makedirs(output_dir, exist_ok=True)

    h, w = 240, 320
    u_grid, v_grid = np.meshgrid(np.arange(w), np.arange(h))

    # Synthetic GT disparity map
    gt_disp = np.ones((h, w), dtype=np.float32) * 10.0
    gt_disp[v_grid > 100] = 30.0

    # Left image im0.png
    checker = ((u_grid // 16) % 2) ^ ((v_grid // 16) % 2)
    im0 = np.zeros((h, w, 3), dtype=np.uint8)
    im0[:, :, 0] = np.where(checker, 220, 40)
    im0[:, :, 1] = np.where(checker, 40, 220)
    im0[:, :, 2] = np.where(checker, 120, 120)

    # Right image im1.png (shifted by gt_disp)
    im1 = np.zeros_like(im0)
    for y in range(h):
        for x in range(w):
            d = int(gt_disp[y, x])
            if x - d >= 0:
                im1[y, x - d] = im0[y, x]

    cv2.imwrite(os.path.join(output_dir, "im0.png"), im0)
    cv2.imwrite(os.path.join(output_dir, "im1.png"), im1)
    write_pfm(os.path.join(output_dir, "disp0.pfm"), gt_disp)

    calib_content = (
        "cam0=[1000.0 0 160.0; 0 1000.0 120.0; 0 0 1]\n"
        "cam1=[1000.0 0 160.0; 0 1000.0 120.0; 0 0 1]\n"
        "doffs=0.0\n"
        "baseline=0.1\n"
        "width=320\n"
        "height=240\n"
        "ndisp=64\n"
        "vmin=0\n"
        "vmax=255\n"
    )
    with open(os.path.join(output_dir, "calib.txt"), "w") as f:
        f.write(calib_content)

    print(f"Created sample Middlebury scene at {output_dir}")


if __name__ == "__main__":
    create_sample_middlebury_scene()

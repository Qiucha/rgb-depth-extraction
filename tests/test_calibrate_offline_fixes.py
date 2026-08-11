import os
import cv2
import numpy as np
import pytest
from tools.calibrate_offline import load_image_pairs

def test_load_image_pairs_resolution_mismatch(tmp_path):
    # Create synthetic image pairs with different resolutions
    pair_dir = tmp_path / "pair_01"
    pair_dir.mkdir()
    
    # main=1920x1080, uw=1280x720
    # Note: shape is (height, width, channels)
    main_img = np.zeros((1080, 1920, 3), dtype=np.uint8)
    uw_img = np.zeros((720, 1280, 3), dtype=np.uint8)
    
    cv2.imwrite(str(pair_dir / "main.jpg"), main_img)
    cv2.imwrite(str(pair_dir / "ultrawide.jpg"), uw_img)
    
    pairs, orig_uw_sizes = load_image_pairs(str(tmp_path))
    
    assert len(pairs) == 1
    assert len(orig_uw_sizes) == 1
    
    img_m, img_u = pairs[0]
    
    # 2. Verifies the returned pairs have matching sizes (img_u resized to match img_m)
    assert img_m.shape[:2] == (1080, 1920)
    assert img_u.shape[:2] == (1080, 1920)
    
    # 3. Verifies that original UW sizes are preserved
    # shape[:2] returns (height, width)
    assert orig_uw_sizes[0] == (720, 1280)

import pytest
import numpy as np
import cv2
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.realworld.calibration_refiner import CalibrationRefiner

def create_synthetic_checkerboard(squares_x=6, squares_y=6, square_size=50, padding=50):
    img_w = squares_x * square_size + 2 * padding
    img_h = squares_y * square_size + 2 * padding
    img = np.ones((img_h, img_w), dtype=np.uint8) * 255
    
    for y in range(squares_y):
        for x in range(squares_x):
            if (x + y) % 2 == 1:
                y1 = padding + y * square_size
                y2 = padding + (y + 1) * square_size
                x1 = padding + x * square_size
                x2 = padding + (x + 1) * square_size
                img[y1:y2, x1:x2] = 0
                
    return img

def test_detect_corners_5x5():
    # 6x6 squares -> 5x5 inner corners
    img = create_synthetic_checkerboard(6, 6, 50, 50)
    
    # Intentionally start with the wrong pattern size to force auto_discovery
    refiner = CalibrationRefiner(pattern_size=(9, 6), square_size_m=0.026, auto_discover=True)
    found, corners, obj_pts = refiner.detect_corners(img)
    
    assert found is True, "Should find the 5x5 inner corner pattern via auto-discovery"
    assert corners is not None
    assert len(corners) == 25, f"Expected 25 corners (5x5), got {len(corners)}"

def test_auto_discover_pattern_size_5x5():
    img = create_synthetic_checkerboard(6, 6, 50, 50)
    
    # Create a dummy image pair list (since it takes a list of pairs)
    # We can just duplicate the image as both main and ultrawide
    pairs = [(img, img), (img, img)]
    
    refiner = CalibrationRefiner(pattern_size=(9, 6), square_size_m=0.026, auto_discover=True)
    discovered = refiner.auto_discover_pattern_size(pairs)
    
    assert discovered == (5, 5), f"Expected (5, 5), got {discovered}"

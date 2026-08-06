"""
Master entry point for running disparity depth extraction and generating HTML digest.
"""

import os
import sys

# Ensure src is in python path
sys.path.insert(0, os.path.abspath('.'))

from src.stereo_depth.digest_generator import generate_digest_data

if __name__ == '__main__':
    print("=== Starting Stereo Depth Information Extraction & Digest Generator ===")
    scenes = generate_digest_data(data_dir='data', output_dir='digest')
    print(f"Successfully processed {len(scenes)} scenes!")

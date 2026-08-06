"""
Parser for Middlebury stereo calibration text files (calib.txt).
"""

import re
import numpy as np


class StereoCalibration:
    def __init__(self, cam0, cam1, doffs, baseline, width, height, ndisp, vmin=0, vmax=255):
        self.cam0 = cam0
        self.cam1 = cam1
        self.doffs = doffs
        self.baseline = baseline
        self.width = width
        self.height = height
        self.ndisp = ndisp
        self.vmin = vmin
        self.vmax = vmax

        # Extract focal length from cam0 matrix (f = cam0[0,0])
        self.focal_length = cam0[0, 0]

    @classmethod
    def parse_file(cls, filepath):
        props = {}
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' in line:
                    key, val = line.split('=', 1)
                    key = key.strip()
                    val = val.strip()
                    props[key] = val

        def parse_matrix(mat_str):
            # Form: [1733.74 0 792.27; 0 1733.74 541.89; 0 0 1]
            clean = mat_str.strip('[]')
            rows = clean.split(';')
            mat = []
            for r in rows:
                elems = [float(x) for x in r.strip().split()]
                mat.append(elems)
            return np.array(mat, dtype=np.float32)

        cam0 = parse_matrix(props['cam0'])
        cam1 = parse_matrix(props['cam1'])
        doffs = float(props.get('doffs', 0.0))
        baseline = float(props.get('baseline', 1.0))
        width = int(props.get('width', 0))
        height = int(props.get('height', 0))
        ndisp = int(props.get('ndisp', 128))
        vmin = int(props.get('vmin', 0))
        vmax = int(props.get('vmax', 255))

        return cls(cam0, cam1, doffs, baseline, width, height, ndisp, vmin, vmax)

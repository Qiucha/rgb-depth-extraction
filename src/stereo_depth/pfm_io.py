"""
PFM (Portable Float Map) I/O module for reading Middlebury stereo ground truth disparity.
"""

import re
import numpy as np


def read_pfm(file_path):
    """
    Reads a PFM file into a numpy array (float32).
    Returns (data, scale) where data is a 2D numpy array [height, width].
    Middlebury PFM files use negative scale factor to indicate little-endian.
    """
    with open(file_path, 'rb') as f:
        header = f.readline().decode('latin-1').strip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise ValueError(f"Not a valid PFM file: {file_path}")

        dim_line = f.readline().decode('latin-1').strip()
        while dim_line.startswith('#'):
            dim_line = f.readline().decode('latin-1').strip()

        dimensions = re.match(r'^(\d+)\s+(\d+)$', dim_line)
        if dimensions:
            width = int(dimensions.group(1))
            height = int(dimensions.group(2))
        else:
            raise ValueError(f"Malformed dimensions in PFM: {dim_line}")

        scale_line = f.readline().decode('latin-1').strip()
        scale = float(scale_line)
        endian = '<' if scale < 0 else '>'

        data = np.fromfile(f, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        data = np.reshape(data, shape)
        # PFM stores rows from bottom to top, so flip vertically
        data = np.flipud(data)

        return data.astype(np.float32), abs(scale)


def write_pfm(file_path, image, scale=-1.0):
    """
    Writes a 2D numpy array (float32) to a PFM file.
    """
    with open(file_path, 'wb') as f:
        if image.dtype != np.float32:
            image = image.astype(np.float32)

        # Flip vertically before writing
        image_flipped = np.flipud(image)

        if image.ndim == 2:
            color = False
        elif image.ndim == 3 and image.shape[2] == 3:
            color = True
        else:
            raise ValueError("Image must have 2 dimensions (grayscale) or 3 (RGB)")

        header = 'PF\n' if color else 'Pf\n'
        f.write(header.encode('latin-1'))
        f.write(f"{image.shape[1]} {image.shape[0]}\n".encode('latin-1'))

        endian = '<' if scale < 0 else '>'
        f.write(f"{scale}\n".encode('latin-1'))

        image_flipped.astype(endian + 'f').tofile(f)

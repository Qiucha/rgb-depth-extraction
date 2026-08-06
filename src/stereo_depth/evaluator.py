"""
Quantitative evaluator for disparity and depth estimation against ground truth.
"""

import numpy as np


class DepthEvaluator:
    @staticmethod
    def evaluate_disparity(pred_disp, gt_disp, valid_mask=None, bad_threshold=1.0):
        """
        Evaluates estimated disparity map against ground truth PFM disparity map.
        PFM files contain inf/nan for invalid/occluded pixels.
        """
        if valid_mask is None:
            mask = np.isfinite(gt_disp) & (gt_disp > 0)
        else:
            mask = np.isfinite(gt_disp) & (gt_disp > 0) & valid_mask

        if not np.any(mask):
            return {
                'rmse': 0.0,
                'mae': 0.0,
                'bad_pixels_ratio': 0.0,
                'bad_pixels_2px': 0.0,
                'valid_count': 0
            }

        diff = np.abs(pred_disp[mask] - gt_disp[mask])

        mae = float(np.mean(diff))
        rmse = float(np.sqrt(np.mean(diff ** 2)))
        bad_pixels_ratio = float(np.mean(diff > bad_threshold) * 100.0)
        bad_pixels_2px = float(np.mean(diff > 2.0) * 100.0)

        return {
            'rmse': round(rmse, 4),
            'mae': round(mae, 4),
            'bad_pixels_ratio': round(bad_pixels_ratio, 2),
            'bad_pixels_2px': round(bad_pixels_2px, 2),
            'valid_count': int(np.sum(mask))
        }

    @staticmethod
    def compare_pipelines(disp_a, disp_c, valid_mask=None):
        """
        Compares output of Pipeline A vs Pipeline C.
        """
        if valid_mask is None:
            mask = np.ones_like(disp_a, dtype=bool)
        else:
            mask = valid_mask

        diff = np.abs(disp_a[mask] - disp_c[mask])

        return {
            'mean_diff': round(float(np.mean(diff)), 4),
            'max_diff': round(float(np.max(diff)), 4),
            'rmse_diff': round(float(np.sqrt(np.mean(diff ** 2))), 4),
            'correlation': round(float(np.corrcoef(disp_a[mask].ravel(), disp_c[mask].ravel())[0, 1]), 4)
        }

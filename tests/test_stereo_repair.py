"""Focused contract tests for the limitation-aware snapshot processing path."""

import base64
import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.realworld.iphone_stereo_processor import IPhoneStereoProcessor
from src.realworld.stereo_artifacts import write_stereo_result
from src.realworld.stereo_contracts import (
    ContractError,
    PixelSize,
    ProcessingDisposition,
    load_calibration_result,
)


class TestStereoRepair(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        width, height = 640, 360
        calibration = {
            "K_main": [[520.0, 0, 320.0], [0, 520.0, 180.0], [0, 0, 1]],
            "D_main": [0, 0, 0, 0, 0],
            "K_ultrawide": [[520.0, 0, 320.0], [0, 520.0, 180.0], [0, 0, 1]],
            "D_ultrawide": [0, 0, 0, 0, 0],
            "R_main_to_ultrawide": np.eye(3).tolist(),
            "T_main_to_ultrawide_m": [0.02, 0, 0],
            "main_native_size": [width, height],
            "ultrawide_native_size": [width, height],
            "lens_model": "pinhole",
        }
        payload = {
            "schema_version": 1,
            "status": "input_limited",
            "reason_code": "TEST_CALIBRATION_LIMITATION",
            "recommended_action": "Capture better calibration evidence.",
            "trusted_calibration": None,
            "diagnostic_candidate": calibration,
        }
        self.calibration_path = self.root / "calibration-result.json"
        self.calibration_path.write_text(json.dumps(payload), encoding="utf-8")
        rng = np.random.default_rng(7)
        left = rng.integers(0, 256, (height, width), dtype=np.uint8)
        left = cv2.GaussianBlur(left, (3, 3), 0)
        right = np.zeros_like(left)
        disparity = 12
        right[:, :-disparity] = left[:, disparity:]
        self.ultrawide = cv2.cvtColor(left, cv2.COLOR_GRAY2BGR)
        self.main = cv2.cvtColor(right, cv2.COLOR_GRAY2BGR)
        self.main_path = self.root / "captured-main.jpg"
        self.ultrawide_path = self.root / "captured-ultrawide.jpg"
        cv2.imwrite(str(self.main_path), self.main)
        cv2.imwrite(str(self.ultrawide_path), self.ultrawide)

    def tearDown(self):
        self.temp.cleanup()

    def test_diagnostic_result_preserves_limitation_and_invalid_nan(self):
        calibration = load_calibration_result(self.calibration_path)
        result = IPhoneStereoProcessor(
            calibration, PixelSize(320, 180)
        ).process(self.main, self.ultrawide)
        self.assertEqual(result.disposition, ProcessingDisposition.DIAGNOSTIC)
        self.assertFalse(result.trusted_depth_eligible)
        self.assertEqual(result.reason_codes, ("TEST_CALIBRATION_LIMITATION",))
        self.assertGreater(float(result.disparity.validity.mean()), 0.1)
        self.assertTrue(np.isnan(result.depth.meters[~result.depth.validity]).all())
        self.assertTrue(
            np.all(result.depth.validity <= result.disparity.validity)
        )

    def test_portrait_capture_is_normalized_to_calibration_native_orientation(self):
        calibration = load_calibration_result(self.calibration_path)
        portrait_main = cv2.rotate(self.main, cv2.ROTATE_90_CLOCKWISE)
        portrait_ultrawide = cv2.rotate(
            self.ultrawide, cv2.ROTATE_90_CLOCKWISE
        )

        result = IPhoneStereoProcessor(
            calibration, PixelSize(320, 180)
        ).process(portrait_main, portrait_ultrawide)

        self.assertEqual(result.rectified.left_ultrawide.shape[:2], (180, 320))
        self.assertGreater(float(result.disparity.validity.mean()), 0.1)

    def test_trusted_calibration_does_not_override_failed_depth_evidence(self):
        payload = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        payload["status"] = "trusted"
        payload["trusted_calibration"] = payload.pop("diagnostic_candidate")
        payload["reason_code"] = None
        trusted_path = self.root / "trusted-calibration-result.json"
        trusted_path.write_text(json.dumps(payload), encoding="utf-8")
        blank = np.zeros_like(self.main)

        result = IPhoneStereoProcessor(
            load_calibration_result(trusted_path), PixelSize(320, 180)
        ).process(blank, blank)

        self.assertFalse(result.trusted_depth_eligible)
        self.assertEqual(result.disposition, ProcessingDisposition.DIAGNOSTIC)
        self.assertIn("DISPARITY_VALIDITY_BELOW_THRESHOLD", result.reason_codes)

    def test_rejected_calibration_returns_diagnostics_without_artifacts(self):
        payload = json.loads(self.calibration_path.read_text(encoding="utf-8"))
        payload["status"] = "rejected"
        payload["diagnostic_candidate"] = None
        payload["reason_code"] = "CALIBRATION_EVIDENCE_REJECTED"
        rejected_path = self.root / "rejected-calibration-result.json"
        rejected_path.write_text(json.dumps(payload), encoding="utf-8")

        result = IPhoneStereoProcessor(
            load_calibration_result(rejected_path)
        ).process(self.main, self.ultrawide)

        self.assertEqual(result.disposition, ProcessingDisposition.REJECTED)
        self.assertIsNone(result.rectified)
        self.assertIsNone(result.disparity)
        self.assertIsNone(result.depth)
        self.assertFalse(result.trusted_depth_eligible)
        output = self.root / "rejected-digest"
        manifest = write_stereo_result(result, output)
        self.assertEqual(manifest["artifacts"], {})
        self.assertIn(
            "CALIBRATION_EVIDENCE_REJECTED",
            (output / "index.html").read_text(encoding="utf-8"),
        )

    def test_artifacts_include_manifest_and_page_local_confidence_control(self):
        calibration = load_calibration_result(self.calibration_path)
        result = IPhoneStereoProcessor(
            calibration, PixelSize(320, 180)
        ).process(self.main, self.ultrawide)
        output = self.root / "digest"
        manifest = write_stereo_result(result, output)
        page = (output / "index.html").read_text(encoding="utf-8")
        self.assertEqual(manifest["camera_order"]["left"], "ultrawide")
        self.assertEqual(manifest["disposition"], "diagnostic")
        self.assertIn("Confidence threshold", page)
        self.assertIn("Inspection only", page)
        self.assertIn("TEST_CALIBRATION_LIMITATION", page)
        self.assertIn("Rectified stereo pair", page)
        self.assertIn('src="rectified_ultrawide.png"', page)
        self.assertIn('src="rectified_main.png"', page)
        self.assertIn("same coordinates as disparity", page)
        self.assertTrue((output / "confidence_u16.png").exists())
        self.assertTrue((output / "disparity_validity.png").exists())

    def test_digest_preserves_and_displays_actual_camera_inputs(self):
        calibration = load_calibration_result(self.calibration_path)
        result = IPhoneStereoProcessor(
            calibration, PixelSize(320, 180)
        ).process(self.main, self.ultrawide)
        output = self.root / "digest"

        manifest = write_stereo_result(
            result,
            output,
            main_input=str(self.main_path),
            ultrawide_input=str(self.ultrawide_path),
        )

        page = (output / "index.html").read_text(encoding="utf-8")
        self.assertTrue((output / "source_main.png").exists())
        self.assertTrue((output / "source_ultrawide.png").exists())
        self.assertIn('src="source_main.png"', page)
        self.assertIn('src="source_ultrawide.png"', page)
        self.assertIn("Actual camera captures", page)
        self.assertIn("source_main.png", manifest["artifacts"])
        self.assertIn("source_ultrawide.png", manifest["artifacts"])

    def test_depth_overlay_reports_metric_depth_at_an_object(self):
        calibration = load_calibration_result(self.calibration_path)
        result = IPhoneStereoProcessor(
            calibration, PixelSize(320, 180)
        ).process(self.main, self.ultrawide)
        output = self.root / "digest"

        write_stereo_result(result, output)

        page = (output / "index.html").read_text(encoding="utf-8")
        display_script = (output / "display_data.js").read_text(encoding="utf-8")
        payload = json.loads(
            display_script.removeprefix("window.STEREO_DISPLAY=").removesuffix(";\n")
        )
        displayed_depth = np.frombuffer(
            base64.b64decode(payload["depth_m_base64"]), dtype="<f4"
        ).reshape(payload["height"], payload["width"])

        np.testing.assert_allclose(
            displayed_depth, result.depth.meters, equal_nan=True
        )
        self.assertIn("Interactive depth overlay", page)
        self.assertIn("Depth overlay opacity", page)
        self.assertIn('id="depthInspector"', page)
        self.assertIn('src="rectified_ultrawide.png"', page)
        self.assertIn('src="depth_map.png"', page)
        self.assertIn("Point at or tap an object", page)

    def test_matrix_only_json_is_not_a_calibration_result(self):
        legacy = self.root / "legacy.json"
        legacy.write_text(json.dumps({"K1": np.eye(3).tolist()}), encoding="utf-8")
        with self.assertRaises(ContractError):
            load_calibration_result(legacy)


if __name__ == "__main__":
    unittest.main()

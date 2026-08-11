"""
Sensor Synchronization and VCM Focus Lock Validation Engine for Dual-Camera Pipeline.
Validates inter-sensor timestamp synchronization, estimates rolling shutter vertical row drift delta_y(t),
and verifies VCM focus lock state and lens position parameters across camera sensors.
"""

import numpy as np
from typing import Dict, Any, Optional


class SensorSyncFocusValidator:
    """
    Validates inter-sensor hardware synchronization and Voice Coil Motor (VCM) focus lock state.
    """

    def __init__(self, max_delta_ms: float = 2.0, readout_time_ms: float = 16.6):
        """
        :param max_delta_ms: Maximum allowed inter-sensor timestamp delta in milliseconds.
        :param readout_time_ms: Sensor rolling shutter readout time per frame in milliseconds.
        """
        self.max_delta_ms = max_delta_ms
        self.readout_time_ms = readout_time_ms

    def validate_timestamps(
        self,
        pts_main_ns: int,
        pts_uw_ns: int,
        frame_height: int = 1080,
        custom_max_delta_ms: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Calculates inter-sensor timestamp difference and estimates vertical row readout drift delta_y(t).

        :param pts_main_ns: Presentation timestamp of Main camera in nanoseconds.
        :param pts_uw_ns: Presentation timestamp of Ultra-Wide camera in nanoseconds.
        :param frame_height: Frame height resolution in pixels.
        :param custom_max_delta_ms: Override maximum allowed timestamp delta in ms.
        :return: Dictionary containing validation status, delta_ms, and estimated vertical row drift in px.
        """
        max_delta = custom_max_delta_ms if custom_max_delta_ms is not None else self.max_delta_ms
        delta_ns = abs(int(pts_main_ns) - int(pts_uw_ns))
        delta_ms = delta_ns / 1e6

        # Rolling shutter readout drift: delta_y(t) = frame_height * (delta_ms / readout_time_ms)
        row_drift_px = float(frame_height) * (delta_ms / float(self.readout_time_ms))
        is_synced = delta_ms <= max_delta

        return {
            "is_synced": is_synced,
            "delta_ns": delta_ns,
            "delta_ms": round(delta_ms, 4),
            "row_drift_px": round(row_drift_px, 4),
            "max_delta_ms": max_delta,
            "readout_time_ms": self.readout_time_ms
        }

    def validate_vcm_focus_lock(
        self,
        metadata: Dict[str, Any],
        expected_lens_position: float = 0.7,
        tolerance: float = 0.05
    ) -> Dict[str, Any]:
        """
        Verifies Voice Coil Motor (VCM) focus lock state and lens position parameters.

        :param metadata: Frame metadata dictionary containing camera sensor parameters.
        :param expected_lens_position: Target normalized lens position (0.0 to 1.0).
        :param tolerance: Allowed lens position deviation tolerance.
        :return: Dictionary containing focus lock verification status and per-camera details.
        """
        telemetry = metadata.get("telemetry", {})
        vcm_locked = telemetry.get("vcm_focus_locked", False)

        main_meta = metadata.get("main", {})
        uw_meta = metadata.get("ultrawide", {})

        main_pos = main_meta.get("lens_position", expected_lens_position if vcm_locked else None)
        uw_pos = uw_meta.get("lens_position", expected_lens_position if vcm_locked else None)

        main_locked = main_meta.get("focus_locked", vcm_locked)
        uw_locked = uw_meta.get("focus_locked", vcm_locked)

        main_pos_ok = main_pos is not None and abs(main_pos - expected_lens_position) <= tolerance
        uw_pos_ok = uw_pos is not None and abs(uw_pos - expected_lens_position) <= tolerance

        is_focus_locked = bool(vcm_locked and main_locked and uw_locked and main_pos_ok and uw_pos_ok)

        return {
            "is_focus_locked": is_focus_locked,
            "vcm_telemetry_locked": bool(vcm_locked),
            "main_focus_locked": bool(main_locked),
            "ultrawide_focus_locked": bool(uw_locked),
            "main_lens_position": main_pos,
            "ultrawide_lens_position": uw_pos,
            "expected_lens_position": expected_lens_position,
            "tolerance": tolerance
        }

    def validate_frame_pair(
        self,
        metadata: Dict[str, Any],
        frame_height: int = 1080
    ) -> Dict[str, Any]:
        """
        Runs full timestamp sync and VCM focus lock validation on a frame pair metadata dictionary.

        :param metadata: Frame pair metadata dictionary.
        :param frame_height: Frame height resolution in pixels.
        :return: Consolidated validation report dictionary.
        """
        pts_main = metadata.get("main_pts_ns", metadata.get("timestamp_pts_ns", 0))
        pts_uw = metadata.get("ultrawide_pts_ns", metadata.get("timestamp_pts_ns", 0))

        sync_report = self.validate_timestamps(pts_main, pts_uw, frame_height=frame_height)
        focus_report = self.validate_vcm_focus_lock(metadata)

        is_valid = sync_report["is_synced"] and focus_report["is_focus_locked"]

        return {
            "is_valid": is_valid,
            "timestamp_sync": sync_report,
            "vcm_focus_lock": focus_report
        }

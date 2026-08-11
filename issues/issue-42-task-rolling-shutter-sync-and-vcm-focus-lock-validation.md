# Issue #42: Task - Rolling Shutter Readout & Hardware Sensor Synchronization Validation

## Question
How do we validate and enforce inter-sensor timestamp synchronization and VCM focus locking in the iOS AVFoundation capture pipeline to eliminate dynamic vertical row drift $\delta y(t)$?

## Resolution
Implemented `SensorSyncFocusValidator` in `src/realworld/sync_focus_validator.py` and updated iOS Swift capture telemetry (`MultiCamSessionManager.swift`). Calculates inter-sensor timestamp deltas $\Delta t_{\text{ms}}$, estimates rolling shutter vertical row readout drift $\delta y(t) = \text{height} \cdot \frac{\Delta t}{T_{\text{readout}}}$, and validates VCM focus lock parameters (`lens_position` = 0.7). Verified with `tests/test_sync_focus_validator.py` and iOS simulator Swift compilation (**56/56 tests passing**).

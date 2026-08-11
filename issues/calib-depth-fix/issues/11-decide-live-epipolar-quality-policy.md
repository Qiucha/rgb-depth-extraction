Type: grilling
Status: resolved
Blocked by: 10

## Question

What live epipolar-accuracy contract should gate Stereo Processing Results so that in-bounds remap coverage cannot be mistaken for successful Stereo Rectification? Decide the measurement, threshold, diagnostic-versus-rejected behavior, and whether a systematic vertical residual may ever be compensated temporarily instead of requiring a new trusted calibration.

## Answer

Keep the operational path simple and do not add the proposed multi-tier live feature gate, vertical-compensation mode, new Processing Extent states, or extra result branches now.

The canonical path remains:

1. Apply Stereo Rectification to the raw Main and Ultra-Wide Stereo Pair.
2. Pass the resulting `RectifiedStereoPair` directly to Epipolar Matching.
3. Convert its hard-valid Disparity Map to metric depth using the same rectified focal length, baseline, and disparity offset.
4. Display the Rectified Stereo Pair immediately before disparity/depth so the human compares artifacts in the correct coordinates.
5. Preserve the Calibration Result limitation as the trust decision. With the current input-limited calibration, outputs remain diagnostic and cannot establish Trusted Depth Eligibility.

The shared processor already implements this sequence: `rectifier.rectify(...)` returns the value passed to `sliding_window.compute(rectified)`, and depth is derived from that disparity plus `rectified.geometry`.

The latest real capture, `snapshot_20260811_141309`, was replayed through this exact path. Rectified images, disparity, and depth all use 640×360 coordinates. It produced 33.65% hard-valid disparity, 33.63% valid depth, median valid disparity 42.21 px, and median diagnostic depth 0.46 m. Visual inspection of `depth_overlay.png` confirms that the stored disparity/depth is spatially registered to the rectified cup-and-laptop view. It remains untrusted because the active Calibration Result is input-limited with `CALIBRATION_RATIONAL_MODEL_FAILS_HELDOUT_VALIDATION`.

Earlier recommendations for a feature-based per-capture gate were superseded by the user's instruction to keep the pipeline as simple as possible. Their provisional glossary entries were removed rather than leaving an unimplemented interface in the domain model.

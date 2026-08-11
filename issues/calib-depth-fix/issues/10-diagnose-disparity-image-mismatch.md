Type: task
Status: resolved
Blocked by: 09

## Question

Determine why the latest Disparity Map does not visually correspond to the actual Main and Ultra-Wide captures shown in the Digest Dashboard. Distinguish presentation coordinate-frame mismatch from invalid Stereo Rectification, wrong camera ordering/disparity sign, and failed Epipolar Matching using reproducible evidence from the exact stored Stereo Processing Result.

## Answer

Two independent problems were confirmed on a frozen copy of the exact `snapshot_20260811_140937` Stereo Processing Result.

1. **The Digest Dashboard compared different coordinate frames.** The actual captures are raw 1920×1080 camera coordinates; the Disparity Map is 640×360 in the rectified physical-left Ultra-Wide coordinate frame. A plain resize of the raw Ultra-Wide capture differs from `rectified_ultrawide.png` by 86.42 intensity levels out of 255 on average. The page displayed the raw captures and the Disparity Map but omitted the rectified reference images, making a correct rectification warp look like a mismatched map. The digest now displays the Rectified Stereo Pair and states that the rectified Ultra-Wide view is the reference for disparity.

2. **The current diagnostic calibration does not rectify this scene accurately enough for trusted matching.** Independent SIFT correspondences inside the hard-valid disparity region show a systematic signed vertical residual of -6.12 px (6.60 px absolute p90), well beyond the 2 px epipolar target. The stored result is already `diagnostic`, with `CALIBRATION_RATIONAL_MODEL_FAILS_HELDOUT_VALIDATION`, and only 37.85% hard-valid disparity coverage. Because the matcher searches horizontally, it cannot model this vertical residual; its median disagreement with independent feature disparity is 2.11 px.

The following suspected causes were falsified:

- **Wrong order/sign:** warping the rectified Main view with the declared `x_left - x_right` disparity gives ZNCC 0.9537, versus 0.5109 for the opposite sign and 0.5258 for zero disparity.
- **Stale/mixed artifacts:** both decoded digest sources equal their manifest input images, their manifest hashes match, and the source/disparity/manifest artifacts share the same generation timestamp.
- **Total matcher hallucination:** the declared disparity reconstructs the rectified correspondence view well on hard-valid pixels, although calibration error prevents trusted metric interpretation.

The existing 99.69% Rectification Validity coverage is not an epipolar-accuracy measurement; it only proves that remap coordinates sample inside both source images. Treating that as sufficient rectification evidence allowed a diagnostic map with a large vertical residual to remain viewable.

### Implemented presentation correction

Added the Rectified Stereo Pair to every digest with explicit `Ultra-Wide · physical left` and `Main · physical right` labels and a statement that disparity uses the rectified Ultra-Wide coordinates. A regression test verifies those references, the persistent server was restarted, and the current live digest was regenerated.

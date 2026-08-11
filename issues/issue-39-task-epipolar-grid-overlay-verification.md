# Issue #39: Task - 20px Epipolar Grid Overlay Verification Engine & Digest Visualizer

## Question
How do we implement a 20px horizontal epipolar grid line overlay across `rectified_main` and `rectified_ultrawide` images in `digest_builder.py` and the HTML dashboard to visually verify row-by-row alignment?

## Resolution
Implemented `epipolar_grid.jpg` generation in `src/realworld/digest_builder.py` concatenating `rect_main` and `rect_uw` side-by-side with alternating 20px red/green horizontal scanlines. Embedded a new `📐 20px Epipolar Grid` viewport tab in the interactive HTML visual digest dashboard (`output_dir/index.html`). Verified with unit test suite in `tests/test_epipolar_grid.py` (**48/48 tests passing**).

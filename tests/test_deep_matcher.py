import unittest
import numpy as np
import os

from src.realworld.deep_matcher import (
    BaseStereoMatcher,
    ClassicalSlidingWindowAdapter,
    DeepDisparityMatcherAdapter,
    get_stereo_matcher
)


class TestDeepDisparityMatcher(unittest.TestCase):
    def setUp(self):
        self.height = 120
        self.width = 160
        np.random.seed(42)
        self.left_img = np.random.randint(0, 255, (self.height, self.width), dtype=np.uint8)
        self.right_img = np.random.randint(0, 255, (self.height, self.width), dtype=np.uint8)

    def test_classical_sliding_window_adapter(self):
        adapter = ClassicalSlidingWindowAdapter(window_size=5, max_disparity=16, metric="zncc")
        disp, mask = adapter.compute_disparity(self.left_img, self.right_img)
        self.assertEqual(disp.shape, (self.height, self.width))
        self.assertEqual(mask.shape, (self.height, self.width))
        self.assertEqual(disp.dtype, np.float32)
        self.assertEqual(mask.dtype, bool)

    def test_deep_disparity_matcher_fallback(self):
        # Without model weights file, should fall back gracefully to classical matcher
        adapter = DeepDisparityMatcherAdapter(model_type="cre_stereo", model_path="missing_model.pth")
        disp, mask = adapter.compute_disparity(self.left_img, self.right_img)
        self.assertEqual(disp.shape, (self.height, self.width))
        self.assertEqual(mask.shape, (self.height, self.width))
        self.assertEqual(disp.dtype, np.float32)
        self.assertEqual(mask.dtype, bool)

    def test_raft_stereo_sequence_output_processing(self):
        adapter = DeepDisparityMatcherAdapter(model_type="raft_stereo")
        # Test processing raw multi-stage prediction list output (like RAFT-Stereo iterations)
        seq_output = [
            np.ones((1, 1, self.height, self.width), dtype=np.float32) * 5.0,
            np.ones((1, 1, self.height, self.width), dtype=np.float32) * 12.0
        ]
        disp, mask = adapter._process_model_output(seq_output, (self.height, self.width))
        self.assertEqual(disp.shape, (self.height, self.width))
        self.assertTrue(np.allclose(disp, 12.0))
        self.assertTrue(np.all(mask))

    def test_factory_function(self):
        m1 = get_stereo_matcher("sliding_window", max_disparity=16)
        self.assertIsInstance(m1, BaseStereoMatcher)
        self.assertIsInstance(m1, ClassicalSlidingWindowAdapter)

        m_zncc = get_stereo_matcher("zncc", max_disparity=16)
        self.assertEqual(m_zncc.metric, "zncc")

        m2 = get_stereo_matcher("cre_stereo")
        self.assertIsInstance(m2, BaseStereoMatcher)
        self.assertIsInstance(m2, DeepDisparityMatcherAdapter)

        m3 = get_stereo_matcher("raft_stereo")
        self.assertIsInstance(m3, BaseStereoMatcher)
        self.assertIsInstance(m3, DeepDisparityMatcherAdapter)

    def test_unknown_matcher_raises_value_error(self):
        with self.assertRaises(ValueError):
            get_stereo_matcher("invalid_matcher")


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing

from src.cube_projection import StereoCameraRig, RoomEnvironment, StereoRoomVisualizer


class TestIntegratedStereoDepth(unittest.TestCase):
    def test_integrated_depth_extraction_untextured(self):
        stereo_rig = StereoCameraRig(focal_length=400.0, baseline=0.2, width=320, height=240, pos_x=0.0, pos_y=0.0, pos_z=-4.0)
        room = RoomEnvironment()

        visualizer = StereoRoomVisualizer(stereo_rig=stereo_rig, room=room, enable_dots=False, layout="3col")
        visualizer.update()

        self.assertIsNotNone(visualizer.depth_im)
        depth_data = visualizer.depth_im.get_array()
        self.assertEqual(depth_data.shape, (240, 320))
        self.assertTrue(np.max(depth_data) > 0.0)

        # Verify Ground Truth depth rendering
        self.assertIsNotNone(visualizer.gt_depth_im)
        gt_data = visualizer.gt_depth_im.get_array()
        self.assertEqual(gt_data.shape, (240, 320))
        self.assertTrue(np.max(gt_data) > 0.0)

    def test_integrated_depth_extraction_textured_3col(self):
        stereo_rig = StereoCameraRig(focal_length=400.0, baseline=0.2, width=320, height=240, pos_x=0.0, pos_y=0.0, pos_z=-4.0)
        room = RoomEnvironment()

        visualizer = StereoRoomVisualizer(stereo_rig=stereo_rig, room=room, enable_dots=True, layout="3col")
        visualizer.update()

        self.assertIsNotNone(visualizer.depth_im)
        depth_data = visualizer.depth_im.get_array()
        self.assertEqual(depth_data.shape, (240, 320))
        self.assertTrue(np.max(depth_data) > 0.0)

        # Verify Ground Truth depth rendering
        self.assertIsNotNone(visualizer.gt_depth_im)
        gt_data = visualizer.gt_depth_im.get_array()
        self.assertEqual(gt_data.shape, (240, 320))
        self.assertTrue(np.max(gt_data) > 0.0)

    def test_integrated_depth_extraction_textured_3row(self):
        stereo_rig = StereoCameraRig(focal_length=400.0, baseline=0.2, width=320, height=240, pos_x=0.0, pos_y=0.0, pos_z=-4.0)
        room = RoomEnvironment()

        visualizer = StereoRoomVisualizer(stereo_rig=stereo_rig, room=room, enable_dots=True, layout="3row")
        visualizer.update()

        self.assertIsNotNone(visualizer.depth_im)
        depth_data = visualizer.depth_im.get_array()
        self.assertEqual(depth_data.shape, (240, 320))
        self.assertTrue(np.max(depth_data) > 0.0)

        self.assertIsNotNone(visualizer.gt_depth_im)
        gt_data = visualizer.gt_depth_im.get_array()
        self.assertEqual(gt_data.shape, (240, 320))
        self.assertTrue(np.max(gt_data) > 0.0)


if __name__ == '__main__':
    unittest.main()

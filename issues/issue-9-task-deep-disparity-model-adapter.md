# Issue #9: Task - Deep Disparity Model Integration Adapter Interface

**Labels**: `wayfinder:task`  
**Parent**: [Map](file:///Users/q/Documents/Projects/robotics_neo/issues/map.md)  
**Status**: Closed  
**Assignee**: Main Agent  
**Blocks**: None  
**Blocked by**: [Issue #3](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-3-prototype-realworld-pipeline-architecture.md)

## Question

How should the Python stereo matching interface be refactored to support pluggable secondary deep matching backends (CREStereo, AnyStereo, RAFT-Stereo) alongside classical 1D Epipolar Sliding Window block matching within `src/realworld/pipeline.py`?

## Resolution

- Design abstract `BaseStereoMatcher` adapter interface.
- Implement `ClassicalSlidingWindowAdapter` wrapping `SlidingWindowMatcher`.
- Implement `DeepDisparityMatcherAdapter` supporting CREStereo / RAFT-Stereo deep disparity models with ONNX/Torch runtime loading and safe fallback to classical matching when model weights are absent.
- Update `run_realworld_pipeline()` to support `matcher_type` configuration.
- Verify with TDD unit tests in `tests/test_deep_matcher.py` and integration tests in `tests/test_realworld_pipeline.py`.

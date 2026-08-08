# Issue #12: Task - Clean Up Test Output Folders & Organize Live Capture Hierarchy

**Labels**: `wayfinder:task`  
**Parent**: [Map](file:///Users/q/Documents/Projects/robotics_neo/issues/map.md)  
**Status**: Closed  
**Assignee**: Main Agent  
**Blocks**: None  
**Blocked by**: None  

## Question

How should temporary test output directories (`digest_test_realworld`, `digest_test_realworld_deep`, `.trash`) be cleaned up and the live iPhone capture hierarchy (`data/live_iphone_captures/seq_YYYYMMDD_HHMMSS/`) be structured for crystal-clear data separation?

## Resolution

- Clean up temporary test run directories from project root.
- Add `.gitignore` rules for temporary digest outputs (`digest_test_*`, `digest_live_*`).
- Structure live iPhone capture output under `data/live_captures/` with timestamped sequence IDs (`seq_YYYYMMDD_HHMMSS`).

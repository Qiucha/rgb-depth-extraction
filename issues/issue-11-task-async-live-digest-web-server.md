# Issue #11: Task - Asynchronous Live Visual Digest Streaming & Immediate Web Server Launch

**Labels**: `wayfinder:task`  
**Parent**: [Map](file:///Users/q/Documents/Projects/robotics_neo/issues/map.md)  
**Status**: Closed  
**Assignee**: Main Agent  
**Blocks**: None  
**Blocked by**: [Issue #10](file:///Users/q/Documents/Projects/robotics_neo/issues/issue-10-task-fix-swift-color-and-dual-cam-extraction.md)  

## Question

How should `IOSBridgeServer` in `src/realworld/ios_bridge/server.py` be refactored to launch the live digest web server immediately upon client connection and run live depth extraction asynchronously without blocking the event loop?

## Resolution

- Initialize digest folder structure and baseline `index.html` on connection start.
- Launch HTTP digest server (`serve_digest`) immediately upon first connection on port 8080.
- Offload real-time stereo block matching and digest HTML updates to a non-blocking background thread worker (`asyncio.to_thread` / `run_in_executor`).

# Issue #38: Task - Replace Interactive Toggle Switches with Pipeline Filter Telemetry Badges
**Assignee:** Antigravity


## Question

How can we replace non-functional interactive toggle switches in `src/realworld/digest_builder.py` with clean static pipeline filter telemetry badges?

## Context & Requirements

1. **Remove Switch Elements**:
   - In `src/realworld/digest_builder.py`, remove `<label class="switch"><input type="checkbox"...></label>` toggle inputs from the sidebar HTML.

2. **Clean Telemetry Status Indicators**:
   - Add clean `ACTIVE` status badges (e.g. `<span class="badge-active">ACTIVE</span>`) next to each filter name (Guided Bilateral, WLS Edge, Speckle Removal, Left-Right Check, 3x3 Median).

3. **Verification**:
   - Run unit and integration tests (`tests/test_snapshot_server.py` & full test suite) verifying clean HTML generation.

## Resolution

1. **Replaced Switch Elements with Active Status Badges**: Removed interactive `<label class="switch">` checkbox elements from `src/realworld/digest_builder.py` and replaced them with clean `<span class="badge-active">ACTIVE</span>` status badges.
2. **Updated Panel Header & JS**: Renamed header to `Pipeline Filter Status` and removed obsolete `updateFilterStates()` JS function.
3. **Verification**: Executed full unit and integration test suite (`.venv/bin/python3 -m unittest discover -s tests`), passing all **47/47 tests cleanly**.


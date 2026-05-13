---
name: tdd-cycle
description: Enforces RED → GREEN → REFACTOR for every new feature. Invoke before writing any new module or function in this project.
---

# TDD Cycle

Every new behaviour in this project follows **one** test-driven cycle. No code lands on a feature branch without a test that previously failed.

## The cycle

1. **RED — Write the failing test first.**
   - One concern per test. If you find yourself writing `and` in the test name, split it.
   - The test must reference the public API as you wish it existed — type signatures, return shape, error cases.
   - Run it. Confirm the failure is the expected kind: `ImportError`, `NameError`, `AttributeError`, or the assertion you wrote.
   - If the test passes immediately, the assertion is too weak — strengthen it before moving on.

2. **GREEN — Implement the minimum to pass.**
   - Write only what the failing test demands. Resist anticipating future tests.
   - Re-run the suite. Confirm only the previously-failing test went green; nothing else broke.
   - If something else broke, you implemented too much. Revert and narrow the change.

3. **REFACTOR — Tidy with the tests still green.**
   - Rename, extract, simplify. The tests stay passing the whole time.
   - No new behaviour during refactor — that needs its own RED→GREEN cycle.

4. **Commit.**
   - One cycle = one commit (see `commit-discipline`).

## Conventions for this repo

- Tests live in `tests/`, mirror the module name: `tests/test_squeezers.py` tests `squeezers.py`.
- Shared fixtures are in `tests/conftest.py` (already populated with `blank_face_img`, `mock_face`, `mock_embedder`). Reuse them — do not redefine.
- Mock heavy external deps (`insightface`, `onnxruntime`) so tests run in milliseconds. The real model is exercised only via manual smoke tests once the Gradio app is up.
- The per-step plan in `docs/superpowers/plans/2026-05-13-...md` already prescribes the test code for Eyal's Tasks 2–7 — copy those tests verbatim into your RED step, don't reinvent them.

## Do not

- Skip RED. "I'll add the test after" defeats the point.
- Write multiple new tests in one commit. The cycle is one test at a time.
- Implement helpers that aren't required by the current failing test ("YAGNI for the cycle scope").

## See also

- `commit-discipline` — what to do after step 4.
- `branch-discipline` — where the commit lands.
- `docs/superpowers/plans/2026-05-13-feature-squeezing-demo.md` — pre-written tests for each task.

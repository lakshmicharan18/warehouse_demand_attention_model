# Debugging Evidence

## Initial issue

During dataset-milestone validation, `pytest` could not run because it was not installed in the active environment.

## Hypothesis

The project dependencies were listed in `requirements.txt`, but the current environment had not installed them.

## Diagnostic step

An attempted `pytest -q` returned `pytest: command not found`; an environment check confirmed NumPy was available but pytest was absent.

## Correction

Installed the existing project requirements with `python -m pip install -r requirements.txt`.

## Verification

The dataset tests then ran successfully. Subsequent full-suite validation reached 37 passing tests.

## What I learned

This was an environment/reproducibility issue, not a mathematical-model bug. A requirements file alone is insufficient unless the active environment is confirmed to have installed it.

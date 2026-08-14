"""Check that the CI test groups cover every file in `tests/`.

`.github/workflows/run_tests.yml` splits `tests/` across several parallel jobs by
naming files explicitly, rather than globbing, so that the slow files can be given a
job each. That means a newly-added test file would otherwise silently never run in
CI. This is run both by `pre-commit` and by the workflow's own `lint` job.
"""

import itertools
import pathlib
import sys

import yaml


ROOT = pathlib.Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github/workflows/run_tests.yml"
TESTS = ROOT / "tests"
# Where to put a new test file absent a reason to do otherwise.
DEFAULT_GROUP = "core"


def expand(matrix: dict) -> list[dict]:
    """Expand a GitHub Actions matrix, following the documented `include` rules.

    An `include` entry is merged into every combination it can be merged into
    without overwriting, and only becomes a combination of its own if it fits
    nowhere. Expanding properly (rather than just reading the `include` list) is
    what lets this script notice a matrix that has collapsed into fewer jobs than
    intended.
    """
    keys = [k for k in matrix if k != "include"]
    combinations = [
        dict(zip(keys, values))
        for values in itertools.product(*(matrix[k] for k in keys))
    ]
    for entry in matrix.get("include", []):
        merged = False
        for combination in combinations:
            if all(combination.get(k) == v for k, v in entry.items() if k in keys):
                combination.update({k: v for k, v in entry.items() if k not in keys})
                merged = True
        if not merged:
            combinations.append(dict(entry))
    return combinations


def main() -> int:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    combinations = expand(workflow["jobs"]["test"]["strategy"]["matrix"])
    relative = WORKFLOW.relative_to(ROOT)
    errors = []

    groups = [combination.get("group", "<unnamed>") for combination in combinations]
    if len(set(groups)) != len(groups):
        errors.append(
            f"test groups are not distinct ({', '.join(map(str, groups))}); an "
            "`include` entry is probably being merged into the wrong combination"
        )

    assigned: dict[str, str] = {}
    for combination, group in zip(combinations, groups):
        files = str(combination.get("files", "")).split()
        if len(files) == 0:
            errors.append(f"group '{group}' runs no test files")
        for name in files:
            if name in assigned:
                errors.append(f"{name} is in both '{assigned[name]}' and '{group}'")
            assigned[name] = group

    on_disk = {path.name for path in TESTS.glob("test_*.py")}
    for name in sorted(on_disk - set(assigned)):
        errors.append(
            f"tests/{name} is not in any CI test group, so it would never run. Add "
            f"it to the '{DEFAULT_GROUP}' group in {relative}, or give it a group of "
            "its own if it is slow enough to bound the whole run."
        )
    for name in sorted(set(assigned) - on_disk):
        errors.append(
            f"tests/{name} is in group '{assigned[name]}' in {relative} but does not "
            "exist"
        )

    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if len(errors) > 0:
        print(
            f"\n{len(combinations)} test job(s) defined: {', '.join(map(str, groups))}",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

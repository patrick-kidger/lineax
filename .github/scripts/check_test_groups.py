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


def expand(matrix: dict) -> tuple[list[dict], list[str]]:
    """Expand a GitHub Actions matrix, following the documented `include` rules.

    An `include` entry is merged into every combination it can be merged into
    without overwriting an original matrix value, and only becomes a combination of
    its own if it fits nowhere. Values added by an earlier `include` entry *can* be
    overwritten by a later one, which silently collapses jobs into one another; such
    overwrites are returned as errors. Expanding properly (rather than just reading
    the `include` list) is what lets this script notice a matrix that has collapsed
    into fewer jobs than intended.
    """
    keys = [k for k in matrix if k != "include"]
    combinations = [
        dict(zip(keys, values))
        for values in itertools.product(*(matrix[k] for k in keys))
    ]
    errors = []
    for entry in matrix.get("include", []):
        merged = False
        for combination in combinations:
            if all(combination.get(k) == v for k, v in entry.items() if k in keys):
                for k, v in entry.items():
                    if k not in keys and k in combination and combination[k] != v:
                        errors.append(
                            f"`include` entry {entry} overwrites {k}={combination[k]!r}"
                            " set by an earlier `include` entry; make the key that "
                            "distinguishes them a matrix dimension"
                        )
                combination.update({k: v for k, v in entry.items() if k not in keys})
                merged = True
        if not merged:
            combinations.append(dict(entry))
    return combinations, errors


def main() -> int:
    workflow = yaml.safe_load(WORKFLOW.read_text())
    combinations, errors = expand(workflow["jobs"]["test"]["strategy"]["matrix"])
    relative = WORKFLOW.relative_to(ROOT)

    # The test matrix is the cross product of the groups with any other dimensions
    # (e.g. the oldest/newest dependency variants). Each such variant must run every
    # test file exactly once, so check each variant on its own.
    variants: dict[str, list[dict]] = {}
    for combination in combinations:
        variant = ", ".join(
            f"{k}={v}"
            for k, v in sorted(combination.items())
            if k not in ("group", "files")
        )
        variants.setdefault(variant, []).append(combination)

    on_disk = {path.name for path in TESTS.glob("test_*.py")}
    for variant, members in variants.items():
        where = f" [{variant}]" if len(variants) > 1 else ""
        groups = [member.get("group", "<unnamed>") for member in members]
        if len(set(groups)) != len(groups):
            errors.append(
                f"test groups{where} are not distinct ({', '.join(map(str, groups))}); "
                "an `include` entry is probably being merged into the wrong combination"
            )

        assigned: dict[str, str] = {}
        for member, group in zip(members, groups):
            files = str(member.get("files", "")).split()
            if len(files) == 0:
                errors.append(f"group '{group}'{where} runs no test files")
            for name in files:
                if name in assigned:
                    errors.append(
                        f"{name} is in both '{assigned[name]}' and '{group}'{where}"
                    )
                assigned[name] = group

        for name in sorted(on_disk - set(assigned)):
            errors.append(
                f"tests/{name} is not in any CI test group{where}, so it would never "
                f"run. Add it to the '{DEFAULT_GROUP}' group in {relative}, or give it "
                "a group of its own if it is slow enough to bound the whole run."
            )
        for name in sorted(set(assigned) - on_disk):
            errors.append(
                f"tests/{name} is in group '{assigned[name]}'{where} in {relative} but "
                "does not exist"
            )

    errors = list(dict.fromkeys(errors))  # The same overwrite hits every group.
    for error in errors:
        print(f"error: {error}", file=sys.stderr)
    if len(errors) > 0:
        print(
            f"\n{len(combinations)} test job(s) defined: "
            + "; ".join(
                f"[{variant}] {', '.join(str(m.get('group')) for m in members)}"
                for variant, members in variants.items()
            ),
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

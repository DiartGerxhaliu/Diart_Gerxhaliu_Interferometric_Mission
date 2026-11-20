"""ESA SpOC-3 interferometric mission configurations for all three levels."""

import importlib.util
from pathlib import Path


REFERENCE_DIR = Path(__file__).resolve().parents[2] / "reference"

CASES = {
    "easy": {
        "problem_id": "tiny-observatory-in-dro",
        "reference_file": (
            "spoc-3-interferometric-mission-tiny-observatory-in-dro-1711466574922.py"
        ),
        "n_sat": 5,
        "grid_size": 11,
        "ic": [
            0.896508460944940632764,
            0.0,
            0.0,
            0.000000000000013951082,
            0.474817948848534454598,
            0.0,
        ],
        "period": 2.6905181697222775,
    },
    "medium": {
        "problem_id": "large-observatory-in-dro",
        "reference_file": (
            "spoc-3-interferometric-mission-large-observatory-in-dro-1711466585914.py"
        ),
        "n_sat": 40,
        "grid_size": 21,
        "ic": [
            0.896508460944940632764,
            0.0,
            0.0,
            0.000000000000013951082,
            0.474817948848534454598,
            0.0,
        ],
        "period": 2.6905181697222775,
    },
    "hard": {
        "problem_id": "large-observatory-in-halo",
        "reference_file": "esa_reference.py",
        "n_sat": 40,
        "grid_size": 21,
        "ic": [
            1.0829551779304256,
            -6.9232801936027592e-27,
            -2.0231744561698364e-01,
            9.7888791827480806e-15,
            -2.0102644884016105e-01,
            2.4744866465838825e-14,
        ],
        "period": 2.383491010514447,
    },
}

CASE_ORDER = ("easy", "medium", "hard")
CHALLENGE_ID = "spoc-3-interferometric-mission"
N_OBSERVATIONS = 3
MU = 0.01215058560962404
SCALING_FACTOR = 1e-4
INFLATION_FACTOR = 1.23

_LOADED_PROBLEMS = {}


def get_case(case_id):
    try:
        return CASES[case_id]
    except KeyError as exc:
        choices = ", ".join(CASE_ORDER)
        raise ValueError(f"Unknown case {case_id!r}; choose one of: {choices}.") from exc


def _load_reference_udp(case_id):
    """Load the official module-level UDP. Filenames are not valid imports."""
    spec = get_case(case_id)
    path = REFERENCE_DIR / spec["reference_file"]
    if not path.is_file():
        raise FileNotFoundError(f"Missing {case_id} evaluator at {path}.")

    import heyoka

    if heyoka.__version__ != "3.2.0":
        raise RuntimeError(
            f"The ESA evaluators require heyoka==3.2.0; found {heyoka.__version__}."
        )

    module_name = f"_esa_official_{case_id}"
    loader_spec = importlib.util.spec_from_file_location(module_name, path)
    if loader_spec is None or loader_spec.loader is None:
        raise ImportError(f"Cannot load {case_id} evaluator from {path}.")
    module = importlib.util.module_from_spec(loader_spec)
    loader_spec.loader.exec_module(module)
    return module.udp


def make_reference_problem(case_id):
    """Return that case's official ESA UDP. Call only in a CPU-reference process."""
    if case_id not in _LOADED_PROBLEMS:
        udp = _load_reference_udp(case_id)
        spec = get_case(case_id)
        if udp.n_sat != spec["n_sat"] or udp.grid_size != spec["grid_size"]:
            raise RuntimeError(
                f"{case_id} evaluator has n_sat={udp.n_sat}, "
                f"grid_size={udp.grid_size}; expected n_sat={spec['n_sat']}, "
                f"grid_size={spec['grid_size']}."
            )
        _LOADED_PROBLEMS[case_id] = udp
    return _LOADED_PROBLEMS[case_id]

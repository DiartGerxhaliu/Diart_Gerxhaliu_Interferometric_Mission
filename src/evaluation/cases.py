"""ESA SpOC-3 interferometric mission configurations for all three levels."""

CASES = {
    "easy": {
        "problem_id": "tiny-observatory-in-dro",
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


def get_case(case_id):
    try:
        return CASES[case_id]
    except KeyError as exc:
        choices = ", ".join(CASE_ORDER)
        raise ValueError(f"Unknown case {case_id!r}; choose one of: {choices}.") from exc


def make_reference_problem(case_id):
    """Create the chosen ESA UDP; call only in a CPU-reference process."""
    from reference.esa_reference import orbital_golomb_array, udp

    if case_id == "hard":
        return udp

    spec = get_case(case_id)
    return orbital_golomb_array(
        n_sat=spec["n_sat"],
        ic=spec["ic"],
        T=spec["period"] * (N_OBSERVATIONS - 1),
        grid_size=spec["grid_size"],
        scaling_factor=SCALING_FACTOR,
        n_meas=N_OBSERVATIONS,
        inflation_factor=INFLATION_FACTOR,
        mu=MU,
        verbose=False,
    )

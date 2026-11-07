"""CPU scoring helpers for the documented and evaluator-code objectives."""

import numpy as np
import scipy.signal


AGGREGATIONS = ("max", "min")


def observation_scores(x, udp):
    """Return the summed XY/XZ/YZ fill factor at each observation."""
    x = np.asarray(x, dtype=np.float64)
    n_sat = udp.n_sat
    if x.shape != (6 * n_sat,) or not np.isfinite(x).all():
        raise ValueError(f"Expected {6 * n_sat} finite decision variables.")

    perturbations = x.reshape(6, n_sat).T
    scores = []
    for index, stm in enumerate(udp.stms):
        relative = (stm @ (perturbations * udp.scaling_factor).T).T[:, :3]
        points = relative / udp.scaling_factor
        if index:
            points = points / udp.inflation_factor

        valid = (points > -1).all(axis=1) & (points < 1).all(axis=1)
        points = points[valid]
        positions = (
            points * udp.grid_size / 2 + int(udp.grid_size / 2)
        ).astype(int)

        occupancy = np.zeros((udp.grid_size,) * 3, dtype=np.float64)
        for i, j, k in positions:
            occupancy[i, j, k] = 1

        projections = (
            occupancy.max(axis=2),
            occupancy.max(axis=1),
            occupancy.max(axis=0),
        )
        plane_scores = []
        for projection in projections:
            correlation = scipy.signal.correlate(
                projection, projection, mode="full"
            )
            correlation[np.abs(correlation) < 1e-8] = 0
            plane_scores.append(np.count_nonzero(correlation) / correlation.size)
        scores.append(sum(plane_scores))
    return np.asarray(scores, dtype=np.float64)


def aggregate_objective(scores, aggregation):
    if aggregation not in AGGREGATIONS:
        raise ValueError(f"aggregation must be one of {AGGREGATIONS}.")
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("scores must be a non-empty 1D array.")
    return -float(values.max() if aggregation == "max" else values.min())


def cpu_objective(x, udp, aggregation="max"):
    """Evaluate with requested aggregation; min can directly use ESA's UDP."""
    if aggregation == "min":
        return float(udp.fitness(np.asarray(x, dtype=np.float64).tolist())[0])
    return aggregate_objective(observation_scores(x, udp), aggregation)

"""Feasibility helpers for evaluator observation-volume bounds."""

import numpy as np


BOUND_MARGIN = 1e-9


def observation_position_maps(stms, inflation_factor):
    """Return each observation's 3x6 map to evaluator-space positions."""
    return np.stack(
        [
            np.asarray(stm[:3, :], dtype=np.float64)
            / (inflation_factor if index else 1.0)
            for index, stm in enumerate(stms)
        ]
    )


def enforce_observation_bounds(candidates, position_maps, n_sat):
    """Scale each satellite state to keep it strictly inside every observation."""
    values = np.asarray(candidates)
    if values.dtype != np.float64 or not values.flags.writeable:
        raise ValueError("Candidates must be writable float64 arrays.")
    if values.ndim not in (1, 2) or values.shape[-1] != 6 * n_sat:
        raise ValueError("Candidates must have shape (6*n_sat,) or (batch, 6*n_sat).")

    rows = values.reshape(-1, 6 * n_sat)
    states = rows.reshape(-1, 6, n_sat).transpose(0, 2, 1)
    maximum_positions = np.zeros((len(rows), n_sat), dtype=np.float64)
    for position_map in position_maps:
        positions = np.einsum("ij,bnj->bni", position_map, states)
        maximum_positions = np.maximum(
            maximum_positions, np.max(np.abs(positions), axis=-1)
        )

    scale = np.minimum(
        1.0,
        (1.0 - BOUND_MARGIN) / np.maximum(maximum_positions, 1e-300),
    )
    states *= scale[..., None]
    return values


def is_inside_observation_bounds(candidate, position_maps, n_sat):
    """Check that every satellite is strictly inside all mapped observation bounds."""
    values = np.asarray(candidate, dtype=np.float64)
    if values.shape != (6 * n_sat,):
        raise ValueError(f"Candidate must contain exactly {6 * n_sat} values.")
    states = values.reshape(6, n_sat).T
    for position_map in position_maps:
        positions = np.einsum("nj,ij->ni", states, position_map)
        if not np.all(np.abs(positions) < 1.0):
            return False
    return True

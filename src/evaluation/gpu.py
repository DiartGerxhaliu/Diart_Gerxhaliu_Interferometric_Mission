
import torch


def autocorrelation_fill(projection):
    """
    Calculate the fraction of nonzero cells in the full
    2D autocorrelation of a projection.
    """
    grid_size = projection.shape[-1]
    output_size = 2 * grid_size - 1

    spectrum = torch.fft.rfft2(
        projection,
        s=(output_size, output_size)
    )

    correlation = torch.fft.irfft2(
        spectrum * spectrum.conj(),
        s=(output_size, output_size)
    )

    correlation = torch.fft.fftshift(
        correlation,
        dim=(-2, -1)
    )

    nonzero = (correlation > 1e-8).sum(dim=(-2, -1))

    return nonzero.to(torch.float64) / (output_size ** 2)


def gpu_fitness_batch(candidates, udp, device="cuda"):
    """
    Evaluate a batch of candidate satellite formations
    using PyTorch on the GPU.

    candidates: tensor or array with shape (batch_size, 6 * n_sat)
    udp: initialized ESA reference problem
    """

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    device = torch.device(device)

    candidates = torch.as_tensor(
        candidates,
        dtype=torch.float64,
        device=device
    )

    if candidates.ndim == 1:
        candidates = candidates.unsqueeze(0)

    batch_size = candidates.shape[0]
    n_sat = udp.n_sat
    grid_size = udp.grid_size

    if candidates.shape[1] != 6 * n_sat:
        raise ValueError(
            f"Expected {6 * n_sat} variables, "
            f"received {candidates.shape[1]}."
        )

    # Decode the chromosome into one 6D perturbation per satellite.
    perturbations = candidates.reshape(
        batch_size, 6, n_sat
    ).transpose(1, 2)

    perturbations = perturbations * udp.scaling_factor

    # Move the precomputed state-transition matrices to the GPU.
    stms = torch.as_tensor(
        udp.stms,
        dtype=torch.float64,
        device=device
    )

    # Propagate all candidate formations across all observations.
    # Shape: (batch, observations, satellites, 6)
    propagated = torch.einsum(
    "mij,bnj->bmni",
    stms,
    perturbations
    )


    # Match the ESA reference: convert propagated positions
    # back to the original chromosome units.
    propagated = propagated / udp.scaling_factor

    observation_scores = []

    for k in range(udp.n_meas):

        points = propagated[:, k, :, :3]

        if k != 0:
            points = points / udp.inflation_factor

        # Keep points strictly inside the observation volume.
        valid = (
            (points > -1).all(dim=-1)
            & (points < 1).all(dim=-1)
        )

        # Match the ESA reference grid conversion.
        indices = (
            points * grid_size / 2
            + int(grid_size / 2)
        ).to(torch.long)

        flat_indices = (
            indices[..., 0] * grid_size * grid_size
            + indices[..., 1] * grid_size
            + indices[..., 2]
        )

        flat_indices = torch.where(
            valid,
            flat_indices,
            torch.zeros_like(flat_indices)
        )

        # Build binary 3D occupancy grids for the whole batch.
        occupancy_flat = torch.zeros(
            (batch_size, grid_size ** 3),
            dtype=torch.float64,
            device=device
        )

        occupancy_flat.scatter_add_(
            1,
            flat_indices,
            valid.to(torch.float64)
        )

        occupancy_flat.clamp_(max=1)

        occupancy = occupancy_flat.reshape(
            batch_size,
            grid_size,
            grid_size,
            grid_size
        )

        # Project the 3D occupancy grid onto three planes.
        xy = occupancy.amax(dim=3)
        xz = occupancy.amax(dim=2)
        yz = occupancy.amax(dim=1)

        # Calculate the three autocorrelation fill factors.
        f1 = autocorrelation_fill(xy)
        f2 = autocorrelation_fill(xz)
        f3 = autocorrelation_fill(yz)

        observation_scores.append(f1 + f2 + f3)

    # ESA objective: minimize the negative weakest observation.
    scores = torch.stack(observation_scores, dim=1)

    return -scores.min(dim=1).values

import numpy as np
from scipy.integrate import cumtrapz


def extended_dot(vecs1: np.ndarray, vecs2: np.ndarray) -> np.ndarray:
    """Peform the dot product of the N vectors in vecs1 (N,x) with the corresponding vectors in vecs2 (N, x)."""

    # because the array dimensions must be changed to use matrix multiplication to perform this operation, einsum is
    # (as far as I know) the fastest way to perform this operation
    return np.einsum('ij,ij->i', vecs1, vecs2)


def ang_vel(mat_traj: np.ndarray, dt) -> np.ndarray:
    """Return the angular velocity of the rotation matrix trajectory (N, 3, 3)."""
    mats_vel = np.gradient(mat_traj, dt, axis=0)
    mats_t = np.swapaxes(mat_traj, -2, -1)
    ang_vel_tensor = mats_vel @ mats_t
    ang_vel_vector = np.stack((ang_vel_tensor[:, 2, 1], ang_vel_tensor[:, 0, 2], ang_vel_tensor[:, 1, 0]), -1)
    return ang_vel_vector


def true_axial_rot(mat_traj: np.ndarray, dt) -> np.ndarray:
    """Return the true axial rotation of the GH or HT matrix trajectory (N, 3, 3).

    Note that this function assumes that the longitudinal axis of the humerus is specified by the y-axis. If not,
    then mat_traj(:,:,1) must be appropriately adjusted. For example, if the z-axis specifies the longitudinal axis of
    the humerus then the appropriate computation is mat_traj(:,:,2)
    """
    av = ang_vel(mat_traj, dt)
    return cumtrapz(extended_dot(av, mat_traj[:, :, 1]), dx=dt, initial=0)

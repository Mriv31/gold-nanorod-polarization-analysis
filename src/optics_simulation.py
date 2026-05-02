"""
Optics simulation for BFP Fresnel response and Fourkas reconstruction errors.

This module simulates the polarization-resolved field at the back focal plane
(BFP) including optional Fresnel transmission and aperture masking effects.
It also compares the reconstructed polarization angles obtained using Fourkas'
formula with the original input angles to estimate the systematic error
introduced by ignoring these optical effects.
"""

import itertools

import numpy as np
from PIL import Image


# Default optical parameters
DEFAULT_NA = 1.3
DEFAULT_NW = 1.33
DEFAULT_NO = 1.51
DEFAULT_BFP_SIZE = (1000, 1000)
DEFAULT_BFP_RADIUS = 350
DEFAULT_ORSCALE = 2.4 / DEFAULT_BFP_RADIUS
DEFAULT_MASK_PATH = "data/image_mirror_binary.tif"
DEFAULT_MASK_CENTER = (217, 230)
DEFAULT_MASK_SCALE = 1.4 / 235  # mm / px


def load_bfp_mask(mask_path=DEFAULT_MASK_PATH):
    """Load a binary BFP mask image from disk."""
    mask_image = Image.open(mask_path)
    return np.asarray(mask_image)


def find_mask_indices(
    dx,
    dy,
    mask,
    center=DEFAULT_MASK_CENTER,
    orscale=DEFAULT_ORSCALE,
    scale=DEFAULT_MASK_SCALE,
):
    """Convert BFP coordinates into mask indices and return masked pixel indices."""
    center_x, center_y = center
    indices_x = np.round(center_x + dx * orscale / scale).astype(int)
    indices_y = np.round(center_y + dy * orscale / scale).astype(int)

    out_of_bounds_x = np.logical_or(indices_x < 0, indices_x >= mask.shape[0])
    out_of_bounds_y = np.logical_or(indices_y < 0, indices_y >= mask.shape[1])
    indices_x[out_of_bounds_x] = 0
    indices_y[out_of_bounds_y] = 0

    selected = mask[indices_x, indices_y]
    return np.where(selected != 0)


def _create_bfp_grid(xl, yl):
    x = np.arange(xl)
    y = np.arange(yl)
    return np.meshgrid(x, y)


def simulate_fresnel_polarization_response(
    phi0,
    t0,
    nofresnel=0,
    use_mask=False,
    mask_path=DEFAULT_MASK_PATH,
    na=DEFAULT_NA,
    nw=DEFAULT_NW,
    no=DEFAULT_NO,
    bfp_size=DEFAULT_BFP_SIZE,
    bfp_radius=DEFAULT_BFP_RADIUS,
    orscale=DEFAULT_ORSCALE,
    mask_center=DEFAULT_MASK_CENTER,
    mask_scale=DEFAULT_MASK_SCALE,
):
    """
    Simulate the polarization-resolved field at the back focal plane.

    Parameters
    ----------
    phi0 : float
        Azimuth angle of the input dipole in degrees.
    t0 : float
        Polar angle of the input dipole in degrees.
    nofresnel : int, optional
        If non-zero, disable Fresnel transmission effects.
    use_mask : bool, optional
        Apply the BFP mask if True.
    mask_path : str, optional
        Path to the binary mask image.
    na : float, optional
        Numerical aperture of the objective.
    nw : float, optional
        Refractive index of the immersion medium.
    no : float, optional
        Refractive index of the sample medium.
    bfp_size : tuple, optional
        Size of the BFP grid in pixels (xl, yl).
    bfp_radius : float, optional
        Radius of the BFP in pixels.
    orscale : float, optional
        Scale factor from BFP coordinate to mask coordinates.
    mask_center : tuple, optional
        Mask image center coordinates in pixels.
    mask_scale : float, optional
        Scale factor for mask coordinates.

    Returns
    -------
    polres : ndarray
        Simulated polarization-resolved field at the BFP shape (3, xl, yl).
    """
    t0_rad = np.deg2rad(t0)
    phi0_rad = np.deg2rad(phi0)
    pol = np.array(
        [
            np.sin(t0_rad) * np.cos(phi0_rad),
            np.sin(t0_rad) * np.sin(phi0_rad),
            np.cos(t0_rad),
        ]
    )

    max_angle = np.arcsin(na / no)
    xl, yl = bfp_size
    cx = xl / 2
    cy = yl / 2

    mask = load_bfp_mask(mask_path) if use_mask else None
    xx, yy = _create_bfp_grid(xl, yl)
    hole_indices = (
        find_mask_indices(
            xx - cx,
            yy - cy,
            mask,
            center=mask_center,
            orscale=orscale,
            scale=mask_scale,
        )
        if use_mask
        else None
    )

    Rh = bfp_radius / np.sin(max_angle)
    RR = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    theta = np.arcsin(np.clip(RR / Rh, -1.0, 1.0))
    thetaw = np.arcsin(np.clip(RR / Rh * no / nw, -1.0, 1.0))
    phi = np.arctan2(yy - cy, xx - cx)

    kw = np.stack(
        [np.sin(thetaw) * np.cos(phi), np.sin(thetaw) * np.sin(phi), np.cos(thetaw)]
    )
    k = np.stack(
        [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
    )
    rotvec = np.stack([np.sin(phi), -np.cos(phi), np.zeros_like(phi)])

    polres = np.cross(kw, pol[:, None, None], axis=0)
    polres = np.cross(polres, kw, axis=0)

    a = np.cross(rotvec, polres, axis=0)
    b = np.cross(rotvec, a, axis=0)
    polres = (
        polres
        + np.sin(thetaw[None, :, :] - theta[None, :, :]) * a
        + (1 - np.cos(theta[None, :, :] - thetaw[None, :, :])) * b
    )

    if not nofresnel:
        tp = (2 * nw * np.cos(thetaw)) / (nw * np.cos(theta) + no * np.cos(thetaw))
        ts = (2 * nw * np.cos(thetaw)) / (nw * np.cos(thetaw) + no * np.cos(theta))

        r = np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)])
        svec = np.cross(k, r, axis=0)
        pvec = np.cross(svec, k, axis=0)
        svec = svec / np.sqrt(np.sum(svec**2, axis=0))
        pvec = pvec / np.sqrt(np.sum(pvec**2, axis=0))

        smag1 = np.sum(svec * polres, axis=0)
        pmag1 = np.sum(pvec * polres, axis=0)

        smag2 = ts * smag1
        pmag2 = tp * pmag1

        polres = smag2[None, :, :] * svec + pmag2[None, :, :] * pvec

    a = np.cross(rotvec, polres, axis=0)
    b = np.cross(rotvec, a, axis=0)
    polres = (
        polres + np.sin(theta[None, :, :]) * a + (1 - np.cos(theta[None, :, :])) * b
    )

    attenuation = np.sqrt(np.maximum(1 - RR**2 / Rh**2, 0.0))
    polres = polres / (Rh**2 * attenuation[None, :, :])
    polres = np.nan_to_num(polres)

    over_na = np.where(theta > max_angle)
    polres[:, over_na[0], over_na[1]] = 0
    if use_mask and hole_indices is not None:
        polres[:, hole_indices[0], hole_indices[1]] = 0

    return polres


def compute_polarization_intensities(polres, sum_over_pixels=True):
    """
    Compute polarization intensities for four analyzer angles from a simulated field.

    Parameters
    ----------
    polres : ndarray
        Polarization field of shape (3, xl, yl).
    sum_over_pixels : bool, optional
        If True, return the summed intensity for each analyzer angle.

    Returns
    -------
    tuple
        Four intensity values (I0, I45, I90, I135) as arrays or scalars.
    """
    field = np.moveaxis(polres, 0, 2)
    I90 = np.dot(field, [0, 1, 0]) ** 2
    I0 = np.dot(field, [1, 0, 0]) ** 2
    I45 = np.dot(field, [np.sqrt(2) / 2, np.sqrt(2) / 2, 0]) ** 2
    I135 = np.dot(field, [np.sqrt(2) / 2, -np.sqrt(2) / 2, 0]) ** 2

    if sum_over_pixels:
        return np.nansum(I0), np.nansum(I45), np.nansum(I90), np.nansum(I135)
    return I0, I45, I90, I135


def compute_fourkas_angles(I0, I45, I90, I135):
    """
    Compute angles from intensities using the Fourkas formula.

    Parameters
    ----------
    I0, I45, I90, I135 : array_like or float
        Intensities measured at 0, 45, 90, and 135 degrees.

    Returns
    -------
    tuple
        (phi, theta) angles in degrees.
    """
    na = DEFAULT_NA
    nw = DEFAULT_NW

    f_phi = 0.5 * np.arctan2(I45 - I135, I0 - I90)
    alpha = np.arcsin(na / nw)
    ca = np.cos(alpha)
    A = 1 / 6 - 1 / 4 * ca + 1 / 12 * ca**3
    B = 1 / 8 * ca - 1 / 8 * ca**3
    C = 7 / 48 - 1 / 16 * ca - 1 / 16 * ca**2 - 1 / 48 * ca**3

    O = I0 + I45 + I90 + I135
    P = I0 - I90 + I45 - I135
    c = np.cos(2 * f_phi)
    s = np.sin(2 * f_phi)

    sinsqtheta = 4 * A * P / (2 * (s + c) * O * C - 4 * B * P)
    sinsqtheta = np.clip(sinsqtheta, 0.0, 1.0)
    f_theta = np.arcsin(sinsqtheta**0.5)

    f_phi = np.rad2deg(f_phi)
    f_theta = np.rad2deg(f_theta)
    return f_phi, f_theta


def compute_fourkas_intensities(phi0, t0):
    """
    Compute expected intensities from Fourkas theory for a given dipole orientation.
    """
    t0_rad = np.deg2rad(t0)
    phi0_rad = np.deg2rad(phi0)

    na = DEFAULT_NA
    nw = DEFAULT_NW
    sst0 = np.sin(t0_rad) ** 2
    c2p = np.cos(2 * phi0_rad)
    s2p = np.sin(2 * phi0_rad)

    alpha = np.arcsin(na / nw)
    ca = np.cos(alpha)
    A = 1 / 6 - 1 / 4 * ca + 1 / 12 * ca**3
    B = 1 / 8 * ca - 1 / 8 * ca**3
    C = 7 / 48 - 1 / 16 * ca - 1 / 16 * ca**2 - 1 / 48 * ca**3

    f_I0 = A + B * sst0 + C * sst0 * c2p
    f_I45 = A + B * sst0 + C * sst0 * s2p
    f_I90 = A + B * sst0 - C * sst0 * c2p
    f_I135 = A + B * sst0 - C * sst0 * s2p

    return f_I0, f_I45, f_I90, f_I135


def compute_reconstruction_error(
    phil,
    thetal,
    nofresnel=0,
    use_mask=False,
    mask_path=DEFAULT_MASK_PATH,
    verbose=False,
):
    """
    Compute the Fourkas reconstruction error over a grid of input angles.

    Parameters
    ----------
    phil : array_like
        Azimuth angles in degrees.
    thetal : array_like
        Polar angles in degrees.
    nofresnel : int, optional
        If non-zero, skip Fresnel transmission effects.
    use_mask : bool, optional
        Apply the BFP mask when simulating the field.
    mask_path : str, optional
        Path to the mask image.
    verbose : bool, optional
        Print progress for each computed angle pair.

    Returns
    -------
    reconstructed_angles : ndarray
        Reconstructed (phi, theta) angles in degrees with shape (len(phil), len(thetal), 2).
    error_phi : ndarray
        Difference between reconstructed and input azimuth angles.
    error_theta : ndarray
        Difference between reconstructed and input polar angles.
    """
    phil = np.asarray(phil, dtype=float)
    thetal = np.asarray(thetal, dtype=float)

    reconstructed_angles = np.zeros((phil.size, thetal.size, 2), dtype=float)
    error_phi = np.zeros((phil.size, thetal.size), dtype=float)
    error_theta = np.zeros((phil.size, thetal.size), dtype=float)

    for i, j in itertools.product(range(phil.size), range(thetal.size)):
        intensities = compute_polarization_intensities(
            simulate_fresnel_polarization_response(
                phil[i],
                thetal[j],
                nofresnel=nofresnel,
                use_mask=use_mask,
                mask_path=mask_path,
            ),
            sum_over_pixels=True,
        )
        phi_rec, theta_rec = compute_fourkas_angles(*intensities)
        if phi_rec < -45:
            phi_rec += 180

        reconstructed_angles[i, j, 0] = phi_rec
        reconstructed_angles[i, j, 1] = theta_rec
        error_phi[i, j] = phi_rec - phil[i]
        error_theta[i, j] = theta_rec - thetal[j]

        if verbose:
            print(
                f"i={i} j={j} input=({phil[i]:.2f},{thetal[j]:.2f}) reconstructed=({phi_rec:.2f},{theta_rec:.2f})"
            )

    return reconstructed_angles, error_phi, error_theta

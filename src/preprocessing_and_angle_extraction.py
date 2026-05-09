import numpy as np
from scipy.optimize import minimize
from functools import partial
from pathlib import Path

try:
    from nptdms import TdmsFile
except ImportError as exc:
    TdmsFile = None
    _NPTDMS_IMPORT_ERROR = exc


# ============================================================================
# OPTICAL PARAMETERS
# ============================================================================
# Correction Matrix Parameters
T_TRANSMITTED_0 = 0.152  # Transmitted intensity for 0° polarization
T_REFLECTED_90 = 0.148  # Reflected intensity for 90° polarization
ALPHA_COEFF = 0.455  # Alpha coefficient for PBS correction
BETA_COEFF = 0.009  # Beta coefficient for PBS correction
PBS_TRANSMITTED = 0.757  # PBS transmission coefficient
PBS_REFLECTED = 0.741  # PBS reflection coefficient
MATRIX_SCALE = 8  # Scaling factor for correction matrix inversion

# Fourkas Method Parameters
DEFAULT_NUMERICAL_APERTURE = 1.3  # Numerical aperture of objective
DEFAULT_REFRACTIVE_INDEX_WATER = 1.33  # Refractive index of water


def true_best_coeff_func_mat(params, ac, mat):
    """
    Objective function for optimization to find best coefficients for angle extraction.

    Parameters:
    params (list): [a90, a45, a135] coefficients
    ac (np.ndarray): Array of intensities [c0, c90, c45, c135]
    mat (np.ndarray): Correction matrix

    Returns:
    float: Sum of squared differences for optimization
    """
    a90, a45, a135 = params
    ap = np.copy(ac)
    ap[1, :] *= a90
    ap[2, :] *= a45
    ap[3, :] *= a135
    bc = np.dot(mat, ap)
    c0 = bc[0, :]
    c90 = bc[1, :]
    c45 = bc[2, :]
    c135 = bc[3, :]
    return np.sum((c0 + c90 - c45 - c135) ** 2)


def find_best_coeff_using_mat(c0, c90, c45, c135, mat):
    """
    Find the best coefficients for correcting intensities if a optical matrix is applied
     o raw signals optimization.

    Parameters:
    c0, c90, c45, c135 (np.ndarray): Intensity arrays
    mat (np.ndarray): Correction matrix

    Returns:
    np.ndarray: Optimized coefficients [a90, a45, a135]
    """
    ac = np.asarray(np.vstack((c0, c90, c45, c135)))
    result = minimize(partial(true_best_coeff_func_mat, ac=ac, mat=mat), [1, 1, 1])
    return result.x


def T_Icor_Matrix():  # ordre 0,90,45,135, with 45 reflected
    """
    Generate the correction matrix for intensity correction in polarimetry.
    Coefficients of reflection of each cube were measured and used to build this matrix.

    Returns:
    np.ndarray: Inverse of the correction matrix divided by MATRIX_SCALE
    """
    ret = np.zeros([4, 4])
    ret[0][0] = T_TRANSMITTED_0
    ret[0][1] = 0
    ret[1][0] = 0
    ret[1][1] = T_REFLECTED_90
    alpha = ALPHA_COEFF
    beta = BETA_COEFF
    t = PBS_TRANSMITTED
    r = PBS_REFLECTED
    ret[2][0] = -alpha * beta * r  # Reflected
    ret[2][1] = alpha * beta * r
    ret[2][2] = alpha * alpha * r
    ret[2][3] = beta * beta * r
    ret[3][0] = -alpha * beta * t  # Transmitted
    ret[3][1] = alpha * beta * t
    ret[3][2] = beta * beta * t
    ret[3][3] = alpha * alpha * t
    return np.linalg.inv(ret) / MATRIX_SCALE


def Fourkas(
    c0, c90, c45, c135, NA=DEFAULT_NUMERICAL_APERTURE, nw=DEFAULT_REFRACTIVE_INDEX_WATER
):
    """
    Extract orientation angle (phi) and polar angle (theta) from polarisation signals using Fourkas
    method.

    Parameters:
    c0, c90, c45, c135 (float or np.ndarray): Input polarization signals
    NA (float): Numerical aperture (default: DEFAULT_NUMERICAL_APERTURE)
    nw (float): Refractive index of water (default: DEFAULT_REFRACTIVE_INDEX_WATER)
    Returns:
    tuple: (phi, theta) angles
    """
    alpha = np.arcsin(NA / nw)
    A = 1 / 6 - 1 / 4 * np.cos(alpha) + 1 / 12 * np.cos(alpha) ** 3
    B = 1 / 8 * np.cos(alpha) - 1 / 8 * np.cos(alpha) ** 3
    C = 7 / 48 - np.cos(alpha) / 16 - np.cos(alpha) ** 2 / 16 - np.cos(alpha) ** 3 / 48
    phi = 0.5 * np.arctan2(
        (c45 / 2 - c135 / 2), (c0 / 2 - c90 / 2)
    )  # this one I modified to symmetrize

    cs = np.cos(2 * phi)
    ss = np.sin(2 * phi)

    OP = c0 + c45 + c90 + c135
    P = c0 - c90 + c45 - c135

    sinsqtheta = 4 * A * P / (2 * (ss + cs) * OP * C - 4 * B * P)
    test = np.sqrt(sinsqtheta)  
    test = np.where(test > 1, 1, test)
    theta1 = np.arcsin(test)
    return phi, theta1


def Fourkas_extraction(
    c0: np.ndarray,
    c90: np.ndarray,
    c45: np.ndarray,
    c135: np.ndarray,
    NA: float = DEFAULT_NUMERICAL_APERTURE,
    nw: float = DEFAULT_REFRACTIVE_INDEX_WATER,
):
    """Extract phi_unwrapped and theta1 from four APD channels."""

    phi, theta1 = Fourkas(c0, c90, c45, c135, NA=NA, nw=nw)
    phi_unwrapped = np.unwrap(phi, period=np.pi)
    return phi_unwrapped, theta1


def load_tdms_channels(tdms_path, start_index=0, max_samples=250000):
    """Load the first `max_samples` points from the first four TDMS channels."""
    if TdmsFile is None:
        raise ImportError(
            "nptdms is required to read TDMS files. Install it with `pip install nptdms`."
        ) from _NPTDMS_IMPORT_ERROR

    tdms_path = Path(tdms_path)
    if not tdms_path.exists():
        raise FileNotFoundError(f"TDMS file not found: {tdms_path}")

    tdms_file = TdmsFile.read(str(tdms_path))
    channels = []
    for group in tdms_file.groups():
        channels.extend(group.channels())

    if len(channels) < 4:
        raise ValueError(
            f"Expected at least 4 channels in {tdms_path}, found {len(channels)}."
        )

    return [
        np.asarray(ch.data[start_index : start_index + max_samples], dtype=float)
        for ch in channels[:4]
    ]


if __name__ == "__main__":
    # Use the raw TDMS data from the Zenodo directory when running this script.
    # Place the downloaded dataset in `data/raw/` or update the path below to point to Zenodo data.
    tdms_path = Path(__file__).resolve().parent / "data" / "raw" / "PotAB_Motor1.tdms"
    c90, c45, c135, c0 = load_tdms_channels(
        tdms_path, start_index=250000 * 280, max_samples=500000
    )  # 90, 45, 135, 0 -  is the order in our TDMS file, but it may differ in other files. Adjust as needed.
    mat = T_Icor_Matrix()
    cor_coeff_path = (
        Path(__file__).resolve().parent / "data" / "raw" / "PotAB_Motor1_cor.npy"
    )
    cor_coeffs = np.load(cor_coeff_path)
    corrected = np.dot(mat, np.asarray([c0, c90, c45, c135]))
    c0c, c90c, c45c, c135c = corrected
    c0c, c90c, c45c, c135c = (
        c0c,
        c90c * cor_coeffs[0],
        c45c * cor_coeffs[1],
        c135c * cor_coeffs[2],
    )
    print(cor_coeffs)
    phi_unwrapped, theta1 = Fourkas_extraction(c0c, c90c, c45c, c135c)
    from matplotlib import pyplot as plt

    plt.plot(phi_unwrapped[::100])  # Quick check of phi_unwrapped
    plt.show()
    plt.plot(c0c)
    plt.plot(c90c)
    plt.plot(c45c)
    plt.plot(c135c)
    plt.show()

    print(f"Loaded {len(phi_unwrapped)} samples from {tdms_path}")
    print("First 10 phi_unwrapped:", phi_unwrapped[:10])
    print("First 10 theta1:", theta1[:10])

"""3D Polarimetry Visualization Module

Corresponds to Figure 1 in the paper.

This module creates publication-quality 3D visualizations of polarimetry measurement data
on a hemisphere. The visualization displays:
- A hemispherical reference grid (spherical coordinate system)
- Measured orientation (phi) and polar (theta) angles as point clouds
- Reference markers and orientation indicators

The TDMS input data used for these visualizations come from the Zenodo repository.

Key components:
    - plot_cylinder: Renders oriented cylinders in 3D space
    - spherical_to_cartesian: Converts spherical to Cartesian coordinates
    - create_polarimetry_figure: Main visualization function
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_cylinder(axes, direction_x, direction_y, direction_z, radius=1, num_points=50):
    """Plot a cylinder oriented along a specified direction vector.

    Renders a cylinder in 3D space with arbitrary orientation, useful for visualizing
    molecular dipole moments or other oriented structures.

    Args:
        axes: Matplotlib 3D axes object
        direction_x (float): X-component of direction vector
        direction_y (float): Y-component of direction vector
        direction_z (float): Z-component of direction vector
        radius (float): Cylinder radius
        num_points (int): Resolution for cylinder mesh generation

    Returns:
        None (modifies axes in-place)
    """
    # Calculate cylinder height from direction vector magnitude
    height = np.sqrt(direction_x**2 + direction_y**2 + direction_z**2)

    # Generate cylinder coordinates in cylindrical system
    theta = np.linspace(0, 2 * np.pi, num_points)
    z = np.linspace(-height / 2, height / 2, num_points)
    theta, z = np.meshgrid(theta, z)

    # Convert to Cartesian coordinates (cylinder axis along z)
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)

    # Normalize direction vector to unit vector
    direction_vector = np.array([direction_x, direction_y, direction_z]) / height

    # Create orthonormal basis: find two perpendicular vectors
    not_parallel = (
        np.array([1, 0, 0]) if direction_vector[0] == 0 else np.array([0, 1, 0])
    )
    basis_vector_1 = np.cross(direction_vector, not_parallel)
    basis_vector_1 /= np.linalg.norm(basis_vector_1)
    basis_vector_2 = np.cross(direction_vector, basis_vector_1)

    # Rotate cylinder to align with direction vector
    cartesian_x = (
        x * basis_vector_1[0] + y * basis_vector_2[0] + z * direction_vector[0]
    )
    cartesian_y = (
        x * basis_vector_1[1] + y * basis_vector_2[1] + z * direction_vector[1]
    )
    cartesian_z = (
        x * basis_vector_1[2] + y * basis_vector_2[2] + z * direction_vector[2]
    )

    # Plot cylinder surface
    axes.plot_surface(cartesian_x, cartesian_y, cartesian_z, color="gold", alpha=1)


def spherical_to_cartesian(radius, azimuth_angle, polar_angle):
    """Convert spherical coordinates to Cartesian coordinates.

    Uses physics convention: theta is polar angle from z-axis, phi is azimuthal angle.

    Args:
        radius (float or np.ndarray): Radial distance
        azimuth_angle (float or np.ndarray): Azimuthal angle (phi)
        polar_angle (float or np.ndarray): Polar angle from z-axis (theta)

    Returns:
        tuple: (x, y, z) Cartesian coordinates
    """
    x = radius * np.sin(polar_angle) * np.cos(azimuth_angle)
    y = radius * np.sin(polar_angle) * np.sin(azimuth_angle)
    z = radius * np.cos(polar_angle)
    return x, y, z


def create_polarimetry_figure(phi_unwrapped, theta, visualization_index=800):
    """Create a 3D visualization of polarimetry measurement data on a hemisphere.

    Displays measured orientation angles (phi) and polar angles (theta) as a point cloud
    on a hemispherical reference grid, with markers indicating specific measurement points.

    Args:
        phi_unwrapped (np.ndarray): Unwrapped azimuthal angle data (0 to 2π)
        theta (np.ndarray): Polar angle data (0 to π)
        visualization_index (int): Index of measurement point to highlight

    Returns:
        tuple: (figure, axes) Matplotlib figure and axes objects
    """
    # Visualization parameters
    num_meridians = 15
    hemisphere_radius = 1
    figure_size_mm = 40  # 40 x 40 mm figure
    dpi = 400

    # Generate hemispherical grid for reference
    azimuth_grid = np.linspace(0, 2 * np.pi, num_meridians * 10)
    polar_grid = np.linspace(0, np.pi / 2, num_meridians)  # Restrict to hemisphere
    azimuth_mesh, polar_mesh = np.meshgrid(azimuth_grid, polar_grid)

    # Convert grid to Cartesian coordinates
    grid_x, grid_y, grid_z = spherical_to_cartesian(1, azimuth_mesh, polar_mesh)

    # Create figure (convert mm to inches: 1 inch = 25.4 mm)
    fig = plt.figure(figsize=(figure_size_mm / 25.4, figure_size_mm / 25.4), dpi=dpi)
    ax = fig.add_subplot(111, projection="3d", position=[0, 0, 1, 1])

    # Plot hemispherical reference grid
    ax.plot_wireframe(grid_x, grid_y, grid_z, color="black", alpha=0.05)

    # Configure axes appearance
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])
    ax.set_axis_off()
    ax.set_box_aspect([1, 1, 0.5])
    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([0, 1])

    # Draw coordinate axes
    arrow_length = 1
    arrow_head_ratio = 0.1
    axis_linewidth = 1

    # X-axis (red convention: along x)
    ax.quiver(
        0,
        0,
        0,
        arrow_length,
        0,
        0,
        color="k",
        linewidth=axis_linewidth,
        arrow_length_ratio=arrow_head_ratio,
    )
    # Y-axis (green convention: along y)
    ax.quiver(
        0,
        0,
        0,
        0,
        arrow_length,
        0,
        color="k",
        linewidth=axis_linewidth,
        arrow_length_ratio=arrow_head_ratio,
    )
    # Z-axis (blue convention: along z)
    ax.quiver(
        0,
        0,
        0,
        0,
        0,
        arrow_length,
        color="k",
        linewidth=axis_linewidth,
        arrow_length_ratio=arrow_head_ratio,
    )

    # Convert measured angles to Cartesian coordinates
    cartesian_x = np.sin(theta) * np.cos(phi_unwrapped)
    cartesian_y = np.sin(theta) * np.sin(phi_unwrapped)
    cartesian_z = np.cos(theta)

    # Plot measured data as point cloud
    ax.scatter(cartesian_x, cartesian_y, cartesian_z, s=1, c="red")
    ax.scatter([0], [0], [0], c="red")  # Origin marker

    # Highlight a specific measurement point with cylinder and reference lines
    plot_cylinder(
        ax,
        cartesian_x[visualization_index],
        cartesian_y[visualization_index],
        cartesian_z[visualization_index],
        radius=0.1,
    )

    # Draw dashed reference lines to highlighted point
    ax.plot(
        [0, cartesian_x[visualization_index]],
        [0, cartesian_y[visualization_index]],
        [0, 0],
        color="black",
        linestyle="--",
        linewidth=1,
        zorder=15,
    )
    ax.plot(
        [0, cartesian_x[visualization_index]],
        [0, cartesian_y[visualization_index]],
        [0, cartesian_z[visualization_index]],
        color="black",
        linestyle="--",
        linewidth=1,
        zorder=16,
    )

    # Set viewpoint
    ax.view_init(elev=20.0, azim=40)

    # Add frame around entire figure
    frame_color = "black"
    frame_linewidth = 2
    fig_width, fig_height = fig.get_size_inches()
    border_rect = plt.Rectangle(
        (0, 0),
        fig_width,
        fig_height,
        linewidth=frame_linewidth,
        edgecolor=frame_color,
        facecolor="none",
        transform=fig.transFigure,
        zorder=20,
    )
    fig.patches.append(border_rect)

    return fig, ax


if __name__ == "__main__":
    """
    Example workflow demonstrating how to load polarimetry data and create 3D visualization.

    This example shows the complete pipeline from loading phi_unwrapped and theta data
    from a file to generating the publication-quality 3D figure.
    """
    from .preprocessing_and_angle_extraction import (
        load_tdms_channels,
        T_Icor_Matrix,
        Fourkas_extraction,
    )
    from pathlib import Path

    tdms_path = Path(__file__).resolve().parent / "data" / "raw" / "PotAB_Motor1.tdms"
    c90, c45, c135, c0 = load_tdms_channels(
        tdms_path, start_index=250000 * 274, max_samples=250000 * 10
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
    phi_unwrapped, theta1 = Fourkas_extraction(c0c, c90c, c45c, c135c)
    fig, ax = create_polarimetry_figure(
        phi_unwrapped[::200], theta1[::200], visualization_index=800
    )
    plt.show()

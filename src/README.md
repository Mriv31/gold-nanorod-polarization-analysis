# Heterogeneity and Multi-scale Dynamics in the Molecular Bearing of the Bacterial Flagellum

Martin Rieu¹,², Daping Xu¹,², Gunasekaran Subramaniam¹,², Ashley L. Nord³, Alexis Courbet³,⁴,⁵,⁶, Hafez El Sayyed¹,², Richard M. Berry*¹,²

1.	Department of Physics, University of Oxford, Oxford, UK
2.	Kavli Institute for Institute for Nanoscience Discovery, Oxford, UK
3.	Centre de Biologie Structurale, Université de Montpellier, CNRS, INSERM, Montpellier, France
4.	Institute for Protein Design, University of Washington, Seattle, WA, 98105, USA
5.	Department of Biochemistry, University of Washington, Seattle, WA, 98195, USA
6.	Howard Hughes Medical Institute, University of Washington, Seattle, WA, 98105, USA


Author of the code: Martin Rieu (martin.rieu@physics.ox.ac.uk)
Corresponding author: Prof. Richard M. Berry (richard.berry@physics.ox.ac.uk)

This repository contains code for analyzing raw APD measurements from single gold nanorods. Each APD records one polarization-scattered intensity from the rod at 0°, 45°, 90°, or 135°. The code transforms these signals into time trajectories of gold nanorod orientation and performs the subsequent analysis used in the paper.

Raw experimental data used with this code are available from Zenodo at https://doi.org/10.5281/zenodo.20088460.

The repo is organized around figure-generation scripts, data analysis helpers, and optics simulation utilities. Most modules are designed to accept precomputed arrays such as `phi_unwrapped` and correlation channel data rather than relying on raw file loaders.

Note: many scripts use relative imports (for example, `from .kimograph import compute_kimograph`), so they should be run as part of the `src` package. From the repository root, use commands such as:

- `python -m src.analysis_kimograph_dynamics`
- `python -m src.preprocessing_and_angle_extraction`

Alternatively, change into the `src` directory and run module form:

- `cd src`
- `python -m analysis_kimograph_dynamics`

Avoid executing files directly with `python analysis_kimograph_dynamics.py` because relative imports may fail.

## Repository Structure
- `preprocessing_and_angle_extraction.py` - Takes raw polarization APD data and returns `phi_unwrapped`. This is the first step in the pipeline.
- `step_detection.py` - Takes `phi_unwrapped` and returns `filtered_phi`, `time_boundaries`, and `levels` from step detection. Most downstream functions build on these outputs.
- `kimograph.py` - Kimograph computation and plotting utilities.
- `MSD_and_statistics.py` - Mean squared displacement and related statistical analysis.
- `ECF.py` - Empirical characteristic function / additional analysis helpers.
- `map_local_to_global_states.py` - State mapping to global statistics (used for motor PotAB_Motor1 and Figure 3 only).
- `transition_and_lifetimes_statistics.py` - Transition-state and lifetime statistics analysis.
- `visualization_hemispherical_polarimetry.py` - 3D polarimetry visualization module (corresponds to Figure 1); builds publication-quality hemispherical plots of measured orientation (`phi`) and polar (`theta`) angles.
- `analysis_motor_segment_transitions.py` - Motor segment transition analysis (corresponds to Figure 5); includes state detection, KDE-based peak finding, and segment analysis for different motors.
- `analysis_kimograph_dynamics.py` - Kimograph-based phase dynamics analysis (corresponds to Figure 6); constructs kimograph, phase histograms, and trace panels from unwrapped phase trajectories.
- `analysis_apd_correlation_stability.py` - APD correlation and phase stability analysis (corresponds to Supplementary Figure S3); accepts `phi_unwrapped` and four correlation channels, then computes histogram peaks, PSD, and Allan deviation.
- `analysis_drag_estimation_free_rotation.py` - Drag estimation for freely rotating nanorods (corresponds to Supplementary Figure S7); estimates drag coefficients from Welch PSD fits.
- `optics_simulation.py` - Optics simulation utilities for back-focal-plane Fresnel polarization response and Fourkas reconstruction error analysis.


## Figure Mapping

### Figure 1
`visualization_hemispherical_polarimetry.py` contains code to generate 3D polarimetry visualizations. It converts spherical coordinates to Cartesian coordinates and renders oriented dipole-like cylinders on a hemispherical reference grid. This script is the main entry point for Figure 1-style polarimetry plots.

### Figure 2
No dedicated analysis file.

### Figure 3
`transition_and_lifetimes_statistics.py`

### Figure 4
`MSD_and_statistics.py`

### Figure 5
`analysis_motor_segment_transitions.py` provides a complete Figure 5 builder. It loads motor segment data, detects states using KDE, maps states to clusters, and plots the segment-level analysis for Motor 2 and Motor 4.

### Figure 6
`analysis_kimograph_dynamics.py` contains both a lower-level routine and a higher-level builder function. It builds the kimograph panel, the phase histogram panel, and the phase trace panel from a provided dataset.

## Supplementary Figures

### Supplementary Figure S3
`analysis_measurement_noise.py` is a reusable analysis module. It expects:
- `phi_unwrapped`: unwrapped phase trace

### Supplementary Figure S7
`analysis_drag_estimation_free_rotation.py` is written to accept a list of `phi_unwrapped` traces. It computes drag estimates from PSD fits and builds the figure panels without raw file loading.

## Optics and Simulation

`optics_simulation.py` includes reusable utilities for simulating back-focal-plane optics and estimating reconstruction error from Fourkas polarization formulas. This module is intended for optical modeling and comparison with experimental orientation recovery.

## Requirements

- Python 3.x
- `numpy`
- `matplotlib`
- `scipy`
- `Pillow` (for `optics_simulation.py` mask loading)

## License

This repository is licensed under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
Users may use, reuse, redistribute, and modify the code, provided that appropriate credit is given to Martin Rieu, University of Oxford, and the corresponding author Richard M. Berry. See the `LICENSE` file for full terms.

## Contact

This README is intended to support the paper *Heterogeneity and multi-scale dynamics in the molecular bearing of the bacterial flagellum*. For figure details not encoded in this repo, please provide the missing descriptions manually.

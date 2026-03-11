# Surf_Benchmarking
# Physics-Informed Rejection Sampling for Ground Water Flow (GWF)

This repository contains a JAX-accelerated pipeline for generating and filtering physical fields based on the Darcy Flow equation. It uses Gaussian Random Fields (GRF) as coefficients and a Conjugate Gradient (CG) solver to simulate pressure distributions.

## 🚀 Features
- **GPU Acceleration**: Fully vectorized PDE solver using `JAX` and `jax.vmap`.
- **HDF5 Storage**: Efficient large-scale data management using `h5py` to prevent memory overflow.
- **Rotation Augmentation**: Leveraging physical symmetry to 4x the dataset without extra PDE solves.
- **Flexible Observation Modes**: Supports both Sparse Random Sensors and Low-Resolution Full-Field observations.

## 🛠 Installation

To replicate the environment, ensure you have Python 3.10+ and a CUDA-enabled environment. You can install the dependencies using:

```bash
pip install -r requirements.txt
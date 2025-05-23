# LFADS eFish 

This repository contains a PyTorch implementation of Latent Factor Analysis via Dynamical Systems (LFADS) for analyzing neural population activity of electrosensory pyramidal cells. LFADS model is based on the original paper "LFADS: Latent Factor Analysis via Dynamical Systems" by Pandarinath et al. (2018). The model is designed to extract latent dynamics from neural population activity, obtained using Neuropixel probes, and to reconstruct neural firing rates from latent dynamics. 
The model is written in python using PyTorch and is designed to be flexible and extensible for various applications. The current version supports invariant encoding task during chirp communication between two conspecific. Other stimuli including envelope, localization, and naturalistic stimuli will be added in future releases.

## LFADS Overview

LFADS is a variational auto-encoder framework that uses nonlinear dynamical systems (a recurrent neural network) to model time series data. It was specifically designed for neural recordings, where the goal is to extract latent neural dynamics from high-dimensional, noisy spike trains.

Key features:
- Infers smooth, low-dimensional latent trajectories from neural population activity
- Reconstructs neural firing rates from latent dynamics
- Disentangles initial conditions and temporal inputs
- Handles multi-trial data with trial-specific initial conditions

## Citation
If you use this code in your research, please cite the original LFADS paper:

```
@article{pandarinath2018inferring,
  title={Inferring single-trial neural population dynamics using sequential auto-encoders},
  author={Pandarinath et al.},
  journal={Nature methods},
  volume={15},
  number={10},
  pages={805--815},
  year={2018}
}
```

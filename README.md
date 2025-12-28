# Analytical IR-Drop Modeling for Memristive Crossbar Neural Networks

This repository presents a **physics-informed analytical framework** for modeling IR-drop effects in memristive crossbar arrays used for neural network inference.

The work focuses on quantifying the impact of physical non-idealities on analog matrix–vector multiplication (MVM) and on evaluating compensation strategies at the system level.

---

## Overview

Analog in-memory computing based on memristive crossbars promises high energy efficiency, but its accuracy is strongly affected by IR-drop and line resistance effects.

This project implements an analytical model derived from Kirchhoff’s laws to compute **effective conductance matrices** that account for IR-drop. The model is integrated into the forward pass of a multilayer perceptron and evaluated on the MNIST dataset.

Three inference modes are compared:
- Ideal digital inference  
- Analog inference with IR-drop  
- Analog inference with four-matrix compensation  

---

## Main Contributions

- Physics-informed analytical modeling of IR-drop effects  
- Sparse linear system formulation derived from circuit equations  
- Conjugate Gradient solver with CPU parallelization  
- Efficient caching of crossbar structures and conductance matrices  
- Support for tiled crossbar architectures  
- Layer-wise comparison between digital and analog inference  
- Quantitative and qualitative evaluation on MNIST  

---

## Methodology

For each linear layer, network weights are mapped to memristive conductances and partitioned into tiled crossbar arrays.

IR-drop effects are modeled by solving a sparse linear system derived from circuit equations. The resulting effective conductance matrix is then used to perform analog matrix–vector multiplication.

A four-matrix compensation scheme is implemented to mitigate non-ideal effects and improve inference accuracy.

---

## Results
The framework reports:
- Digital inference accuracy
- Analog accuracy with IR-drop
- Analog accuracy with four-matrix compensation
Layer-wise heatmaps, histograms, and final class probability plots are generated to provide physical insight into the effects of non-idealities.


---

## Disclaimer
This repository contains research code associated with an ongoing scientific work.
The implementation is provided for transparency and reproducibility purposes.
Details and final results will be presented in a forthcoming publication.

--- 

## Notes
This repository prioritizes physical interpretability and system-level analysis over hardware-specific simulation, enabling scalable evaluation of analog neural network architectures. The project is part of ongoing research on analog in-memory computing and hardware-aware neural network modeling.


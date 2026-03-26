# Python + HPC (PBS) for Collective Quantum Optics

This repository contains a tutorial style presentations with the tools I used or developed through my PhD and that could be used for ongoing and future projects in our group. 
It includes:
- Scientific Python workflows
- VS Code productivity for research code
- HPC cluster execution with PBS job submission

## Main topics 
1. **MCWF simulations** (trajectory-based workflows)
2. **Mean-field (MF) many-ODE systems** (parallelization for large systems and vectorization)
3. **Matrix diagonalization on GPU**


## Repository layout
- `docs/` tex slides and useful materials
- `examples/` code examples for all topics 
- `pbs/` placeholder folder for cluster scripts
- `data/`, `results/`, placeholders for project artifacts
- `.vscode/` recommended editor setup

## Description of the past tutorials (March 2026)
- **Lecture 1:** problem overview + mean-field equations in a wave-guide 
- **Lecture 2:** master equation and how to solve it. Examples of calculation with MCWF for driven Dicke problem. This codes basically were used for the paper 

[1] Nikita Leppenen, Ephraim Shahmoon, Quantum correlated steady states under competing collective and individual decay, arXiv:2404.02134, https://arxiv.org/abs/2404.02134

Monte-Carlo Wave Function for single spin in quantum dot -- Figure 2 in 

[2] Nikita Leppenen, Dmitry S. Smirnov, Birefringent Spin-Photon Interface Generates Polarization Entanglement, Adv Quantum Technol.2024, 7, 2400193. https://doi.org/10.1002/qute.202400193

Code example is not added yet 

## Possible new tutorials 

- **Bonus: Green Function** Details of how to implement Diadic Green Function in different dimension. Basically some tools of my current project
- **Matrix Diagonalization** I have a lot of results for the Liouvillian diagonalization. 

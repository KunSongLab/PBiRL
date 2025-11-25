# Pareto-based Bi-objective Ramp Metering with Reinforcement Learning

A reinforcement learning framework for Pareto-based bi-objective ramp metering control on expressway networks, optimizing both traffic efficiency and revenue objectives.

## Overview

This repository implements a Pareto-based bi-objective deep Q-learning approach for coordinated ramp metering control. The framework utilizes prioritized experience replay and operates on the Asymmetric Cell Transmission Model (ACTM) for realistic traffic flow simulation.

## Features

- **Bi-objective Optimization**: Simultaneously optimizes traffic efficiency and toll revenue to obtain Pareto-optimal solutions
- **Pareto Front Exploration**: Discovers trade-off surfaces between conflicting objectives
- **Deep Q-Network (DQN)**: Model-free reinforcement learning with prioritized experience replay
- **ACTM Integration**: High-fidelity traffic simulation using C++ compiled shared libraries
- **Ring Expressway Scenario**: Pre-configured test environment for ramp metering evaluation

## Quick Start

### Prerequisites

- Python 3.11
- NumPy, PyTorch, Matplotlib (see `requirements.txt`)
- C++ compiled ACTM library (included)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/KunSongLab/BiRL.git
cd BiRL
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install ACTM library globally:
   
   Copy the appropriate shared library to your Python site-packages directory:
   
   - **Linux**: `monstac_api.so`
   - **Windows**: `monstac_api.pyd`
   
   ```bash
   # Find your site-packages directory
   python -c "import site; print(site.getsitepackages())"
   
   # Copy the library (Linux example)
   cp monstac_api.so /path/to/site-packages/
   ```

### Running the Code

Execute the main training script:

```bash
python controls/run/RingFreeway/main_dqn_priority_mo.py
```

## ACTM Shared Library

The Asymmetric Cell Transmission Model (ACTM) is implemented in C++ for computational efficiency and compiled as a shared library for Python integration:

- **monstac_api.so**: Linux/Unix binary
- **monstac_api.pyd**: Windows binary

These libraries can be placed in your Python `site-packages` directory for global access across projects, or kept in the project root for local use.

## Data Privacy Notice

The traffic flow data in `scenarios/data/RingFreeway/demands.csv` consists of randomly generated values for privacy protection purposes. The actual traffic demand patterns used in the original research have been replaced with synthetic data to protect sensitive information. Users should replace this file with their own traffic demand data for real-world applications.

## Configuration

Key parameters can be adjusted:

- Network topology (see `edges.csv, nodes.csv`)
- Training episodes
- Reward function weights
- DQN hyperparameters (learning rate, batch size, epsilon decay)

## Results

Training results, including:
- Episode rewards
- Pareto front and hypervolume visualization
- Policy performance metrics

will be saved to the `results/` directory (auto-created during execution).

## Citation

If you use this code in your research, please cite:

```bibtex
@software{BiRL,
  author = {Kun Song},
  title = {Bi-objective Ramp Metering with Reinforcement Learning},
  year = {2025},
  url = {clone https://github.com/KunSongLab/BiRL}
}
```

## Contact

For questions or issues, please open an issue on github.

# Deep Q-Network (DQN) Implementation for Super Mario Bros

**43008 Reinforcement Learning - Assignment 3, Part E**

**Author:** Ricki Yang (Student ID: 25657673)  
**Email:** Yizhuo.yang@student.uts.edu.au  
**Institution:** University of Technology Sydney  
**Date:** November 2024

---

## Project Overview

This repository contains a complete Deep Q-Network (DQN) implementation for training reinforcement learning agents to play Super Mario Bros. The project demonstrates the **specialization-generalization trade-off** in deep reinforcement learning through systematic comparison of single-level versus multi-level training paradigms.

## Key Results Summary

### Experiment 17 (Successful Run)
- **Total Training Steps:** 5,000,000
- **Training Environment:** AutoDL RTX 5090 (24GB VRAM)
- **Training Duration:** ~6 hours
- **Q-value Growth:** 68.2% increase over training
- **Single-Level Performance:** 100% completion rate on World 1-1
- **Sequential Performance:** Consistent failure at World 1-2 transition
- **Generalization:** Demonstrates classic overfitting pattern

### Core Findings
1. **Specialization vs Generalization Trade-off:** Single-level training produces perfect specialists but brittle agents
2. **Sequential Failure Pattern:** All single-level trained agents fail immediately upon environment transition
3. **Training Distribution Impact:** Experience diversity fundamentally shapes agent capabilities

---

## Repository Structure

```
super-mario-dqn/
├── README.md                           # This documentation
├── LICENSE                             # MIT License
├── CITATION.cff                        # Academic citation format
├── requirements.txt                    # Python dependencies
├── code/
│   ├── mario_dqn_autodl.py            # Main DQN training implementation
│   ├── mario_dqn_local_m4.py          # M4 Pro optimized version
│   ├── mario.py                        # Environment wrapper
│   ├── evaluate_final_model.py        # Single-level evaluation
│   └── evaluate_multi_stage.py        # Sequential evaluation
├── models/
│   └── mario_DQN_autodl_rtx5090_5000000_steps.zip  # Trained model (exp17)
├── results/
│   ├── training_summary.json          # Training metrics summary
│   ├── q_value_monitor.csv            # Q-value evolution data
│   ├── evaluation_single_stage.json   # Single-level test results
│   └── evaluation_continuous.json     # Sequential test results
├── figures/
│   ├── dqn_learning_summary.png       # Training curves
│   ├── dqn_qvalue_evolution.png       # Q-value progression
│   └── dqn_evaluation_results.png     # Performance comparison
└── videos/                            # Gameplay recordings
    ├── mario_continuous-step-*.mp4    # Continuous 1-1→1-2 progression videos  
    └── mario_dqn_eval-step-*.mp4      # Single-stage 1-1 evaluation video
```

---

## Technical Configuration

### Hardware
- **Device**: MacBook Pro M4 Pro / AutoDL RTX 5090
- **RAM**: 24GB unified memory / 32GB system RAM
- **GPU**: Apple MPS / NVIDIA CUDA
- **Acceleration**: ~450-500 steps/second (M4) / ~800 steps/second (RTX 5090)

### Algorithm: Deep Q-Network (DQN)
- **Network Architecture**: IMPALA CNN
  - Conv1: 16 filters, 8×8, stride 4
  - Conv2: 32 filters, 4×4, stride 2
  - Conv3: 32 filters, 3×3, stride 1
  - Fully connected: 256 units
- **Features**: Double DQN, Target Network, Experience Replay

### Hyperparameters
```python
Total timesteps:       5,000,000
Learning rate:         1.4e-5
Batch size:            256 (M4) / 512 (RTX 5090)
Buffer size:           50,000 (M4) / 100,000 (RTX 5090)
Learning starts:       10,000
Target update:         10,000
Train frequency:       4
Gradient steps:        2
Exploration fraction:  15%
Final epsilon:         0.01
Gamma (discount):      0.95
```

### Environment Preprocessing
- Frame skip: 4 (process every 4th frame)
- Grayscale: RGB → Grayscale conversion
- Resize: 84×84 pixels
- Frame stack: 4 consecutive frames
- Action space: SIMPLE_MOVEMENT (7 actions)

---

## Training Results

### Q-Value Progression
| Phase | Steps | Mean Q-Value | Max Q-Value |
|-------|-------|--------------|-------------|
| Early | 0-200K | 1.024 | 2.464 |
| Middle | 2.4-2.6M | 1.335 | 3.395 |
| Final | 4.8-5.0M | 1.723 | 2.838 |

**Q-Value Growth**: +68.2% (from 1.024 to 1.723)

### Learning Metrics
- **Peak Q-value:** 4.350 (achieved at step 500,000)
- **Training stability:** Moderate (some volatility in later stages)
- **Convergence:** Partial (Q-values still improving at end)

---

## Evaluation Results

### Single-Stage Evaluation (Stage 1-1)
- **Episodes:** 10
- **Completion Rate:** 100% (10/10) 
- **Mean Reward:** 3,296 ± 936
- **Reward Range:** 2,984 to 6,105
- **Mean Steps:** 345
- **Performance:** Perfect mastery of stage 1-1

### Continuous Multi-Stage Evaluation
- **Episodes:** 5
- **Stages Completed:** 1.0 per episode (only stage 1-1)
- **Mean Reward:** 10,448
- **Mean Steps:** 2,348 (345 in 1-1 + ~2,000 in 1-2 before death)
- **Stage 1-1:** 100% completion (5/5)
- **Stage 1-2:** 0% completion (0/5)
- **Generalization:** Limited to first stage only

---

## Usage Instructions

### Installation
```bash
git clone https://github.com/YizhuoYang07/super-mario-dqn.git
cd super-mario-dqn
pip install -r requirements.txt
```

### Training a New Model
```bash
# AutoDL RTX 5090 (recommended for fast training)
python code/mario_dqn_autodl.py --timesteps 5000000

# Local M4 Pro (for local development)
python code/mario_dqn_local_m4.py --timesteps 3000000
```

### Evaluating Trained Model
```bash
# Single-level evaluation
python code/evaluate_final_model.py --model models/mario_DQN_autodl_rtx5090_5000000_steps.zip

# Sequential evaluation  
python code/evaluate_multi_stage.py --model models/mario_DQN_autodl_rtx5090_5000000_steps.zip
```

---

## Requirements

### Dependencies
- Python 3.11+
- PyTorch (with MPS/CUDA support)
- Stable-Baselines3
- Gymnasium
- Custom Mario environment (included)

### Hardware Requirements

#### Minimum (Local Training)
- 16GB RAM
- Apple Silicon M1/M2/M3/M4 or NVIDIA GPU
- 50GB storage space

#### Recommended (Cloud Training)
- 24GB VRAM (RTX 5090/4090)
- 32GB System RAM
- 100GB NVMe storage

---

## Conclusions

### Key Findings
1. DQN can effectively learn to complete Super Mario Bros stage 1-1 with 100% success
2. IMPALA CNN architecture is sufficient for visual feature extraction
3. Apple MPS acceleration provides viable alternative to CUDA GPUs
4. Multi-stage generalization requires specific training strategies beyond simple exposure
5. 5M steps is sufficient for single-stage mastery but insufficient for multi-stage learning

### Future Work
1. Implement curriculum learning for progressive stage difficulty
2. Extend training to 10M-20M steps for better multi-stage performance
3. Experiment with PPO or A3C for improved exploration
4. Add stage-specific reward shaping
5. Compare performance across different algorithm families

---

## Academic Context

### Citation
If you use this code in your research, please cite:

```bibtex
@misc{yang2024dqn_mario,
  title={Deep Q-Network Implementation for Super Mario Bros: A Study of Specialization-Generalization Trade-offs},
  author={Yang, Ricki},
  year={2024},
  institution={University of Technology Sydney},
  course={43008 Reinforcement Learning}
}
```

### Acknowledgments
This implementation builds upon:
- Mnih et al. (2015) - Deep Q-Network foundations
- Espeholt et al. (2018) - IMPALA CNN architecture
- Stable-Baselines3 framework
- Gymnasium interface standards

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

This is an academic project for coursework evaluation. For questions or discussions about the research, please contact the author.

**Repository URL:** https://github.com/YizhuoYang07/super-mario-dqn.git

---

**Submission Date:** November 2024  
**Contact:** Ricki Yang (25657673) - Yizhuo.yang@student.uts.edu.au

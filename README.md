# Deep Q-Network (DQN) Implementation for Super Mario Bros# DQN Training for Super Mario Bros

## 43008 Reinforcement Learning - Assignment 3, Part E

**Student**: Ricki Yang (Student ID: 25657673)  

**Author:** Ricki Yang (Student ID: 25657673)

**Course**: 43008 - Reinforcement Learning  

**Email:** Yizhuo.yang@student.uts.edu.au  

**Platform**: MacBook Pro M4 Pro (24GB RAM, MPS acceleration)  

**Institution:** University of Technology Sydney  

**Training Date**: October, 2024  

**Date:** November 2024



## Project Overview---



This submission contains a complete Deep Q-Network (DQN) implementation for training agents to play Super Mario Bros. The project demonstrates the specialization-generalization trade-off in reinforcement learning through systematic comparison of single-level versus multi-level training paradigms.## 📁 Submission Structure



## Key Results Summary```

final_submission/

### Experiment 17 (Successful Run)├── README.md                          # This file

- **Total Training Steps:** 5,000,000├── EXPERIMENT_REPORT.md               # Detailed experiment documentation

- **Training Environment:** AutoDL RTX 5090 (24GB VRAM)├── code/                              # Training and evaluation scripts

- **Training Duration:** ~6 hours│   ├── mario_dqn_local_m4.py         # Main training script (M4 Pro optimized)

- **Q-value Growth:** 68.2% increase over training│   ├── mario.py                       # Environment wrappers

- **Single-Level Performance:** 100% completion rate on World 1-1│   ├── evaluate_final_model.py       # Single-stage evaluation

- **Sequential Performance:** Consistent failure at World 1-2 transition│   └── evaluate_multi_stage.py       # Multi-stage evaluation

- **Generalization:** Demonstrates classic overfitting pattern├── models/                            # Trained models

│   ├── mario_DQN_5000000_steps.zip   # Final model (5M steps)

### Core Findings│   └── checkpoints/                   # Intermediate checkpoints

1. **Specialization vs Generalization Trade-off:** Single-level training produces perfect specialists but brittle agents├── results/                           # Training and evaluation results

2. **Sequential Failure Pattern:** All single-level trained agents fail immediately upon environment transition│   ├── training_summary.json         # Q-value progression data

3. **Training Distribution Impact:** Experience diversity fundamentally shapes agent capabilities│   ├── q_value_monitor.csv           # Q-value monitoring

│   ├── evaluation_single_stage.json  # Single-stage evaluation results

## File Structure│   └── evaluation_continuous.json    # Multi-stage evaluation results

└── videos/                            # Gameplay recordings

```    ├── single_stage/                  # 1-1 stage videos (10 episodes)

final_submission/    └── continuous/                    # Continuous progression videos (5 episodes)

├── README.md                           # This file```

├── code/

│   ├── mario_dqn_autodl.py            # Main DQN training implementation---

│   ├── mario_dqn_local_m4.py          # M4 Pro optimized version

│   ├── mario.py                        # Environment wrapper## 🎯 Experiment Overview

│   ├── evaluate_final_model.py        # Single-level evaluation

│   └── evaluate_multi_stage.py        # Sequential evaluation### **Objective**

├── models/Train a Deep Q-Network (DQN) agent to play Super Mario Bros using reinforcement learning on a MacBook Pro M4 Pro with MPS (Metal Performance Shaders) GPU acceleration.

│   └── mario_DQN_autodl_rtx5090_5000000_steps.zip  # Trained model (exp17)

├── results/### **Key Results**

│   ├── training_summary.json          # Training metrics summary- ✅ **Training Completed**: 5,000,000 steps (100% complete)

│   ├── q_value_monitor.csv            # Q-value evolution data- ✅ **Stage 1-1 Mastery**: 100% completion rate (10/10 episodes)

│   ├── evaluation_single_stage.json   # Single-level test results- ✅ **Efficient Strategy**: Average 345 steps to complete stage 1-1

│   └── evaluation_continuous.json     # Sequential test results- ✅ **High Stability**: Consistent performance across all evaluation episodes

├── figures/- ⚠️ **Limited Generalization**: 0% completion on stage 1-2

│   ├── dqn_learning_summary.png       # Training curves

│   ├── dqn_qvalue_evolution.png       # Q-value progression---

│   └── dqn_evaluation_results.png     # Performance comparison

└── videos/                            # Gameplay recordings
    ├── mario_continuous-step-*.mp4    # Continuous 1-1→1-2 progression videos  
    └── mario_dqn_eval-step-*.mp4      # Single-stage 1-1 evaluation video
```## 🔧 Technical Configuration

```

### **Hardware**

## Implementation Highlights- **Device**: MacBook Pro M4 Pro

- **RAM**: 24GB unified memory

### 1. IMPALA CNN Architecture- **GPU**: Apple MPS (Metal Performance Shaders)

- Convolutional neural network optimized for visual feature extraction- **Acceleration**: ~450-500 steps/second

- 84x84x4 frame stacking for temporal information

- Efficient processing of RGB game frames### **Algorithm: Deep Q-Network (DQN)**

- **Network Architecture**: IMPALA CNN

### 2. Advanced DQN Features  - Conv1: 16 filters, 8×8, stride 4

- **Double DQN:** Reduces overestimation bias  - Conv2: 32 filters, 4×4, stride 2

- **Target Network:** Stable updates every 10K steps  - Conv3: 32 filters, 3×3, stride 1

- **Experience Replay:** 50K-100K sample buffer  - Fully connected: 256 units

- **Epsilon-Greedy Exploration:** Balanced exploration-exploitation- **Features**: Double DQN, Target Network, Prioritized Experience Replay



### 3. Training Optimizations### **Hyperparameters**

- **Hardware-Specific Configurations:** Optimized for RTX 5090 and M4 Pro```python

- **Multi-Platform Support:** AutoDL cloud and local MacBook trainingTotal timesteps:       5,000,000

- **Resume Capability:** Checkpoint-based training continuationLearning rate:         1.4e-5

- **Memory Efficiency:** Conservative buffer sizing for 24GB RAM systemsBatch size:            256

Buffer size:           50,000

### 4. Evaluation FrameworkLearning starts:       10,000

- **Single-Level Testing:** Performance on training distributionTarget update:         10,000

- **Sequential Testing:** Sustained performance across level transitionsTrain frequency:       4

- **Video Recording:** Visual analysis of agent behaviorGradient steps:        2

- **Q-Value Monitoring:** Learning progression trackingExploration fraction:  15%

Final epsilon:         0.01

## Key Technical ContributionsGamma (discount):      0.95

Parallel environments: 4

### 1. Platform Optimization```

```python

# AutoDL RTX 5090 Configuration### **Environment Preprocessing**

buffer_size=100_000        # Large buffer for GPU memory- Frame skip: 4 (process every 4th frame)

batch_size=512            # Maximized GPU throughput- Grayscale: RGB → Grayscale

gradient_steps=2          # Increased learning per update- Resize: 84×84 pixels

- Frame stack: 4 consecutive frames

# M4 Pro Configuration  - Action space: SIMPLE_MOVEMENT (7 actions)

buffer_size=50_000        # Memory-efficient for 24GB unified memory

batch_size=256            # MPS-optimized batch size---

```

## 📊 Training Results

### 2. Robust Environment Interface

- Gymnasium-compatible Mario environment### **Q-Value Progression**

- Dual gym/gymnasium API support| Phase | Steps | Mean Q-Value | Max Q-Value |

- NumPy 2.0 compatibility fixes|-------|-------|--------------|-------------|

- Metal Performance Shaders (MPS) acceleration| Early | 0-200K | 1.024 | 2.464 |

| Middle | 2.4-2.6M | 1.335 | 3.395 |

### 3. Comprehensive Evaluation| Final | 4.8-5.0M | 1.723 | 2.838 |

```python

# Sequential evaluation revealing generalization limits**Q-Value Growth**: +68.2% (from 1.024 to 1.723)

def evaluate_sequential_performance():

    """Tests agent across dependent task sequences"""### **Learning Metrics**

    # World 1-1: 100% success rate- Peak Q-value: 4.350 (achieved at step 500,000)

    # World 1-2: 0% success rate (immediate failure)- Training stability: Moderate (some volatility in later stages)

```- Convergence: Partial (Q-values still improving at end)



## Experimental Results---



### Training Performance (Exp17)## 🎮 Evaluation Results

- **Early Q-values (0-1M steps):** Mean = 1.024, Volatility = 0.067

- **Late Q-values (4-5M steps):** Mean = 1.723, Volatility = 0.093### **Single-Stage Evaluation (Stage 1-1)**

- **Peak Q-value:** 4.350- **Episodes**: 10

- **Convergence:** Achieved by 3M steps- **Completion Rate**: 100% (10/10) ✅

- **Mean Reward**: 3,296 ± 936

### Evaluation Performance- **Reward Range**: 2,984 to 6,105

- **World 1-1 Completion:** 100% (10/10 episodes)- **Mean Steps**: 345

- **Mean Reward:** 3,296 ± 936- **Performance**: Perfect mastery of stage 1-1

- **Episode Length:** 345 steps (consistent)

- **World 1-2 Performance:** 0% success rate### **Continuous Multi-Stage Evaluation**

- **Episodes**: 5

### Generalization Analysis- **Stages Completed**: 1.0 per episode (only stage 1-1)

- **Training Distribution:** Single level (World 1-1)- **Mean Reward**: 10,448

- **Test Distribution:** Sequential progression (1-1 → 1-2 → ...)- **Mean Steps**: 2,348 (345 in 1-1 + ~2,000 in 1-2 before death)

- **Result:** Perfect within-distribution, zero transfer- **Stage 1-1**: 100% completion (5/5)

- **Stage 1-2**: 0% completion (0/5)

## Research Implications- **Generalization**: Limited to first stage only



### 1. Specialization-Generalization Trade-off---

The results empirically demonstrate that concentrated experience produces brittle specialists while diverse sampling yields adaptable agents lacking completion capability.

## 🎬 Video Recordings

### 2. Sequential Assessment Necessity

Traditional RL benchmarks evaluating isolated episodes miss critical failure modes visible only in dependent task sequences.### **Single-Stage Videos** (10 episodes)

- All episodes show perfect completion of stage 1-1

### 3. Training Distribution Impact- Consistent strategy: direct path to goal

Experience diversity fundamentally shapes agent capabilities along dual dimensions of task mastery and environmental adaptability.- Average completion time: 345 steps

- **Note**: Videos play at 4× speed due to frame skip = 4

## Usage Instructions

### **Continuous-Stage Videos** (5 episodes)

### Training a New Model- Shows progression from 1-1 to 1-2

```bash- Demonstrates agent dying in stage 1-2

# AutoDL RTX 5090 (recommended)- Highlights limitation in generalization

python mario_dqn_autodl.py --timesteps 5000000

---

# Local M4 Pro

python mario_dqn_local_m4.py --timesteps 3000000## 📈 Performance Analysis

```

### **Strengths**

### Evaluating Trained Model1. ✅ **Perfect Stage 1-1 Mastery**: 100% success rate demonstrates effective learning

```bash2. ✅ **Efficient Strategy**: 345-step completion is highly optimized

# Single-level evaluation3. ✅ **High Consistency**: Minimal variance across episodes (σ = 936 on 10 episodes)

python evaluate_final_model.py --model models/mario_DQN_autodl_rtx5090_5000000_steps.zip4. ✅ **Stable Policy**: Deterministic policy shows robust convergence

5. ✅ **MPS Acceleration**: Successfully utilized Apple Silicon GPU

# Sequential evaluation  

python evaluate_multi_stage.py --model models/mario_DQN_autodl_rtx5090_5000000_steps.zip### **Weaknesses**

```1. ⚠️ **Limited Generalization**: Failed to master stage 1-2 (0% completion)

2. ⚠️ **Single-Stage Overfitting**: Despite multi-stage training, only learned 1-1

### Resume Training3. ⚠️ **Training Distribution**: Likely spent disproportionate time on stage 1-1

```bash4. ⚠️ **Exploration Decay**: May have reduced exploration too early

python mario_dqn_autodl.py --resume --timesteps 10000000

```### **Potential Improvements**

1. **Curriculum Learning**: Force exposure to later stages

## Dependencies2. **Longer Training**: 10M-20M steps may be needed for multi-stage mastery

3. **Stage-Specific Rewards**: Bonus rewards for reaching new stages

- Python 3.11+4. **Prioritized Stage Sampling**: Ensure balanced experience across all stages

- PyTorch (MPS/CUDA support)

- Stable-Baselines3---

- Gymnasium

- Custom Mario environment (included)## 🔬 Technical Achievements



## Hardware Requirements### **Implementation Highlights**

1. **MPS GPU Acceleration**: Successfully leveraged Apple Silicon for ~3× speedup

### Minimum (Local Training)2. **Gymnasium Compatibility**: Handled both old gym and new gymnasium APIs

- 16GB RAM3. **Efficient Preprocessing**: Optimized wrapper chain for real-time processing

- Apple Silicon M1/M2/M3/M4 or NVIDIA GPU4. **Stable Training**: No crashes or divergence over 5M steps

- 50GB storage space5. **Reproducible Results**: Consistent evaluation outcomes



### Recommended (Cloud Training)### **Code Quality**

- 24GB VRAM (RTX 5090/4090)- Clean architecture with modular design

- 32GB System RAM- Comprehensive logging and monitoring

- 100GB NVMe storage- Automatic checkpointing every 250K steps

- Video recording for qualitative analysis

## Acknowledgments- JSON export for quantitative analysis



This implementation builds upon:---

- Mnih et al. (2015) - Deep Q-Network foundations

- IMPALA CNN architecture (Espeholt et al., 2018)## 📝 Conclusions

- Stable-Baselines3 framework

- Gymnasium interface standards### **Success Criteria**

- ✅ Complete 5M step training run

## Citation- ✅ Achieve >50% completion rate on stage 1-1

- ✅ Generate training metrics and visualizations

If you use this code in your research, please cite:- ✅ Record gameplay videos for analysis

- ⚠️ Multi-stage generalization (partial success)

```bibtex

@misc{yang2024dqn_mario,### **Key Findings**

  title={Deep Q-Network Implementation for Super Mario Bros: A Study of Specialization-Generalization Trade-offs},1. DQN can effectively learn to complete Super Mario Bros stage 1-1 with 100% success

  author={Yang, Ricki},2. IMPALA CNN architecture is sufficient for visual feature extraction

  year={2024},3. MPS acceleration on M4 Pro provides viable alternative to CUDA GPUs

  institution={University of Technology Sydney},4. Multi-stage generalization requires specific training strategies beyond simple exposure

  course={43008 Reinforcement Learning}5. 5M steps is sufficient for single-stage mastery but insufficient for multi-stage

}

```### **Future Work**
1. Implement curriculum learning for progressive stage difficulty
2. Extend training to 10M-20M steps
3. Experiment with PPO or A3C for better exploration
4. Add stage-specific reward shaping
5. Compare performance across different algorithm families

---

## 📖 References

- Mnih, V., et al. (2015). Human-level control through deep reinforcement learning. *Nature*.
- Espeholt, L., et al. (2018). IMPALA: Scalable Distributed Deep-RL. *ICML*.
- Van Hasselt, H., et al. (2016). Deep Reinforcement Learning with Double Q-Learning. *AAAI*.

---

## 🔗 File Locations

**Training Script**: `code/mario_dqn_local_m4.py`  
**Environment Wrappers**: `code/mario.py`  
**Trained Model**: `models/mario_DQN_5000000_steps.zip`  
**Training Data**: `results/q_value_monitor.csv`  
**Evaluation Results**: `results/evaluation_single_stage.json`  
**Videos**: `videos/single_stage/` and `videos/continuous/`

---

## ✅ Verification

To verify results, run:
```bash
# Evaluate single-stage performance
python code/evaluate_final_model.py

# Evaluate continuous multi-stage performance
python code/evaluate_multi_stage.py
```

Expected output:
- Single-stage: 100% completion rate on 1-1
- Continuous: 100% completion on 1-1, 0% on 1-2

---

**Submission Date**: October 27, 2024  
**Contact**: Ricki Yang (25657673)

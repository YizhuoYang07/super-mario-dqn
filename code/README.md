# Source Code Documentation

This directory contains the complete DQN implementation and evaluation framework.

## Core Files

### 1. Training Implementation
- **mario_dqn_autodl.py** - Main DQN training script (AutoDL RTX 5090 optimized)
- **mario_dqn_local_m4.py** - M4 Pro optimized version for local training
- **mario.py** - Environment wrappers and preprocessing pipeline

### 2. Evaluation Framework
- **evaluate_final_model.py** - Single-stage evaluation with video recording
- **evaluate_multi_stage.py** - Continuous multi-stage evaluation

## Architecture Overview

### DQN Implementation
```python
# Core Components
- IMPALA CNN backbone (ResNet-style)
- Double DQN with target network
- Experience replay buffer (50K-100K capacity)
- Epsilon-greedy exploration
- Hardware-optimized configurations
```

### Environment Pipeline
```python
# Preprocessing Chain
NoopResetEnv → MaxAndSkipEnv → WarpFrame → 
FrameStack → RewardScaling → ActionWrapper
```

## Technical Specifications

### Hardware Optimization
- **AutoDL RTX 5090**: Large buffer (100K), high batch size (512)
- **M4 Pro**: Memory-efficient buffer (50K), MPS-optimized batch size (256)

### Training Hyperparameters
```python
learning_rate = 1.4e-5
gamma = 0.95
target_update_interval = 10_000
exploration_fraction = 0.15
total_timesteps = 5_000_000
```

### Evaluation Protocols
- **Single-stage**: 10 episodes on World 1-1
- **Continuous**: 5 episodes progressing 1-1 → 1-2 → ...
- **Deterministic**: Policy without exploration for consistent evaluation

## Code Quality Standards

### Documentation
- Comprehensive docstrings for all functions
- Type hints for improved maintainability
- Inline comments explaining algorithmic choices

### Error Handling
- Robust checkpoint loading/saving
- Graceful degradation for missing dependencies
- Hardware compatibility checks

### Reproducibility
- Fixed random seeds
- Deterministic evaluation protocols
- Comprehensive logging

## Usage Examples

### Training
```bash
# Full training run
python mario_dqn_autodl.py --timesteps 5000000

# Resume from checkpoint  
python mario_dqn_autodl.py --resume --timesteps 10000000
```

### Evaluation
```bash
# Single-stage evaluation
python evaluate_final_model.py --model ../models/mario_DQN_autodl_rtx5090_5000000_steps.zip

# Multi-stage evaluation
python evaluate_multi_stage.py --model ../models/mario_DQN_autodl_rtx5090_5000000_steps.zip
```

## Dependencies

See `../requirements.txt` for complete dependency list. Key requirements:
- PyTorch 2.0+ (CUDA/MPS support)
- Stable-Baselines3 2.1+
- Gymnasium 0.29+
- Custom Mario environment (included)
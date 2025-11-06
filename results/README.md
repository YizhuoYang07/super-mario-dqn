# Experiment Results Documentation

This directory contains detailed experimental results and analysis for the DQN implementation.

## Contents

### Training Metrics
- `training_summary.json` - Complete training statistics and Q-value progression
- `q_value_monitor.csv` - Timestep-by-timestep Q-value evolution data

### Evaluation Results
- `evaluation_single_stage.json` - Performance on World 1-1 (training distribution)
- `evaluation_continuous.json` - Sequential evaluation across World 1-1 → 1-2

## Key Findings Summary

### Training Performance (5M Steps)
- **Q-value Growth**: 68.2% increase (1.024 → 1.723)
- **Peak Q-value**: 4.350 at 500K steps
- **Training Stability**: Moderate volatility in later phases
- **Convergence**: Partial (still improving at termination)

### Evaluation Performance
- **World 1-1**: 100% completion rate (10/10 episodes)
- **World 1-2**: 0% completion rate (immediate failure)
- **Mean Reward (1-1)**: 3,296 ± 936
- **Episode Length**: 345 steps (highly consistent)

### Research Implications
These results demonstrate the **specialization-generalization trade-off**:
- Perfect performance on training distribution (World 1-1)
- Complete failure on adjacent environment (World 1-2)
- Evidence of overfitting to specific environment features

## Data Format Specifications

### training_summary.json
```json
{
  "total_steps": int,
  "q_value_growth_pct": float,
  "early_mean_q": float,
  "late_mean_q": float,
  "peak_q_value": float,
  "early_volatility": float,
  "late_volatility": float,
  "stability_improvement_pct": float
}
```

### evaluation_single_stage.json
```json
{
  "model_path": str,
  "n_episodes": int,
  "mean_reward": float,
  "std_reward": float,
  "completion_rate": float,
  "episode_rewards": [float, ...],
  "episode_lengths": [int, ...],
  "episode_completions": [bool, ...]
}
```

### q_value_monitor.csv
```
step,mean_q_value,max_q_value,min_q_value,std_q_value
```

## Reproducibility

All results can be reproduced using:
1. Model: `../models/mario_DQN_autodl_rtx5090_5000000_steps.zip`
2. Evaluation scripts: `../code/evaluate_*.py`
3. Environment: World 1-1 for single-stage, 1-1→1-2 for continuous

## Academic Context

These results support the hypothesis that training distribution diversity creates fundamental trade-offs between task mastery and environmental adaptability in deep reinforcement learning.
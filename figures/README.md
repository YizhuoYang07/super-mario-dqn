# Visualization Gallery

This directory contains key visualizations and figures from the DQN training and evaluation experiments.

## Figure Descriptions

### 1. Training Performance
- **dqn_learning_summary.png/pdf** - Complete training curves showing Q-value progression over 5M steps
- **dqn_qvalue_evolution.png/pdf** - Detailed Q-value evolution with volatility analysis

### 2. Evaluation Results  
- **dqn_evaluation_results.png/pdf** - Performance comparison between single-stage and continuous evaluation

## Key Visual Insights

### Q-Value Progression
The learning curves demonstrate:
- Initial rapid learning phase (0-500K steps)
- Peak performance around 500K steps (Q-value = 4.350)
- Gradual decline and stabilization (1M-5M steps)
- Final 68.2% net improvement in Q-values

### Performance Analysis
The evaluation charts reveal:
- Perfect specialization on World 1-1 (100% completion)
- Complete generalization failure on World 1-2 (0% completion)
- Consistent reward patterns within training distribution
- Immediate performance collapse outside training distribution

## Technical Specifications

### Figure Generation
All figures generated using:
- **Data Source**: Q-value monitoring during training (`../results/q_value_monitor.csv`)
- **Evaluation Data**: Performance metrics from `../results/evaluation_*.json`
- **Tools**: Matplotlib with academic publication styling
- **Format**: Both PNG (web) and PDF (print) versions provided

### Usage in Academic Context
These visualizations support the core thesis on specialization-generalization trade-offs in deep reinforcement learning and can be referenced in academic publications with proper citation.

## Reproduction
Figures can be regenerated using the data analysis scripts in `../code/` with the corresponding result files.
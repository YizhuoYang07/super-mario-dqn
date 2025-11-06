# Video Documentation

This directory contains representative gameplay videos demonstrating agent performance across different evaluation scenarios.

## Video Contents

### 1. Single-Stage Evaluation
**File**: `mario_dqn_eval-step-345-to-step-2345.mp4`
- **Description**: Agent playing World 1-1 only
- **Performance**: Perfect completion in 345 steps
- **Purpose**: Demonstrates mastery of training distribution

### 2. Continuous Progression Evaluation  
**File**: `mario_continuous-step-2348-to-step-12348.mp4`
- **Description**: Agent progressing from World 1-1 to World 1-2
- **Performance**: Completes 1-1 (345 steps) then fails in 1-2 (~2000 steps)
- **Purpose**: Shows generalization limits at environment transition

## Technical Details

### Recording Specifications
- **Frame Rate**: 30 FPS (appears 4x speed due to frame_skip=4)
- **Resolution**: 240x256 (original NES resolution)
- **Format**: MP4 with H.264 encoding
- **Duration**: Variable based on episode length

### Episode Analysis

#### World 1-1 Performance
- **Strategy**: Direct path to flag pole
- **Consistency**: Identical 345-step completion across all episodes
- **Success Rate**: 100% (10/10 evaluation episodes)

#### World 1-2 Performance  
- **Initial Behavior**: Maintains 1-1 movement patterns
- **Failure Mode**: Cannot adapt to underground level mechanics
- **Survival Time**: ~2000 steps before death
- **Success Rate**: 0% (0/5 evaluation episodes)

## Research Significance

These videos provide qualitative evidence for the core research findings:

1. **Specialization Success**: Perfect, consistent performance on training environment
2. **Generalization Failure**: Immediate breakdown when environment changes
3. **Behavioral Analysis**: Agent applies learned 1-1 strategies inappropriately in 1-2

## Viewing Instructions

Videos can be viewed with any standard media player. The apparent 4x speed is due to the frame skipping preprocessing (frame_skip=4) used during training and evaluation.

## Academic Use

These videos serve as supplementary material for the research on specialization-generalization trade-offs in deep reinforcement learning and can be referenced in academic presentations or publications.
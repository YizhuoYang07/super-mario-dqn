"""
Deep Q-Network (DQN) Training for Super Mario Bros Environment
LOCAL M4 PRO OPTIMIZED - 5M Steps Configuration

Memory-efficient configuration for Apple Silicon M4 Pro with 24GB RAM.
Optimized for 5M timesteps (~6 hours training duration).

Hardware Optimization for M4 Pro:
- Unified Memory: 24GB (conservative batch/buffer sizing)
- Batch size: 256 (MPS optimized)
- Buffer size: 50K (memory friendly)
- Expected performance: 800K-1M steps/hour
- Target: 5M steps in 5-6 hours

Key Differences from Cloud Version:
- Smaller batch size (256 vs 512) for MPS compatibility
- Smaller buffer (50K vs 100K) for memory efficiency  
- Lower total timesteps (5M default vs 10M)
- No RTX 5090 CUDA workarounds

Authors: Ricki Yang
Course: 43008 - Reinforcement Learning
Institution: University of Technology Sydney
Date: October 2024
Platform: MacBook Pro M4 Pro 24GB
"""

import os
import sys
import time
import subprocess
import json
import argparse
import glob
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import (
    BaseCallback, 
    CheckpointCallback, 
    EvalCallback,
    StopTrainingOnNoModelImprovement
)
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# Import Mario environment components
from mario import make_mario_env, ImpalaCNN


def safe_env_step(env, action) -> Tuple[Any, float, bool, bool, Dict]:
    """
    Safely handle environment step() with compatibility for both gym and gymnasium formats.
    Also handles vectorized environments where info might be a list.
    
    Args:
        env: Environment instance
        action: Action to take
        
    Returns:
        Tuple of (obs, reward, terminated, truncated, info)
    """
    step_result = env.step(action)
    
    if len(step_result) == 4:
        # Old gym format: (obs, reward, done, info)
        obs, reward, done, info = step_result
        terminated = done
        truncated = False
    elif len(step_result) == 5:
        # New gymnasium format: (obs, reward, terminated, truncated, info)
        obs, reward, terminated, truncated, info = step_result
    else:
        raise ValueError(f"Unexpected step result length: {len(step_result)}")
    
    # Handle vectorized environment where info is a list
    if isinstance(info, list) and len(info) > 0:
        info = info[0]
    
    # Ensure info is a dictionary
    if not isinstance(info, dict):
        info = {}
    
    return obs, reward, terminated, truncated, info


def safe_env_reset(env) -> Any:
    """
    Safely handle environment reset() with compatibility for both gym and gymnasium formats.
    Also handles vectorized environments.
    
    Args:
        env: Environment instance
        
    Returns:
        Initial observation
    """
    reset_result = env.reset()
    
    if isinstance(reset_result, tuple):
        # New gymnasium format: (obs, info)
        obs, info = reset_result
        if isinstance(info, list) and len(info) > 0:
            info = info[0]
    else:
        # Old gym format: obs
        obs = reset_result
    
    return obs


class CompletionAwareEvalCallback(BaseCallback):
    """
    Evaluation callback that tracks true completion rate instead of just rewards.
    
    This callback evaluates the model at specified intervals and tracks both
    reward performance and actual level completion (flag_get) to prevent
    reward hacking scenarios from being considered successful.
    
    Args:
        eval_env: Environment for evaluation
        eval_freq: Frequency of evaluation (in training steps)
        n_eval_episodes: Number of episodes per evaluation
        deterministic: Whether to use deterministic policy
        verbose: Verbosity level
    """
    
    def __init__(
        self,
        eval_env: gym.Env,
        eval_freq: int = 50000,
        n_eval_episodes: int = 10,
        deterministic: bool = True,
        verbose: int = 1
    ):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.deterministic = deterministic
        self.evaluations_results = []
        self.evaluations_timesteps = []
        self.best_completion_rate = -np.inf
        self.best_mean_reward = -np.inf
        
    def _on_step(self) -> bool:
        """Evaluate model performance at specified frequency."""
        if self.n_calls % self.eval_freq == 0:
            # Perform evaluation with completion tracking
            eval_results = self._evaluate_with_completion_tracking()
            
            # Store results
            self.evaluations_results.append(eval_results)
            self.evaluations_timesteps.append(self.num_timesteps)
            
            # Log to TensorBoard
            self.logger.record("eval/mean_reward", eval_results["mean_reward"])
            self.logger.record("eval/true_completion_rate", eval_results["true_completion_rate"])
            self.logger.record("eval/reward_hacking_rate", eval_results["reward_hacking_rate"])
            self.logger.record("eval/mean_episode_length", eval_results["mean_episode_length"])
            
            # Update best performance
            if eval_results["true_completion_rate"] > self.best_completion_rate:
                self.best_completion_rate = eval_results["true_completion_rate"]
                if self.verbose > 0:
                    print(f"New best completion rate: {self.best_completion_rate:.1f}%")
                    
            if eval_results["mean_reward"] > self.best_mean_reward:
                self.best_mean_reward = eval_results["mean_reward"]
            
            if self.verbose > 0:
                print(f"Eval at {self.num_timesteps} steps:")
                print(f"  Completion Rate: {eval_results['true_completion_rate']:.1f}%")
                print(f"  Mean Reward: {eval_results['mean_reward']:.2f}")
                print(f"  Reward Hacking: {eval_results['reward_hacking_rate']:.1f}%")
        
        return True
    
    def _evaluate_with_completion_tracking(self) -> Dict[str, float]:
        """Evaluate model with completion tracking."""
        episode_rewards = []
        completion_count = 0
        high_reward_no_completion = 0
        episode_lengths = []
        high_reward_threshold = 1000
        
        for _ in range(self.n_eval_episodes):
            obs = safe_env_reset(self.eval_env)
            episode_reward = 0
            episode_length = 0
            completed = False
            
            while True:
                action, _ = self.model.predict(obs, deterministic=self.deterministic)
                obs, reward, terminated, truncated, info = safe_env_step(self.eval_env, action)
                
                episode_reward += reward
                episode_length += 1
                
                # Check for true completion
                if info.get("flag_get", False):
                    completed = True
                    completion_count += 1
                    break
                    
                # Check if episode is done
                done = terminated or truncated
                if done:
                    if episode_reward > high_reward_threshold:
                        high_reward_no_completion += 1
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        return {
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "true_completion_rate": (completion_count / self.n_eval_episodes) * 100,
            "reward_hacking_rate": (high_reward_no_completion / self.n_eval_episodes) * 100,
            "mean_episode_length": np.mean(episode_lengths),
            "completion_count": completion_count,
        }


class QValueMonitorCallback(BaseCallback):
    """
    Callback for monitoring Q-value statistics during DQN training.
    
    This callback records mean, standard deviation, maximum, and minimum Q-values
    at specified intervals for training analysis and convergence monitoring.
    
    Args:
        log_freq: Frequency of Q-value logging (in training steps)
        verbose: Verbosity level for logging output
    """
    
    def __init__(self, log_freq: int = 10000, verbose: int = 0) -> None:
        super().__init__(verbose)
        self.log_freq = log_freq
        self.q_values_history = []
        
    def _on_step(self) -> bool:
        """Monitor Q-values at specified frequency."""
        if self.n_calls % self.log_freq == 0:
            if hasattr(self.model, "q_net") and hasattr(self.model, "replay_buffer"):
                # Ensure replay buffer has sufficient samples
                if self.model.replay_buffer.size() > self.model.batch_size:
                    try:
                        # Sample batch from replay buffer for Q-value analysis
                        replay_data = self.model.replay_buffer.sample(
                            self.model.batch_size, env=self.model._vec_normalize_env
                        )
                        with th.no_grad():
                            q_values = self.model.q_net(replay_data.observations)
                            q_values_np = q_values.cpu().numpy()
                            
                        # Compute Q-value statistics
                        stats = {
                            "step": self.num_timesteps,
                            "mean_q_value": np.mean(q_values_np),
                            "std_q_value": np.std(q_values_np),
                            "max_q_value": np.max(q_values_np),
                            "min_q_value": np.min(q_values_np),
                        }
                        
                        self.q_values_history.append(stats)
                        
                        # Log to TensorBoard
                        self.logger.record("train/mean_q_value", stats["mean_q_value"])
                        self.logger.record("train/std_q_value", stats["std_q_value"])
                        self.logger.record("train/max_q_value", stats["max_q_value"])
                        self.logger.record("train/min_q_value", stats["min_q_value"])
                        
                        if self.verbose > 0:
                            print(f"Step {self.num_timesteps}: Q-values - "
                                  f"Mean: {stats['mean_q_value']:.3f}, "
                                  f"Std: {stats['std_q_value']:.3f}")
                                  
                    except Exception as e:
                        if self.verbose > 0:
                            print(f"Warning: Q-value monitoring failed: {e}")
        return True
    
    def save_q_values(self, filepath: str) -> None:
        """Save Q-value history to CSV file."""
        if self.q_values_history:
            df = pd.DataFrame(self.q_values_history)
            df.to_csv(filepath, index=False)
            if self.verbose > 0:
                print(f"Q-value monitoring data saved to: {filepath}")


def setup_device() -> th.device:
    """
    Configure and return the appropriate device for training.
    
    Prioritizes CUDA for AutoDL RTX 5090 instances.
    
    Returns:
        torch.device: Configured device for model training
    """
    if th.cuda.is_available():
        device = th.device("cuda")
        gpu_count = th.cuda.device_count()
        current_device = th.cuda.current_device()
        gpu_name = th.cuda.get_device_name(current_device)
        gpu_memory = th.cuda.get_device_properties(current_device).total_memory / 1e9
        
        print("Device Configuration: NVIDIA CUDA")
        print(f"  - GPU Count: {gpu_count}")
        print(f"  - Current GPU: {gpu_name}")
        print(f"  - GPU Memory: {gpu_memory:.1f}GB")
        print(f"  - CUDA Version: {th.version.cuda}")
        
        if "5090" in gpu_name or gpu_memory >= 24:
            print("  - Optimized for: AutoDL RTX 5090 (24GB VRAM)")
            print("  - Configuration: Batch 512, Buffer 100K, Aggressive training")
            print("  - Expected performance: 3-5M steps/hour")
        else:
            print(f"  - Detected GPU: {gpu_name}")
            print("  - Warning: Configuration optimized for RTX 5090")
            
    elif th.backends.mps.is_available():
        device = th.device("mps")
        print("Device Configuration: Apple Silicon MPS")
        print("  - Device: M4 Pro (24GB RAM)")
        print("  - Configuration: Batch 256, Buffer 50K, Memory-friendly")
        print("  - Expected performance: 800K-900K steps/hour")
    else:
        device = th.device("cpu")
        print("Device Configuration: CPU")
        print("  - Warning: This script requires GPU for optimal performance")
        print("  - Training will be significantly slower on CPU")
        
    return device


def create_experiment_folder(resume_exp_dir: Optional[str] = None) -> Tuple[int, str, str, str, str]:
    """
    Create or resume experiment directory structure with unique numbering.
    
    Automatically increments experiment number to avoid overwriting
    previous results. Creates subdirectories for models, logs, and videos.
    
    Args:
        resume_exp_dir: If provided, resume from this experiment directory
    
    Returns:
        Tuple containing:
        - experiment_number: Unique experiment identifier
        - experiment_dir: Main experiment directory path
        - model_dir: Model checkpoint directory path
        - log_dir: TensorBoard log directory path
        - video_dir: Evaluation video directory path
    """
    results_dir = Path("results/dqn_autodl")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    if resume_exp_dir:
        # Resume from existing experiment
        exp_dir = Path(resume_exp_dir)
        if not exp_dir.exists():
            raise ValueError(f"Resume experiment directory does not exist: {resume_exp_dir}")
        
        exp_num = int(exp_dir.name.replace("exp", ""))
        print(f"Resuming from existing experiment: {exp_dir}")
        
    else:
        # Find next available experiment number
        exp_num = 1
        while (results_dir / f"exp{exp_num}").exists():
            exp_num += 1
        
        # Create new experiment directory structure
        exp_dir = results_dir / f"exp{exp_num}"
        print(f"Creating new experiment: {exp_dir}")
    
    # Ensure all subdirectories exist
    model_dir = exp_dir / "models"
    log_dir = exp_dir / "logs"
    video_dir = exp_dir / "videos"
    
    for directory in [exp_dir, model_dir, log_dir, video_dir]:
        directory.mkdir(parents=True, exist_ok=True)
        
    print(f"Experiment Directory: {exp_dir}")
    return exp_num, str(exp_dir), str(model_dir), str(log_dir), str(video_dir)


def start_tensorboard(log_dir: str, port: int = 6006) -> None:
    """
    Launch TensorBoard server for monitoring training progress.
    
    Optimized for AutoDL JupyterLab environment with proper host binding.
    
    Args:
        log_dir: Directory containing TensorBoard logs
        port: Port number for TensorBoard server (default: 6006)
    """
    try:
        tensorboard_cmd = [
            sys.executable, "-m", "tensorboard.main",
            "--logdir", log_dir,
            "--port", str(port),
            "--host", "0.0.0.0",
            "--reload_interval", "30"
        ]
        
        print("Starting TensorBoard for AutoDL environment...")
        print(f"TensorBoard available at: http://localhost:{port}")
        print(f"Log directory: {log_dir}")
        
        subprocess.Popen(
            tensorboard_cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        print("TensorBoard server started successfully!")
        
    except Exception as e:
        print(f"Warning: Failed to start TensorBoard: {e}")
        print(f"You can manually start TensorBoard with:")
        print(f"  tensorboard --logdir {log_dir} --port {port} --host 0.0.0.0")


def find_latest_checkpoint(model_dir: str) -> Optional[str]:
    """
    Find the latest checkpoint file in the model directory.
    
    Searches for checkpoint files with pattern *_steps.zip and returns the one
    with the highest step number.
    
    Args:
        model_dir: Directory containing model checkpoints
        
    Returns:
        Path to latest checkpoint file, or None if no checkpoints found
    """
    if not os.path.exists(model_dir):
        return None
    
    # Search for checkpoint files
    checkpoint_patterns = [
        os.path.join(model_dir, "*_steps.zip"),
        os.path.join(model_dir, "*autodl*.zip"),
        os.path.join(model_dir, "*.zip")
    ]
    
    checkpoints = []
    for pattern in checkpoint_patterns:
        checkpoints.extend(glob.glob(pattern))
    
    if not checkpoints:
        return None
    
    # Sort by modification time (most recent first)
    checkpoints.sort(key=os.path.getmtime, reverse=True)
    
    # Try to extract step number from filename for better sorting
    def extract_steps(filename):
        basename = os.path.basename(filename)
        if "_steps.zip" in basename:
            try:
                return int(basename.split("_steps.zip")[0].split("_")[-1])
            except (ValueError, IndexError):
                pass
        return 0
    
    # Sort by step number if available
    checkpoints_with_steps = [(f, extract_steps(f)) for f in checkpoints]
    checkpoints_with_steps.sort(key=lambda x: x[1], reverse=True)
    
    latest_checkpoint = checkpoints_with_steps[0][0]
    print(f"Found latest checkpoint: {latest_checkpoint}")
    return latest_checkpoint


def find_matching_files(checkpoint_path: str) -> Dict[str, Optional[str]]:
    """
    Find matching replay buffer and vecnormalize files for a checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        
    Returns:
        Dictionary with paths to replay_buffer and vecnormalize files
    """
    base_path = checkpoint_path.replace(".zip", "")
    model_dir = os.path.dirname(checkpoint_path)
    
    # Extract step number or identifier from checkpoint filename
    basename = os.path.basename(base_path)
    
    # Search for matching files
    replay_buffer_path = None
    vecnormalize_path = None
    
    # Common patterns for replay buffer and vecnormalize files
    possible_patterns = [
        f"{base_path}_replay_buffer.pkl",
        f"{base_path.replace('mario_DQN', 'mario_DQN')}_replay_buffer.pkl",
        os.path.join(model_dir, f"*{basename.split('_')[-1]}_replay_buffer.pkl"),
    ]
    
    for pattern in possible_patterns:
        matches = glob.glob(pattern)
        if matches:
            replay_buffer_path = matches[0]
            break
    
    vecnorm_patterns = [
        f"{base_path}_vecnormalize.pkl",
        f"{base_path.replace('mario_DQN', 'mario_DQN')}_vecnormalize.pkl",
        os.path.join(model_dir, f"*{basename.split('_')[-1]}_vecnormalize.pkl"),
    ]
    
    for pattern in vecnorm_patterns:
        matches = glob.glob(pattern)
        if matches:
            vecnormalize_path = matches[0]
            break
    
    return {
        "replay_buffer": replay_buffer_path,
        "vecnormalize": vecnormalize_path
    }


def load_checkpoint_with_env(
    checkpoint_path: str,
    env: gym.Env,
    device: th.device
) -> Tuple[DQN, Optional[str], Optional[str]]:
    """
    Load model from checkpoint with proper environment setup.
    
    Args:
        checkpoint_path: Path to model checkpoint
        env: Training environment
        device: Training device
        
    Returns:
        Tuple of (loaded_model, replay_buffer_path, vecnormalize_path)
    """
    print(f"Loading model from checkpoint: {checkpoint_path}")
    
    try:
        # Load the model
        model = DQN.load(
            checkpoint_path,
            env=env,
            device=device,
            print_system_info=False
        )
        
        # Find matching auxiliary files
        aux_files = find_matching_files(checkpoint_path)
        
        print(f"Model loaded successfully from: {checkpoint_path}")
        if aux_files["replay_buffer"]:
            print(f"Found replay buffer: {aux_files['replay_buffer']}")
        if aux_files["vecnormalize"]:
            print(f"Found vecnormalize: {aux_files['vecnormalize']}")
            
        return model, aux_files["replay_buffer"], aux_files["vecnormalize"]
        
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        raise


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments for training configuration."""
    parser = argparse.ArgumentParser(
        description="DQN Training for Super Mario Bros - AutoDL RTX 5090 Optimized with Resume Support",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--resume", 
        action="store_true",
        help="Resume training from latest checkpoint"
    )
    
    parser.add_argument(
        "--checkpoint", 
        type=str,
        help="Specific checkpoint file to resume from"
    )
    
    parser.add_argument(
        "--timesteps", 
        type=int, 
        default=5_000_000,  # M4 Pro default: 5M steps (~6 hours)
        help="Total training timesteps"
    )
    
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["auto", "cuda", "mps", "cpu"],
        default="auto",
        help="Training device"
    )
    
    return parser.parse_args()


def comprehensive_evaluation(
    model: DQN,
    eval_env: gym.Env,
    n_eval_episodes: int = 10,
    deterministic: bool = True
) -> Dict[str, Any]:
    """
    Perform comprehensive evaluation of trained model with completion tracking.
    
    Evaluates the model on multiple episodes and computes detailed statistics
    including TRUE completion rate (flag_get), reward vs completion analysis,
    and detection of reward hacking scenarios.
    
    Args:
        model: Trained DQN model
        eval_env: Environment for evaluation
        n_eval_episodes: Number of episodes to evaluate
        deterministic: Whether to use deterministic policy
        
    Returns:
        Dictionary containing evaluation metrics including completion tracking
    """
    print(f"Evaluating model performance with completion tracking...")
    print(f"Episodes: {n_eval_episodes}, Deterministic: {deterministic}")
    
    episode_rewards = []
    episode_lengths = []
    completion_count = 0
    high_reward_no_completion = 0
    completion_rewards = []
    non_completion_rewards = []
    reward_hacking_episodes = []
    
    high_reward_threshold = 1000
    
    for episode in range(n_eval_episodes):
        obs = safe_env_reset(eval_env)
        episode_reward = 0
        episode_length = 0
        completed = False
        
        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = safe_env_step(eval_env, action)
            
            episode_reward += reward
            episode_length += 1
            
            # Check for true level completion (flag_get)
            if info.get("flag_get", False):
                completed = True
                completion_count += 1
                completion_rewards.append(episode_reward)
                break
                
            # Check if episode is done
            done = terminated or truncated
            if done:
                non_completion_rewards.append(episode_reward)
                
                # Detect potential reward hacking
                if episode_reward > high_reward_threshold:
                    high_reward_no_completion += 1
                    reward_hacking_episodes.append({
                        "episode": episode + 1,
                        "reward": episode_reward,
                        "length": episode_length,
                        "reason": "High reward without completion"
                    })
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
    
    # Compute comprehensive statistics
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    completion_rate = (completion_count / n_eval_episodes) * 100
    reward_hacking_rate = (high_reward_no_completion / n_eval_episodes) * 100
    
    results = {
        "n_episodes": n_eval_episodes,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_episode_length": np.mean(episode_lengths),
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        
        # Completion tracking metrics
        "true_completion_rate": completion_rate,
        "completion_count": completion_count,
        "mean_completion_reward": np.mean(completion_rewards) if completion_rewards else 0,
        "mean_non_completion_reward": np.mean(non_completion_rewards) if non_completion_rewards else 0,
        
        # Reward hacking detection
        "reward_hacking_rate": reward_hacking_rate,
        "high_reward_no_completion_count": high_reward_no_completion,
        "reward_hacking_episodes": reward_hacking_episodes,
        "high_reward_threshold": high_reward_threshold,
    }
    
    # Print detailed evaluation summary
    print("=" * 60)
    print("COMPREHENSIVE EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Episodes Evaluated: {n_eval_episodes}")
    print(f"  Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"  Mean Episode Length: {np.mean(episode_lengths):.1f}")
    print(f"  Reward Range: {np.min(episode_rewards):.2f} to {np.max(episode_rewards):.2f}")
    print()
    print("COMPLETION ANALYSIS:")
    print(f"  True Completion Rate: {completion_rate:.1f}% ({completion_count}/{n_eval_episodes})")
    
    if completion_rewards:
        print(f"  Mean Completion Reward: {np.mean(completion_rewards):.2f}")
    if non_completion_rewards:
        print(f"  Mean Non-Completion Reward: {np.mean(non_completion_rewards):.2f}")
    
    print()
    print("REWARD HACKING ANALYSIS:")
    print(f"  High Reward, No Completion: {reward_hacking_rate:.1f}% ({high_reward_no_completion}/{n_eval_episodes})")
    print(f"  Threshold Used: {high_reward_threshold} points")
    
    if reward_hacking_episodes:
        print(f"  Detected Episodes:")
        for ep_info in reward_hacking_episodes:
            reward_val = float(ep_info['reward']) if hasattr(ep_info['reward'], '__iter__') else ep_info['reward']
            length_val = int(ep_info['length']) if hasattr(ep_info['length'], '__iter__') else ep_info['length']
            print(f"    Episode {ep_info['episode']}: {reward_val:.0f} points, {length_val} steps")
    
    print("=" * 60)
    
    return results


def main():
    """
    Main training function for DQN on Super Mario Bros - AutoDL RTX 5090 Optimized.
    
    Implements complete training pipeline with aggressive hyperparameters
    optimized for 8-hour training window on RTX 5090 GPU instances.
    Includes resume from checkpoint support for interrupted training.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    print("=" * 80)
    print("DQN Training for Super Mario Bros - M4 Pro Local Optimized")
    print("=" * 80)
    
    # Configure device
    if args.device == "auto":
        device = setup_device()
    else:
        device = th.device(args.device)
        print(f"Using specified device: {device}")
    
    # Determine resume configuration
    resume_from_checkpoint = False
    checkpoint_path = None
    resume_exp_dir = None
    
    if args.checkpoint:
        # Specific checkpoint file provided
        if not os.path.exists(args.checkpoint):
            raise FileNotFoundError(f"Checkpoint file not found: {args.checkpoint}")
        checkpoint_path = args.checkpoint
        resume_from_checkpoint = True
        # Extract experiment directory from checkpoint path
        resume_exp_dir = str(Path(args.checkpoint).parent.parent)
        print(f"Resuming from specific checkpoint: {checkpoint_path}")
        
    elif args.resume:
        # Auto-find latest experiment
        results_dir = Path("results/dqn_autodl")
        if results_dir.exists():
            exp_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("exp")])
            if exp_dirs:
                latest_exp = exp_dirs[-1]
                model_dir = latest_exp / "models"
                checkpoint_path = find_latest_checkpoint(str(model_dir))
                if checkpoint_path:
                    resume_from_checkpoint = True
                    resume_exp_dir = str(latest_exp)
                    print(f"Auto-resuming from latest experiment: {latest_exp}")
                else:
                    print(f"No checkpoints found in latest experiment {latest_exp}")
            else:
                print("No previous experiments found")
    
    # Create or resume experiment folder
    exp_num, exp_dir, model_dir, log_dir, video_dir = create_experiment_folder(resume_exp_dir)
    
    # Start TensorBoard monitoring
    start_tensorboard(log_dir)
    
    # Environment configuration - Multi-level training for generalization
    training_stages = [
        '1-1', '1-2', '1-3', '1-4',
        '2-1', '2-2', '2-3', '2-4',
        '3-1', '3-2', '3-3', '3-4',
        '4-1', '4-2', '4-3', '4-4'
    ]
    
    n_envs = 4  # Parallel environments for distributed training
    
    print("\nEnvironment Configuration:")
    print(f"  Training stages: {len(training_stages)} levels (worlds 1-4)")
    print(f"  Parallel environments: {n_envs} (distributed training)")
    print("  Frame processing: 84x84 grayscale, 4-frame stack")
    print("  Action space: SIMPLE_MOVEMENT (7 actions)")
    print("  Random noop reset: Disabled (compatibility mode)")
    print("  Multi-level generalization: Enabled")
    
    # Create training environment - use world 1-1 as starting point
    # With use_single_stage_episodes=False, Mario will continue to next stages after completion
    train_env = make_mario_env(
        "SuperMarioBros-1-1-v0",
        n_envs=n_envs,
        env_kwargs={},
        wrapper_kwargs={
            "frame_skip": 4,
            "screen_size": 84,
            "noop_max": 0,  # Temporarily disabled due to nes_py compatibility issue
            "use_single_stage_episodes": False,  # Continue to next stage after completion
        },
        vec_normalize_kwargs={
            "training": True,
            "norm_obs": False,
            "norm_reward": True,
            "clip_obs": 10.0,
            "clip_reward": 10.0,
            "gamma": 0.95,
        },
    )
    
    # Create evaluation environment - also start from 1-1, allow stage progression
    eval_env = make_mario_env(
        "SuperMarioBros-1-1-v0",
        n_envs=1,
        env_kwargs={},
        wrapper_kwargs={
            "frame_skip": 4,
            "screen_size": 84,
            "noop_max": 0,  # Disabled due to nes_py _did_step() compatibility issue
            "use_single_stage_episodes": False,  # Continue to next stage after completion
        },
        vec_normalize_kwargs={
            "training": False,
            "norm_obs": False,
            "norm_reward": True,
            "clip_obs": 10.0,
            "clip_reward": 10.0,
            "gamma": 0.95,
        },
    )
    
    # M4 PRO FIXED CONFIGURATION - Memory Optimized
    total_timesteps = args.timesteps
    learning_rate = 1.4e-5  # Stable for 1x IMPALA CNN
    batch_size = 256        # MPS optimized for M4 Pro
    buffer_size = 50_000    # Memory-friendly (half of cloud version)
    
    print("\n🍎 M4 Pro Optimized Hyperparameters (Fixed Configuration):")
    
    learning_starts = 10_000
    target_update_interval = 10_000
    train_freq = 4
    gradient_steps = 2
    exploration_fraction = 0.15
    exploration_final_eps = 0.01
    gamma = 0.95
    
    print(f"  Platform: MacBook Pro M4 Pro 24GB")
    print(f"  Total timesteps: {total_timesteps:,}")
    print(f"  Learning rate: {learning_rate} (1x IMPALA scale)")
    print(f"  Batch size: {batch_size} (MPS optimized)")
    print(f"  Buffer size: {buffer_size:,} (memory-friendly)")
    print(f"  Learning starts: {learning_starts:,}")
    print(f"  Target update: {target_update_interval:,}")
    print(f"  Train frequency: {train_freq}")
    print(f"  Gradient steps: {gradient_steps}")
    print(f"  Exploration: {exploration_fraction:.2%} decay period")
    print(f"  Final epsilon: {exploration_final_eps}")
    print(f"  Gamma: {gamma}")
    print(f"  Parallel environments: {n_envs}")
    print(f"  Effective experience: {total_timesteps * n_envs:,} samples")
    print(f"  Expected duration: ~{total_timesteps/850000:.1f}h @ 850K steps/hour (M4 Pro MPS)")
    
    # Model configuration
    policy_kwargs = {
        "features_extractor_class": ImpalaCNN,
        "features_extractor_kwargs": {"features_dim": 256},
        "normalize_images": True,
    }
    
    # Initialize or load DQN model
    if resume_from_checkpoint and checkpoint_path:
        print("\n" + "=" * 60)
        print("RESUMING FROM CHECKPOINT")
        print("=" * 60)
        
        # Create temporary model for loading
        temp_model = DQN(
            policy="CnnPolicy",
            env=train_env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=1.0,
            gamma=gamma,
            target_update_interval=target_update_interval,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=1.0,
            exploration_final_eps=exploration_final_eps,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            verbose=1,
            device=device,
        )
        
        # Load from checkpoint
        model, replay_buffer_path, vecnormalize_path = load_checkpoint_with_env(
            checkpoint_path, train_env, device
        )
        
        # Load replay buffer if available
        if replay_buffer_path and os.path.exists(replay_buffer_path):
            try:
                print(f"Loading replay buffer from: {replay_buffer_path}")
                model.load_replay_buffer(replay_buffer_path)
                print("Replay buffer loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load replay buffer: {e}")
        
        # Load VecNormalize if available
        if vecnormalize_path and os.path.exists(vecnormalize_path):
            try:
                print(f"Loading VecNormalize from: {vecnormalize_path}")
                train_env = VecNormalize.load(vecnormalize_path, train_env)
                model.env = train_env
                print("VecNormalize loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load VecNormalize: {e}")
        
        print(f"Successfully resumed from checkpoint!")
        print(f"Current training step: {model.num_timesteps}")
        print("=" * 60)
        
    else:
        print("\n" + "=" * 60)
        print("STARTING NEW TRAINING")
        print("=" * 60)
        
        # Initialize DQN model with aggressive hyperparameters
        model = DQN(
            policy="CnnPolicy",
            env=train_env,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=1.0,
            gamma=gamma,
            target_update_interval=target_update_interval,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=1.0,
            exploration_final_eps=exploration_final_eps,
            policy_kwargs=policy_kwargs,
            tensorboard_log=log_dir,
            verbose=1,
            device=device,
        )
    
    # Configure training callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=250_000,
        save_path=model_dir,
        name_prefix="mario_DQN_autodl_rtx5090",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    
    q_value_monitor = QValueMonitorCallback(log_freq=5_000)
    
    completion_eval_callback = CompletionAwareEvalCallback(
        eval_env=eval_env,
        eval_freq=50_000,
        n_eval_episodes=8,
        deterministic=True,
        verbose=1
    )
    
    # Execute training
    if resume_from_checkpoint:
        remaining_timesteps = total_timesteps - model.num_timesteps
        print(f"\nResuming M4 Pro Local Training...")
        print(f"Already completed: {model.num_timesteps:,} steps")
        print(f"Remaining training: {remaining_timesteps:,} steps")
        if remaining_timesteps <= 0:
            print("Training already completed! Running evaluation only.")
            remaining_timesteps = 0
    else:
        remaining_timesteps = total_timesteps
        print(f"\nInitiating M4 Pro Local Training...")
        print(f"Total training steps: {total_timesteps:,}")
    
    print(f"Expected duration: 6-7 hours")
    print(f"Target completion rate: 30-60%")
    print("Multi-level generalization across 16 Mario stages")
    print("Monitor progress at: http://localhost:6006")
    print("=" * 80)
    
    start_time = time.time()
    
    try:
        if remaining_timesteps > 0:
            model.learn(
                total_timesteps=total_timesteps,
                callback=[checkpoint_callback, q_value_monitor, completion_eval_callback],
                log_interval=100,
                tb_log_name="mario_DQN_autodl_rtx5090",
                reset_num_timesteps=False,  # Important: don't reset when resuming
                progress_bar=True,
            )
        
        training_time = time.time() - start_time
        if remaining_timesteps > 0:
            print(f"Training completed successfully in {training_time/3600:.2f} hours")
        else:
            print("No additional training needed - model already at target timesteps")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
        training_time = time.time() - start_time
        print(f"Training time before interruption: {training_time/3600:.2f} hours")
    
    # Save final model
    final_model_path = os.path.join(model_dir, f"dqn_mario_autodl_{total_timesteps}_steps.zip")
    model.save(final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    # Save Q-value monitoring data
    q_value_path = os.path.join(exp_dir, "q_value_monitor.csv")
    q_value_monitor.save_q_values(q_value_path)
    print(f"Q-value data saved: {q_value_path}")
    
    # Perform comprehensive final evaluation
    print("\nConducting final evaluation...")
    eval_results = comprehensive_evaluation(
        model, eval_env, n_eval_episodes=20, deterministic=True
    )
    
    # Save evaluation results
    eval_results_path = os.path.join(exp_dir, "final_evaluation.json")
    # Convert numpy types to Python native types for JSON serialization
    eval_results_json = {
        k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
        for k, v in eval_results.items()
    }
    with open(eval_results_path, 'w') as f:
        json.dump(eval_results_json, f, indent=2)
    
    print(f"Evaluation results saved: {eval_results_path}")
    print("=" * 80)
    print("M4 Pro Local Training Pipeline Completed Successfully")
    print("=" * 80)


if __name__ == "__main__":
    main()

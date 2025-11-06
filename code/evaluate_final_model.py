#!/usr/bin/env python3
"""
Final Model Evaluation with Video Recording
Evaluates the trained DQN model and generates gameplay videos
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecVideoRecorder, DummyVecEnv
from mario import make_mario_env

def evaluate_model_with_videos(
    model_path,
    n_eval_episodes=2,  # Changed to 2 (skip episode 1, analyze episode 2)
    video_folder="evaluation_videos",
    video_length=2000,
    deterministic=True
):
    """
    Evaluate trained model and record videos
    
    Args:
        model_path: Path to the saved model (.zip file)
        n_eval_episodes: Number of episodes to evaluate (2 to skip first)
        video_folder: Folder to save videos
        video_length: Maximum steps per video
        deterministic: Use deterministic policy
    """
    
    print("="*80)
    print("FINAL MODEL EVALUATION WITH VIDEO RECORDING")
    print("="*80)
    print()
    
    # Check if model exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return None
    
    print(f"Loading model: {model_path}")
    print(f"Model size: {os.path.getsize(model_path) / 1024 / 1024:.1f} MB")
    print()
    
    # Load the trained model
    try:
        model = DQN.load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    print()
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = make_mario_env(
        env_id="SuperMarioBros-1-1-v0",
        n_envs=1,
        seed=42,
        frame_stack=4,
        wrapper_kwargs={
            "noop_max": 0,  # No random noops for evaluation
            "use_single_stage_episodes": True  # Single stage episodes
        },
        env_kwargs={
            "render_mode": "rgb_array"  # Enable video recording
        }
    )
    
    # Wrap with video recorder
    os.makedirs(video_folder, exist_ok=True)
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder,
        record_video_trigger=lambda x: True,  # Record all episodes
        video_length=video_length,
        name_prefix="mario_dqn_eval"
    )
    
    print(f"✅ Environment created")
    print(f"📹 Videos will be saved to: {video_folder}")
    print()
    
    # Evaluation
    print("="*80)
    print(f"EVALUATING: {n_eval_episodes} episodes")
    print("="*80)
    print()
    
    episode_rewards = []
    episode_lengths = []
    completions = []
    
    obs = eval_env.reset()
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    
    while episode_count < n_eval_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = eval_env.step(action)
        
        episode_reward += reward[0]
        episode_length += 1
        
        if done[0]:
            # Extract info from vectorized environment
            if isinstance(info, list) and len(info) > 0:
                ep_info = info[0]
            else:
                ep_info = info
            
            # Check completion
            flag_get = ep_info.get('flag_get', False)
            if hasattr(flag_get, '__iter__'):
                flag_get = flag_get[0] if len(flag_get) > 0 else False
            
            completions.append(flag_get)
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            episode_count += 1
            print(f"Episode {episode_count}/{n_eval_episodes}: "
                  f"Reward={episode_reward:.0f}, "
                  f"Length={episode_length}, "
                  f"Completed={'✅' if flag_get else '❌'}")
            
            episode_reward = 0
            episode_length = 0
            obs = eval_env.reset()
    
    eval_env.close()
    
    # Skip first episode and analyze episodes 2-11
    print()
    print("="*80)
    print("NOTE: Skipping Episode 1 (environment initialization artifact)")
    print("Analyzing Episodes 2-11 (stable state)")
    print("="*80)
    print()
    
    # Use episode 2 for statistics (skip episode 1)
    analysis_rewards = episode_rewards[1:]
    analysis_lengths = episode_lengths[1:]
    analysis_completions = completions[1:]
    
    # Calculate statistics on episode 2
    mean_reward = np.mean(analysis_rewards)
    std_reward = np.std(analysis_rewards)
    mean_length = np.mean(analysis_lengths)
    completion_rate = np.mean(analysis_completions) * 100
    
    print()
    print("="*80)
    print("EVALUATION RESULTS (Episode 2 only)")
    print("="*80)
    print()
    print(f"Total Episodes Run: {n_eval_episodes}")
    print(f"Episodes Analyzed: {len(analysis_rewards)} (excluding Episode 1)")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Reward Range: {min(analysis_rewards):.0f} to {max(analysis_rewards):.0f}")
    print(f"Mean Episode Length: {mean_length:.0f} steps")
    print(f"Level Completion Rate: {completion_rate:.1f}% ({sum(analysis_completions)}/{len(analysis_completions)})")
    print()
    print("Skipped Episode 1 Details:")
    print(f"  Reward: {episode_rewards[0]:.2f}")
    print(f"  Length: {episode_lengths[0]} steps")
    print(f"  Completed: {completions[0]}")
    print()
    
    # Save results with both full and analyzed data
    results = {
        "model_path": str(model_path),
        "total_episodes_run": n_eval_episodes,
        "episodes_analyzed": len(analysis_rewards),
        "note": "Episode 1 skipped due to environment initialization artifact",
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "min_reward": float(min(analysis_rewards)),
        "max_reward": float(max(analysis_rewards)),
        "mean_length": float(mean_length),
        "completion_rate": float(completion_rate),
        "completions": int(sum(analysis_completions)),
        "episode_rewards_analyzed": [float(r) for r in analysis_rewards],
        "episode_lengths_analyzed": [int(l) for l in analysis_lengths],
        "episode_completions_analyzed": [bool(c) for c in analysis_completions],
        "episode_1_skipped": {
            "reward": float(episode_rewards[0]),
            "length": int(episode_lengths[0]),
            "completed": bool(completions[0])
        },
        "all_episode_rewards": [float(r) for r in episode_rewards],
        "all_episode_lengths": [int(l) for l in episode_lengths],
        "all_episode_completions": [bool(c) for c in completions],
    }
    
    results_file = os.path.join(video_folder, "evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📊 Results saved to: {results_file}")
    
    # List generated videos
    video_files = sorted([f for f in os.listdir(video_folder) if f.endswith('.mp4')])
    if video_files:
        print()
        print("📹 Generated Videos:")
        for vf in video_files:
            size_mb = os.path.getsize(os.path.join(video_folder, vf)) / 1024 / 1024
            print(f"   {vf} ({size_mb:.1f} MB)")
    
    print()
    print("="*80)
    
    return results


if __name__ == "__main__":
    # Find the latest experiment
    results_dir = Path("results/dqn_autodl")
    
    # Find latest experiment directory
    exp_dirs = sorted([d for d in results_dir.iterdir() if d.is_dir() and d.name.startswith("exp")])
    
    if not exp_dirs:
        print("❌ No experiment directories found")
        sys.exit(1)
    
    latest_exp = exp_dirs[-1]
    print(f"Using latest experiment: {latest_exp}")
    print()
    
    # Find the final model (5M steps)
    models_dir = latest_exp / "models"
    
    # Try multiple naming patterns
    model_files = sorted([f for f in models_dir.glob("*5000000*.zip") if "replay_buffer" not in f.name])
    
    if not model_files:
        print("❌ No 5M step model found, trying latest checkpoint...")
        model_files = sorted([f for f in models_dir.glob("*.zip") if "replay_buffer" not in f.name])
    
    if not model_files:
        print("❌ No model files found")
        print(f"Searched in: {models_dir}")
        print("Available files:")
        for f in models_dir.iterdir():
            print(f"  - {f.name}")
        sys.exit(1)
    
    model_path = model_files[-1]
    
    # Create video folder in experiment directory
    video_folder = latest_exp / "evaluation_videos_final"
    
    # Run evaluation
    results = evaluate_model_with_videos(
        model_path=str(model_path),
        n_eval_episodes=10,
        video_folder=str(video_folder),
        video_length=2000,
        deterministic=True
    )
    
    if results:
        print()
        print("✅ Evaluation completed successfully!")
        print(f"📁 All results saved to: {video_folder}")

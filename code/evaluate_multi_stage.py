#!/usr/bin/env python3
"""
Multi-Stage Continuous Evaluation
Tests agent's ability to progress through multiple stages continuously
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecVideoRecorder
from mario import make_mario_env

def evaluate_continuous_stages(
    model_path,
    n_eval_episodes=2,  # Changed to 2 (skip episode 1, analyze episode 2)
    video_folder="evaluation_videos_continuous",
    max_steps_per_episode=10000,
    deterministic=True
):
    """
    Evaluate model's ability to progress through multiple stages continuously
    
    Args:
        model_path: Path to trained model
        n_eval_episodes: Number of continuous episodes (2 to skip first)
        video_folder: Folder for videos
        max_steps_per_episode: Maximum steps per episode
        deterministic: Use deterministic policy
    """
    
    print("="*80)
    print("MULTI-STAGE CONTINUOUS EVALUATION")
    print("="*80)
    print()
    
    # Load model
    print(f"Loading model: {model_path}")
    try:
        model = DQN.load(model_path)
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None
    
    print()
    
    # Create environment with CONTINUOUS STAGES
    print("Creating continuous multi-stage environment...")
    print("  Mode: Continuous progression (1-1 → 1-2 → 1-3 → ...)")
    print()
    
    eval_env = make_mario_env(
        env_id="SuperMarioBros-1-1-v0",
        n_envs=1,
        seed=42,
        frame_stack=4,
        wrapper_kwargs={
            "noop_max": 0,
            "use_single_stage_episodes": False  # CONTINUOUS STAGES!
        },
        env_kwargs={
            "render_mode": "rgb_array"
        }
    )
    
    # Wrap with video recorder
    os.makedirs(video_folder, exist_ok=True)
    eval_env = VecVideoRecorder(
        eval_env,
        video_folder,
        record_video_trigger=lambda x: True,
        video_length=max_steps_per_episode,
        name_prefix="mario_continuous"
    )
    
    print(f"✅ Environment created")
    print(f"📹 Videos will be saved to: {video_folder}")
    print()
    
    # Evaluation
    print("="*80)
    print(f"EVALUATING: {n_eval_episodes} continuous episodes")
    print(f"Max steps per episode: {max_steps_per_episode}")
    print("="*80)
    print()
    
    episode_data = []
    
    obs = eval_env.reset()
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    stages_completed = []
    current_stage = "1-1"
    
    while episode_count < n_eval_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, done, info = eval_env.step(action)
        
        episode_reward += reward[0]
        episode_length += 1
        
        # Track stage progression
        if isinstance(info, list) and len(info) > 0:
            ep_info = info[0]
        else:
            ep_info = info
        
        # Check for stage completion
        flag_get = ep_info.get('flag_get', False)
        if hasattr(flag_get, '__iter__'):
            flag_get = flag_get[0] if len(flag_get) > 0 else False
        
        if flag_get:
            stages_completed.append(current_stage)
            print(f"  ✅ Stage {current_stage} completed! (Step {episode_length})")
            # Note: Next stage will auto-load in continuous mode
        
        # Check for episode end (death or max steps)
        if done[0] or episode_length >= max_steps_per_episode:
            episode_count += 1
            
            # Extract final info
            final_world = ep_info.get('world', 1)
            final_stage = ep_info.get('stage', 1)
            final_stage_str = f"{final_world}-{final_stage}"
            
            print()
            print(f"Episode {episode_count}/{n_eval_episodes} finished:")
            print(f"  Total reward: {episode_reward:.0f}")
            print(f"  Total steps: {episode_length}")
            print(f"  Stages completed: {len(stages_completed)}")
            print(f"  Final stage reached: {final_stage_str}")
            print(f"  Stages: {' → '.join(stages_completed) if stages_completed else 'None'}")
            print()
            
            episode_data.append({
                "episode": episode_count,
                "reward": float(episode_reward),
                "length": int(episode_length),
                "stages_completed": len(stages_completed),
                "stages_list": stages_completed.copy(),
                "final_stage": final_stage_str
            })
            
            # Reset for next episode
            episode_reward = 0
            episode_length = 0
            stages_completed = []
            current_stage = "1-1"
            obs = eval_env.reset()
    
    eval_env.close()
    
    # Skip first episode and analyze episodes 2-6
    print()
    print("="*80)
    print("NOTE: Skipping Episode 1 (environment initialization artifact)")
    print("Analyzing Episode 2 only (stable state)")
    print("="*80)
    print()
    
    # Use episode 2 for statistics (skip episode 1)
    analysis_data = episode_data[1:]
    
    # Calculate statistics on episode 2
    total_stages = sum(ep['stages_completed'] for ep in analysis_data)
    mean_stages = np.mean([ep['stages_completed'] for ep in analysis_data])
    mean_reward = np.mean([ep['reward'] for ep in analysis_data])
    mean_length = np.mean([ep['length'] for ep in analysis_data])
    
    print()
    print("="*80)
    print("CONTINUOUS EVALUATION RESULTS (Episode 2 only)")
    print("="*80)
    print()
    print(f"Total Episodes Run: {n_eval_episodes}")
    print(f"Episodes Analyzed: {len(analysis_data)} (excluding Episode 1)")
    print(f"Total stages completed: {total_stages}")
    print(f"Average stages per episode: {mean_stages:.1f}")
    print(f"Mean reward: {mean_reward:.0f}")
    print(f"Mean episode length: {mean_length:.0f} steps")
    print()
    print("Skipped Episode 1 Details:")
    print(f"  Reward: {episode_data[0]['reward']:.0f}")
    print(f"  Length: {episode_data[0]['length']} steps")
    print(f"  Stages: {episode_data[0]['stages_completed']}")
    print()
    
    # Stage completion breakdown (from analyzed episode only)
    print("Stage Completion Breakdown (Episode 2 only):")
    all_stages = []
    for ep in analysis_data:
        all_stages.extend(ep['stages_list'])
    
    from collections import Counter
    stage_counts = Counter(all_stages)
    for stage, count in sorted(stage_counts.items()):
        print(f"  {stage}: {count} times")
    print()
    
    # Save results with both full and analyzed data
    results = {
        "model_path": str(model_path),
        "total_episodes_run": n_eval_episodes,
        "episodes_analyzed": len(analysis_data),
        "note": "Episode 1 skipped due to environment initialization artifact",
        "total_stages_completed": int(total_stages),
        "mean_stages_per_episode": float(mean_stages),
        "mean_reward": float(mean_reward),
        "mean_length": float(mean_length),
        "episode_data_analyzed": analysis_data,
        "stage_completion_counts": dict(stage_counts),
        "episode_1_skipped": episode_data[0],
        "all_episode_data": episode_data
    }
    
    results_file = os.path.join(video_folder, "continuous_evaluation_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📊 Results saved to: {results_file}")
    
    # List videos
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
    model_path = "/Users/rickiyang/Documents/UTS/43008-Reinforcement Learning/AT3/gymnasium-mario/results/dqn_autodl/exp17/models/mario_DQN_autodl_rtx5090_5000000_steps.zip"
    
    video_folder = "/Users/rickiyang/Documents/UTS/43008-Reinforcement Learning/AT3/gymnasium-mario/results/dqn_autodl/exp17/evaluation_continuous"
    
    print("Testing continuous stage progression capability...")
    print()
    
    results = evaluate_continuous_stages(
        model_path=model_path,
        n_eval_episodes=5,
        video_folder=video_folder,
        max_steps_per_episode=10000,
        deterministic=True
    )
    
    if results:
        print()
        print("✅ Continuous evaluation completed!")
        print(f"📁 All results saved to: {video_folder}")

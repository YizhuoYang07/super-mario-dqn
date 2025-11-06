"""
Super Mario Bros DQN Training with CNN-based Observations
"""

from __future__ import annotations  # Enable Python 3.8 compatibility for type hints
import os
from typing import Tuple, Dict, Any

import gymnasium as gym
import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from gymnasium import spaces
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from nes_py.wrappers import JoypadSpace
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecFrameStack,
    VecNormalize,
)

# Try to import gymnasium_super_mario_bros first (preferred), fall back to gym_super_mario_bros
try:
    import gymnasium_super_mario_bros as mario_env_module
    from gymnasium_super_mario_bros.actions import SIMPLE_MOVEMENT
    USING_GYMNASIUM_MARIO = True
except ImportError:
    import gym_super_mario_bros as mario_env_module
    from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
    USING_GYMNASIUM_MARIO = False
    print("Warning: Using old gym_super_mario_bros. Consider upgrading to gymnasium-super-mario-bros.")


class ImpalaCNN(BaseFeaturesExtractor):
    """
    Deep residual CNN architecture for visual RL tasks.

    Architecture:
        - 3 ConvSequences with channels [16, 32, 32] (base configuration, scale=1)
        - Each ConvSequence: Conv3x3 → MaxPool2d(3×3, stride=2) → 2 ResidualBlocks
        - ResidualBlock: ReLU → Conv3x3 → ReLU → Conv3x3 → Add skip connection
        - Final: Flatten → ReLU → Linear(3872→256)

    :param observation_space: The observation space of the environment
    :param features_dim: Number of features extracted (output of final linear layer, default 256)
    :param channels: List of output channels for each convolutional sequence (default [16, 32, 32])
    :param normalized_image: Whether to assume that the image is already normalized
        or not (this disables dtype and bounds checks): when True, it only checks that
        the space is a Box and has 3 dimensions.
        Otherwise, it checks that it has expected dtype (uint8) and bounds (values in [0, 255]).
    """

    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int = 256,
        channels: list = None,
        normalized_image: bool = False,
    ) -> None:
        if channels is None:
            channels = [16, 32, 32]

        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]

        layers = []
        in_channels = n_input_channels

        for out_channels in channels:
            layers.append(self._make_conv_sequence(in_channels, out_channels))
            in_channels = out_channels

        self.cnn = nn.Sequential(*layers)

        with th.no_grad():
            # Create sample with correct shape: (batch, channels, height, width)
            # observation_space.shape should be (n_stack, height, width) after VecFrameStack
            if len(observation_space.shape) == 3:
                # Already stacked: (channels, height, width)
                sample_shape = (1,) + observation_space.shape
            else:
                # Single frame: (height, width) - shouldn't happen after VecFrameStack
                sample_shape = (1, n_input_channels) + observation_space.shape
            
            sample_input = th.zeros(sample_shape, dtype=th.float32)
            if not normalized_image:
                sample_input = sample_input + 128.0  # Use mid-range value instead of sampling
            else:
                sample_input = sample_input + 0.5
            
            cnn_output = self.cnn(sample_input)
            n_flatten = cnn_output.reshape(cnn_output.size(0), -1).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Apply orthogonal initialization with sqrt(2) scaling for ReLU networks.
        """
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def _make_conv_sequence(self, in_channels: int, out_channels: int) -> nn.Module:
        """
        Create a convolutional sequence: Conv -> MaxPool -> ResBlock -> ResBlock

        :param in_channels: Number of input channels
        :param out_channels: Number of output channels
        :return: Sequential module containing the conv sequence
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_residual_block(out_channels),
            self._make_residual_block(out_channels),
        )

    def _make_residual_block(self, channels: int) -> nn.Module:
        """
        Create a residual block: ReLU -> Conv -> ReLU -> Conv -> Add

        :param channels: Number of channels (same for input and output)
        :return: ResidualBlock module
        """
        return ResidualBlock(channels)

    def forward(self, observations: th.Tensor) -> th.Tensor:
        if observations.dtype == th.uint8:
            observations = observations.float() / 255.0

        features = self.cnn(observations)
        features = features.reshape(features.size(0), -1)
        features = nn.functional.relu(features)
        return self.linear(features)


class ResidualBlock(nn.Module):
    """
    Residual block for IMPALA CNN.

    Architecture: ReLU -> Conv3x3 -> ReLU -> Conv3x3 -> Add residual connection

    :param channels: Number of input and output channels
    """

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x: th.Tensor) -> th.Tensor:
        inputs = x
        x = nn.functional.relu(x)
        x = self.conv1(x)
        x = nn.functional.relu(x)
        x = self.conv2(x)
        return x + inputs


# ============================================================================
# OLD GYM API WRAPPERS (for use before GymToGymnasiumWrapper conversion)
# ============================================================================

class OldGymMaxAndSkipFrame(gym.Wrapper):
    """
    Frame skipping wrapper for OLD gym API (step returns 4 values).
    Must be applied BEFORE GymToGymnasiumWrapper.
    """
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=env.observation_space.dtype)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        info = {}
        
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        
        # Max pooling over last 2 frames
        max_frame = self._obs_buffer.max(axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        return self.env.reset()


class OldGymGrayScaleObservation(gym.ObservationWrapper):
    """Grayscale observation for OLD gym API."""
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        from gym import spaces as gym_spaces
        self.observation_space = gym_spaces.Box(
            low=0, high=255, 
            shape=env.observation_space.shape[:2],  # Remove color channel
            dtype=np.uint8
        )
    
    def observation(self, obs):
        import cv2
        return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)


class OldGymResizeObservation(gym.ObservationWrapper):
    """Resize observation for OLD gym API."""
    def __init__(self, env, width=84, height=84):
        gym.ObservationWrapper.__init__(self, env)
        from gym import spaces as gym_spaces
        self.width = width
        self.height = height
        self.observation_space = gym_spaces.Box(
            low=0, high=255,
            shape=(height, width),
            dtype=np.uint8
        )
    
    def observation(self, obs):
        import cv2
        return cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)


# ============================================================================
# GYMNASIUM API CONVERSION WRAPPER
# ============================================================================

class GymToGymnasiumWrapper(gym.Env):
    """
    Robust converter from old gym API to new gymnasium API.
    Inherits from gymnasium.Env to be compatible with gymnasium wrappers.
    
    Conversions:
    - reset() returns (obs, info) instead of just obs
    - step() returns (obs, reward, terminated, truncated, info) instead of (obs, reward, done, info)
    - Filters gymnasium-specific parameters (seed, options) before passing to gym environment
    - Patches nes_py _did_step() method to accept new gymnasium signature
    """
    
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.metadata = getattr(env, 'metadata', {})
        self.render_mode = getattr(env, 'render_mode', None)
        self.spec = getattr(env, 'spec', None)
        
        # CRITICAL FIX: Patch nes_py _did_step() to handle gymnasium's (terminated, truncated) signature
        # nes_py's _did_step(done) expects 1 arg, but nes_env.py calls it with 2 args in new versions
        unwrapped = self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env
        if hasattr(unwrapped, '_did_step'):
            original_did_step = unwrapped._did_step
            
            # Create wrapper that accepts both old (done) and new (terminated, truncated) signatures
            def patched_did_step(done_or_terminated, truncated=None):
                # If called with 2 args (terminated, truncated), combine them
                if truncated is not None:
                    done = done_or_terminated or truncated
                else:
                    done = done_or_terminated
                return original_did_step(done)
            
            unwrapped._did_step = patched_did_step
    
    def reset(self, **kwargs):
        # Remove all gymnasium-specific parameters that old gym doesn't support
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        
        # Call underlying gym environment reset
        result = self.env.reset(**kwargs)
        
        # Convert return value to gymnasium format
        if isinstance(result, tuple):
            return result  # Already (obs, info) format
        else:
            return result, {}  # Convert obs to (obs, info)
    
    def step(self, action):
        result = self.env.step(action)
        
        # Convert return value to gymnasium format
        if len(result) == 5:
            return result  # Already (obs, reward, terminated, truncated, info)
        else:
            # Old gym format: (obs, reward, done, info)
            obs, reward, done, info = result
            return obs, reward, done, False, info  # (obs, reward, terminated, truncated, info)
    
    def render(self):
        if hasattr(self.env, 'render'):
            return self.env.render()
    
    def close(self):
        if hasattr(self.env, 'close'):
            return self.env.close()
    
    def seed(self, seed=None):
        if hasattr(self.env, 'seed'):
            return self.env.seed(seed)
    
    @property
    def unwrapped(self):
        return self.env.unwrapped if hasattr(self.env, 'unwrapped') else self.env


class NoopResetEnv(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Sample initial states by taking random number of no-ops on reset.

    This increases the diversity of initial states and prevents the agent from
    overfitting to a fixed starting configuration. Particularly useful for Mario
    levels where enemies move even when Mario doesn't.

    :param env: Environment to wrap
    :param noop_max: Maximum value of no-ops to run (default 30)
    """

    def __init__(self, env: gym.Env, noop_max: int = 30) -> None:
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        # Filter out gymnasium-specific parameters (seed, options) 
        # because underlying environment is old gym API wrapped by GymToGymnasiumWrapper
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        obs, info = self.env.reset(**kwargs)
        
        # Determine number of no-ops
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            # NumPy 1.26+ uses integers() instead of randint() for Generator
            rng = self.unwrapped.np_random
            if hasattr(rng, 'integers'):
                noops = rng.integers(1, self.noop_max + 1)
            else:
                noops = rng.randint(1, self.noop_max + 1)
        
        assert noops > 0
        
        # Execute no-ops
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            
            # If episode ends during no-ops, reset WITHOUT kwargs to avoid re-passing seed
            if terminated or truncated:
                obs, info = self.env.reset()
        
        return obs, info


class MaxAndSkipFrame(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Return only every ``skip``-th frame (frameskipping)
    and return the max between the two last frames.

    :param env: Environment to wrap
    :param skip: Number of ``skip``-th frame
        The same action will be taken ``skip`` times.
    """

    def __init__(self, env: gym.Env, skip: int = 4) -> None:
        super().__init__(env)
        assert env.observation_space.dtype is not None
        assert env.observation_space.shape is not None
        self._obs_buffer = np.zeros(
            (2, *env.observation_space.shape), dtype=env.observation_space.dtype
        )
        self._skip = skip

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        # Filter gymnasium-specific parameters for old gym compatibility
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        return self.env.reset(**kwargs)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        total_reward = 0.0
        terminated = truncated = False
        flag_get_detected = False

        for i in range(self._skip):
            obs, reward, terminated, truncated, info = self.env.step(action)

            if info.get("flag_get", False):
                flag_get_detected = True

            done = terminated or truncated
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += float(reward)

            if done or flag_get_detected:
                break

        max_frame = self._obs_buffer.max(axis=0)

        if flag_get_detected:
            info["flag_get"] = True

        return max_frame, total_reward, terminated, truncated, info


class GrayScaleObservation(gym.ObservationWrapper):
    """Convert observations to grayscale."""
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        obs_shape = self.observation_space.shape[:2]
        self.observation_space = spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
    
    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        # Filter gymnasium-specific parameters for old gym compatibility
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        obs, info = self.env.reset(**kwargs)
        # Apply observation transformation
        return self.observation(obs), info
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        import cv2
        return cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)


class ResizeObservation(gym.ObservationWrapper):
    """Resize observations to specified dimensions and add channel dimension for VecFrameStack."""
    
    def __init__(self, env: gym.Env, width: int = 84, height: int = 84):
        super().__init__(env)
        self.width = width
        self.height = height
        # Add channel dimension (1,) for grayscale - required for VecFrameStack
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(height, width, 1), dtype=np.uint8
        )
    
    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        # Filter gymnasium-specific parameters for old gym compatibility
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        obs, info = self.env.reset(**kwargs)
        # Apply observation transformation
        return self.observation(obs), info
    
    def observation(self, obs: np.ndarray) -> np.ndarray:
        import cv2
        resized = cv2.resize(obs, (self.width, self.height), interpolation=cv2.INTER_AREA)
        # Add channel dimension: (H, W) -> (H, W, 1)
        return resized[:, :, np.newaxis]


class MarioWrapper(gym.Wrapper[np.ndarray, int, np.ndarray, int]):
    """
    Super Mario Bros preprocessing wrapper.

    Applies the following preprocessing steps:

    * Action space reduction: SIMPLE_MOVEMENT (7 actions) by default
    * No-op reset: Random 1-30 no-ops on reset for initial state diversity
    * Frame skipping with max-pooling: 4 by default (handles NES sprite flickering)
    * Resize to square grayscale image: 84x84 by default
    * Optional: Single-stage episode mode


    :param env: Environment to wrap (Super Mario Bros environment)
    :param frame_skip: Frequency at which the agent experiences the game.
        This correspond to repeating the action ``frame_skip`` times.
    :param screen_size: Resize frame to a square image of this size
    :param action_space: Action space to use (default: SIMPLE_MOVEMENT)
    :param noop_max: Maximum number of no-ops on reset (default 30, set to 0 to disable)
    :param use_single_stage_episodes: If True, each episode terminates after completing one stage.
        When False (default), episodes continue across multiple stages
        (e.g., 1-1 → 1-2 → 1-3...) until death or game over.
    """

    def __init__(
        self,
        env: gym.Env,
        frame_skip: int = 4,
        screen_size: int = 84,
        action_space=SIMPLE_MOVEMENT,
        noop_max: int = 30,
        use_single_stage_episodes: bool = False,
    ) -> None:
        # Handle both gymnasium_super_mario_bros (new) and gym_super_mario_bros (old)
        
        # Step 1: JoypadSpace (works with raw env, both old gym and new gymnasium)
        env = JoypadSpace(env, action_space)
        
        # Step 1.5: If using old gym version, wrap with GymToGymnasiumWrapper AFTER JoypadSpace
        if not USING_GYMNASIUM_MARIO:
            env = GymToGymnasiumWrapper(env)
        
        # Step 2: No-op reset for initial state diversity
        if noop_max > 0:
            env = NoopResetEnv(env, noop_max=noop_max)
        
        # Step 3: Frame skipping with max-pooling
        if frame_skip > 1:
            env = MaxAndSkipFrame(env, skip=frame_skip)
        
        # Step 4: Grayscale and resize
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, width=screen_size, height=screen_size)

        super().__init__(env)
        self._use_single_stage_episodes = use_single_stage_episodes
        
        # Convert gym action/observation spaces to gymnasium for SB3 compatibility
        # Action space: gym.spaces.Discrete -> gymnasium.spaces.Discrete
        if hasattr(self.env.action_space, 'n'):
            self.action_space = spaces.Discrete(self.env.action_space.n)
        
        # Observation space: already set by ResizeObservation with gymnasium.spaces.Box
        # but ensure it's using gymnasium.spaces if needed
        if hasattr(self.env.observation_space, 'shape'):
            self.observation_space = spaces.Box(
                low=0, 
                high=255, 
                shape=self.env.observation_space.shape, 
                dtype=np.uint8
            )

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        # Filter gymnasium-specific parameters for old gym compatibility
        kwargs.pop('seed', None)
        kwargs.pop('options', None)
        return self.env.reset(**kwargs)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        if self._use_single_stage_episodes and info.get("flag_get", False):
            truncated = True

        return obs, reward, terminated, truncated, info


def make_mario_env(
    env_id="SuperMarioBros-1-1-v0",
    n_envs=1,
    seed=None,
    max_episode_steps=None,
    frame_stack=4,
    wrapper_kwargs=None,
    vec_normalize_kwargs=None,
    env_kwargs=None,
    monitor_dir=None,
):
    """
    Create a wrapped, monitored VecEnv for Super Mario Bros with Atari-style preprocessing.
    Always uses DummyVecEnv (single-process) for consistent behavior.

    Parameters:
        env_id: The environment ID
            - Single stage: "SuperMarioBros-1-1-v0", "SuperMarioBros-1-2-v0", etc.
        n_envs: Number of environments to create
        seed: Random seed
        max_episode_steps: Maximum episode length in steps (default None, uses environment default of 9999999)
        frame_stack: Number of frames to stack (default 4)
        wrapper_kwargs: Dict of kwargs for MarioWrapper. Defaults:
            - frame_skip: 4
            - screen_size: 84
            - noop_max: 30
            - use_single_stage_episodes: False
        vec_normalize_kwargs: Dict of kwargs for VecNormalize, or None to disable.
            - Pass None: Disable VecNormalize completely (no normalization)
            - Pass dict (can be empty {}): Apply VecNormalize with values merged into defaults
            Default VecNormalize settings:
            - training: True
            - norm_obs: False
            - norm_reward: True
            - clip_obs: 10.0
            - clip_reward: 10.0
            - gamma: 0.982
        env_kwargs: Dict passed to gym.make(), e.g., {"render_mode": "rgb_array"}
        monitor_dir: Directory for Monitor wrapper (optional)

    Returns:
        VecEnv: The wrapped vectorized environment with frame stacking and optionally VecNormalize

    Example:
        env = make_mario_env("SuperMarioBros-1-1-v0", n_envs=8)
    """
    if wrapper_kwargs is None:
        wrapper_kwargs = {}
    wrapper_kwargs.setdefault("frame_skip", 4)
    wrapper_kwargs.setdefault("screen_size", 84)
    wrapper_kwargs.setdefault("noop_max", 30)
    wrapper_kwargs.setdefault("use_single_stage_episodes", False)

    if env_kwargs is None:
        env_kwargs = {}

    if max_episode_steps is not None:
        env_kwargs["max_episode_steps"] = max_episode_steps

    # Create environment factory function
    def make_env(rank):
        def _init():
            # Use mario_env_module (auto-selected gymnasium or gym version)
            if USING_GYMNASIUM_MARIO:
                env = mario_env_module.make(env_id, **env_kwargs)
            else:
                # Old gym version - needs disable_env_checker
                env = mario_env_module.make(env_id, disable_env_checker=True, **env_kwargs)
            
            # Remove TimeLimit wrapper if present
            if hasattr(env, 'env') and env.__class__.__name__ == 'TimeLimit':
                env = env.env  # Unwrap TimeLimit
            
            env = MarioWrapper(env, **wrapper_kwargs)
            
            # Re-apply TimeLimit after MarioWrapper if max_episode_steps was specified
            if max_episode_steps is not None:
                from gymnasium.wrappers import TimeLimit as GymnasiumTimeLimit
                env = GymnasiumTimeLimit(env, max_episode_steps=max_episode_steps)
            
            if monitor_dir is not None:
                from stable_baselines3.common.monitor import Monitor
                env = Monitor(env, monitor_dir)
            # Seed the environment's RNG without calling reset
            if seed is not None:
                env.action_space.seed(seed + rank)
                env.observation_space.seed(seed + rank)
            return env
        return _init

    # Create vectorized environment
    env_fns = [make_env(i) for i in range(n_envs)]
    vec_env = DummyVecEnv(env_fns)

    # Stack frames: (84, 84, 1) -> (84, 84, 4) channels-last format
    vec_env = VecFrameStack(vec_env, n_stack=frame_stack)

    # CRITICAL: Transpose to channels-first for PyTorch CNN: (N, H, W, C) -> (N, C, H, W)
    from stable_baselines3.common.vec_env import VecTransposeImage
    vec_env = VecTransposeImage(vec_env)
    # Shape after transpose: (N, 4, 84, 84) ✓

    # Apply VecNormalize if dict provided (None = disabled)
    if vec_normalize_kwargs is not None:
        default_vec_normalize_kwargs = {
            "training": True,
            "norm_obs": False,
            "norm_reward": True,
            "clip_obs": 10.0,
            "clip_reward": 10.0,
            "gamma": 0.982,
        }
        default_vec_normalize_kwargs.update(vec_normalize_kwargs)
        vec_env = VecNormalize(vec_env, **default_vec_normalize_kwargs)

    return vec_env


def create_experiment_folder(base_dir="results/ppo"):
    """Create experiment folder structure."""
    os.makedirs(base_dir, exist_ok=True)

    exp_num = 1
    while os.path.exists(os.path.join(base_dir, f"exp{exp_num}")):
        exp_num += 1

    exp_dir = os.path.join(base_dir, f"exp{exp_num}")
    model_dir = os.path.join(exp_dir, "models")
    log_dir = os.path.join(exp_dir, "logs")
    video_dir = os.path.join(exp_dir, "videos")

    os.makedirs(model_dir)
    os.makedirs(log_dir)
    os.makedirs(video_dir)

    print(f"Created Experiment {exp_num}: {exp_dir}")

    return exp_num, exp_dir, model_dir, log_dir, video_dir


def evaluate_policy(
    model,
    test_env,
    video_dir=None,
    video_fps=60,
    n_episodes=1,
    deterministic=True,
    record_video=True,
    save_results=True,
    results_path=None,
    model_dir=None,
):
    """
    Evaluate policy, optionally record video and save results.
    Automatically loads best_model if available.

    :param model: Trained model
    :param test_env: Test environment (created externally with render_mode='rgb_array' if recording video)
    :param video_dir: Directory to save video (will be named mario_gameplay.mp4)
    :param video_fps: Video framerate (default 60)
    :param n_episodes: Number of episodes to evaluate (default 1)
    :param deterministic: Use deterministic policy (default True)
    :param record_video: Whether to record video (default True)
    :param save_results: Whether to save results to CSV (default True)
    :param results_path: Path to save results CSV (required if save_results=True)
    :param model_dir: Directory containing best_model subdirectory (optional)
    :return: Dictionary with statistics
    """
    if model_dir is not None:
        best_model_path = os.path.join(model_dir, "best_model", "best_model.zip")
        if os.path.exists(best_model_path):
            model = PPO.load(best_model_path)

    if record_video:
        if video_dir is None:
            raise ValueError("video_dir must be provided when record_video=True")
        video_path = os.path.join(video_dir, "mario_gameplay.mp4")
    else:
        video_path = None

    captured_frames = []
    step_called_since_reset = [False]

    if record_video:

        def capture_all_frames_in_skip():
            def wrapped_step(self, action):
                step_called_since_reset[0] = True
                total_reward = 0.0
                terminated = truncated = False
                flag_get_detected = False

                for i in range(self._skip):
                    obs, reward, terminated, truncated, info = self.env.step(action)

                    if hasattr(self.env, "render"):
                        frame = self.env.render()
                        if frame is not None:
                            captured_frames.append(frame.copy())

                    if info.get("flag_get", False):
                        flag_get_detected = True

                    done = terminated or truncated
                    if i == self._skip - 2:
                        self._obs_buffer[0] = obs
                    if i == self._skip - 1:
                        self._obs_buffer[1] = obs
                    total_reward += float(reward)

                    if done or flag_get_detected:
                        break

                max_frame = self._obs_buffer.max(axis=0)
                if flag_get_detected:
                    info["flag_get"] = True

                return max_frame, total_reward, terminated, truncated, info

            return wrapped_step

        def capture_noop_reset_frames():
            def wrapped_reset(self, **kwargs):
                is_auto_reset = step_called_since_reset[0]
                should_record_noop = not is_auto_reset

                obs, info = self.env.reset(**kwargs)
                if self.override_num_noops is not None:
                    noops = self.override_num_noops
                else:
                    noops = self.unwrapped.np_random.integers(1, self.noop_max + 1)
                assert noops > 0

                internal_reset_occurred = False

                for _ in range(noops):
                    obs, _, terminated, truncated, info = self.env.step(self.noop_action)

                    if should_record_noop and not internal_reset_occurred:
                        if hasattr(self.env, "render"):
                            frame = self.env.render()
                            if frame is not None:
                                captured_frames.append(frame.copy())

                    if terminated or truncated:
                        obs, info = self.env.reset(**kwargs)
                        internal_reset_occurred = True

                step_called_since_reset[0] = False

                return obs, info

            return wrapped_reset

        current_env = test_env.envs[0]
        noop_env = None
        while hasattr(current_env, "env"):
            if current_env.__class__.__name__ == "MaxAndSkipFrame":
                current_env.step = lambda action, env=current_env: capture_all_frames_in_skip()(
                    env, action
                )
            elif current_env.__class__.__name__ == "NoopResetEnv":
                noop_env = current_env
            current_env = current_env.env

        if noop_env is not None:
            noop_env.reset = lambda **kwargs: capture_noop_reset_frames()(noop_env, **kwargs)

    episode_rewards = []
    episode_lengths = []

    for _ in range(n_episodes):
        obs = test_env.reset()
        episode_reward = 0
        episode_step = 0

        while True:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done, _ = test_env.step(action)
            episode_reward += reward[0]
            episode_step += 1

            if done[0]:
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_step)
                break

    test_env.close()

    if record_video and len(captured_frames) > 0:
        clip = ImageSequenceClip(captured_frames, fps=video_fps)
        clip.write_videofile(video_path)

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)

    if save_results:
        if results_path is None:
            raise ValueError("results_path must be provided when save_results=True")

        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(results_path), exist_ok=True)

        results_df = pd.DataFrame(
            {
                "Episode": range(1, len(episode_rewards) + 1),
                "Reward": episode_rewards,
                "Length": episode_lengths,
            }
        )

        summary_df = pd.DataFrame(
            {
                "Metric": ["Mean Reward", "Std Reward", "Mean Length"],
                "Value": [f"{mean_reward:.2f}", f"{std_reward:.2f}", f"{mean_length:.2f}"],
            }
        )

        with open(results_path, "w") as f:
            results_df.to_csv(f, index=False)
            f.write("\n")
            summary_df.to_csv(f, index=False)

    return {
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "mean_reward": mean_reward,
        "std_reward": std_reward,
        "mean_length": mean_length,
        "min_reward": np.min(episode_rewards),
        "max_reward": np.max(episode_rewards),
    }


if __name__ == "__main__":
    exp_num, exp_dir, model_dir, log_dir, video_dir = create_experiment_folder()

    train_env = make_mario_env(
        "SuperMarioBrosRandomStages-v0",
        n_envs=4,
        wrapper_kwargs={
            "frame_skip": 4,
            "screen_size": 84,
            "use_single_stage_episodes": True,
            "noop_max": 80,
        },
        vec_normalize_kwargs={
            "training": True,
            "norm_obs": False,
            "norm_reward": True,
            "clip_obs": 10.0,
            "clip_reward": 10.0,
            "gamma": 0.982,
        },
        env_kwargs={"stages": ["1-1", "1-2", "1-3", "1-4"]},
        monitor_dir=f"{log_dir}/train",
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=50000,
        save_path=f"{model_dir}/checkpoints",
        name_prefix="mario_PPO",
        verbose=1,
    )

    callbacks = [checkpoint_callback]

    model = PPO(
        "CnnPolicy",
        train_env,
        n_steps=4096,
        batch_size=64,
        n_epochs=10,
        learning_rate=1.4e-5,
        gamma=0.982,
        gae_lambda=0.901,
        ent_coef=1.81e-3,
        clip_range=0.335,
        vf_coef=0.643,
        max_grad_norm=0.578,
        policy_kwargs=dict(
            features_extractor_class=ImpalaCNN,
            features_extractor_kwargs=dict(
                features_dim=256,
                channels=[16, 32, 32],
                normalized_image=False,
            ),
            net_arch=dict(pi=[256], vf=[256]),
        ),
        verbose=0,
        tensorboard_log=os.path.join(log_dir, "tensorboard"),
    )

    model.learn(
        total_timesteps=5e6,
        callback=callbacks,
        tb_log_name="mario_PPO",
        progress_bar=True,
    )

    test_env = make_mario_env(
        "SuperMarioBrosRandomStages-v0",
        wrapper_kwargs={
            "frame_skip": 4,
            "screen_size": 84,
            "use_single_stage_episodes": True,
            "noop_max": 80,
        },
        vec_normalize_kwargs={
            "training": False,
            "norm_reward": False,
        },
        env_kwargs={"stages": ["1-1", "1-2", "1-3", "1-4"], "render_mode": "rgb_array"},
    )

    evaluate_policy(
        model,
        test_env,
        video_dir=video_dir,
        n_episodes=5,
        results_path=os.path.join(log_dir, "test", "test_results.csv"),
        model_dir=model_dir,
    )

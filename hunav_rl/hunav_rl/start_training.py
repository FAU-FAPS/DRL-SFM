"""Training module for reinforcement learning agents in navigation tasks."""


import os
import sys
import rclpy
from rclpy.node import Node
from gymnasium.envs.registration import register, registry
import gymnasium as gym
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    StopTrainingOnNoModelImprovement,
)
from hunav_rl.callbacks import StartStopCallback
import json
import pickle
from ament_index_python.packages import get_package_share_directory

class ReplayBufferSaveCallback(BaseCallback):
    """Custom callback to save replay buffer after evaluation and for best
    models."""
    
    def __init__(self, save_path: str, verbose: int = 0):
        """Initialize the replay buffer save callback.
        
        Args:
            save_path: Directory path where replay buffers will be saved.
            verbose: Verbosity level for callback output.
        """
        super(ReplayBufferSaveCallback, self).__init__(verbose)
        self.save_path = save_path
        self.best_mean_reward = -float('inf')
        
    def _on_step(self) -> bool:
        return True
    
    def save_replay_buffer(self, suffix: str = ""):
        """Save the replay buffer to disk."""
        if hasattr(self.model, 'replay_buffer') and \
           self.model.replay_buffer is not None:
            buffer_path = os.path.join(
                self.save_path,
                f"replay_buffer{suffix}.pkl"
            )
            with open(buffer_path, 'wb') as f:
                pickle.dump(self.model.replay_buffer, f)
            if self.verbose > 0:
                print(f"Replay buffer saved to {buffer_path}")


class CustomEvalCallback(EvalCallback):
    """
    Custom evaluation callback that saves replay buffer after each evaluation.
    """
    def __init__(self, *args, replay_buffer_save_path: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.replay_buffer_save_path = replay_buffer_save_path
        if replay_buffer_save_path:
            self.replay_buffer_callback = ReplayBufferSaveCallback(
                replay_buffer_save_path
            )
        else:
            self.replay_buffer_callback = None

    def _on_step(self) -> bool:
        result = super()._on_step()
        
        # Save replay buffer after evaluation
        if (self.replay_buffer_callback and self.eval_freq > 0
            and self.n_calls % self.eval_freq == 0):
            self.replay_buffer_callback.model = self.model
            self.replay_buffer_callback.save_replay_buffer("_latest")
            # Save replay buffer if this evaluation improved the best score
            if hasattr(self, 'best_mean_reward') \
               and hasattr(self, 'last_mean_reward'):
                if self.last_mean_reward > self.best_mean_reward:
                    self.replay_buffer_callback.save_replay_buffer("_best")
                    if self.verbose > 0:
                        print(
                            f"New best! Saved replay buffer with mean reward: "
                            f"{self.last_mean_reward:.2f}"
                        )
        
        return result
    
class TrainingNode(Node):
    """ROS 2 Node for training reinforcement learning models."""
    
    def __init__(self):
        """Initialize the training node."""
        super().__init__("wheelchair_training")
        self.declare_parameter("num_envs", 2)
        self.declare_parameter("first_ros_domain_id", 10)
        self.declare_parameter("save_buffer", False)
        self._training_mode = "training"
        self.declare_parameter("continue_training", False)
        if self.get_parameter("continue_training").value:
            self._training_mode = "continue_training"

def make_env(env_id, first_ros_domain_id, obs_mode="costmap"):
    """
    Creates a Gym environment in a subprocess.
    
    Args:
        env_id: The environment ID.
        first_ros_domain_id: The first ROS domain ID to use.
        obs_mode: The observation mode to use (default: "costmap").
    
    Returns:
        A function that initializes the environment instance.
    """
    def _init():
        os.environ["ROS_DOMAIN_ID"] = str(first_ros_domain_id + env_id)
        import rclpy
        rclpy.init(args=None)
        from gymnasium.envs.registration import register
        try:
            register(
                id="WheelchairEnv-v0",
                entry_point="hunav_rl.wheelchair_env:WheelchairEnv",
                max_episode_steps=10000,
            )
        except Exception as e:
            pass
        env_instance = gym.make(
            'WheelchairEnv-v0',
            env_id=env_id,
            observation_mode=obs_mode
        )
        env_instance = Monitor(env_instance)
        return env_instance
    return _init


def main(args=None):
    """Main entry point for the training script."""
    obs_mode = "humap"  # "humap", "costmap"
    # Initialize ROS in the main process
    rclpy.init(args=args)
    node = TrainingNode()
    node.get_logger().info("Training node has been created")

    # Directories for saving models and logs
    pkg_share_dir = get_package_share_directory('hunav_rl')
    ws_dir = os.path.abspath(
        os.path.join(pkg_share_dir, '..', '..', '..', '..')
    )
    pkg_dir = os.path.join(ws_dir, 'src', 'drl-sfm' ,'hunav_rl')
    trained_models_dir = os.path.join(pkg_dir, 'hunav_rl/rl_models')
    log_dir = os.path.join(pkg_dir, 'hunav_rl/logs')
    replay_buffer_dir = os.path.join(pkg_dir, 'hunav_rl/replay_buffers')
    os.makedirs(trained_models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(replay_buffer_dir, exist_ok=True)

    # read json config file
    config_file = os.path.join(pkg_dir, 'config', 'training_scenario.json')
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        if "scenario" in config:
            scenario = config["scenario"] # loads a list of strings
    else:
        node.get_logger().warning(
            f"Config file {config_file} does not exist. "
            "Using default config."
        )
        scenario = []
    algorithm = "RecurrentPPO"
    if "ppo" in scenario:
        algorithm = "PPO"
    elif "sac" in scenario:
        algorithm = "SAC"
    if "costmap" in scenario:
        obs_mode = "costmap"
    elif "laser" in scenario:
        obs_mode = "laser"
    elif "drlvo_observation" in scenario:
        obs_mode = "drlvo"
    if "vo_policy" in scenario:
        policy_network = "vo_policy"
    else:
        policy_network = "default"
    # Register the environment in the main process
    if "WheelchairEnv-v0" not in registry:
        register(
            id="WheelchairEnv-v0",
            entry_point="hunav_rl.wheelchair_env:WheelchairEnv",
            max_episode_steps=10000,
        )
    node.get_logger().info("The environment has been registered")
    num_envs = node.get_parameter("num_envs").value
    n_steps = 20480 // num_envs
    batch_size_ = 512
    # calculate the nearest batch_size to batch_size_,
    # so that n_steps * num_envs is divisible by batch_size
    n_steps_total = n_steps * num_envs
    divisors = [
        i for i in range(1, n_steps_total + 1)
        if n_steps_total % i == 0
    ]
    batch_size = min(divisors, key=lambda x: abs(x - batch_size_))
    first_ros_domain_id = node.get_parameter("first_ros_domain_id").value
    save_buffer = node.get_parameter("save_buffer").value

    node.get_logger().info(f"Using {num_envs} environments for training")
    node.get_logger().info(
        f"Replay buffer saving: "
        f"{'enabled' if save_buffer else 'disabled'}"
    )

    # Create the SubprocVecEnv using the specified number of environments
    envs = [
        make_env(i, first_ros_domain_id, obs_mode)
        for i in range(num_envs)
    ]
    train_env = SubprocVecEnv(envs)
    stop_callback = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=25,
        min_evals=25,
        verbose=1,
    )
    
    # Only pass replay_buffer_save_path if save_buffer is True
    eval_callback = CustomEvalCallback(
        train_env,
        callback_after_eval=stop_callback,
        eval_freq=n_steps,
        best_model_save_path=trained_models_dir,
        n_eval_episodes=50,
        replay_buffer_save_path=replay_buffer_dir if save_buffer else None
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=n_steps,
        save_path=trained_models_dir,
        name_prefix='test_checkpoint'
    )
    pause_resume_callback = StartStopCallback(
        first_ros_domain_id,
        num_envs,
        update_steps=n_steps_total,
        verbose=1,
    )
    callbacks = [pause_resume_callback, eval_callback, checkpoint_callback]
    if algorithm == "SAC":
        callbacks = [eval_callback, checkpoint_callback]
    if obs_mode == "costmap":
        from hunav_rl.feature_extractor_costmap import (
            CustomCombinedExtractorCostmap,
        )
        policy_kwargs = {
            "features_extractor_class": CustomCombinedExtractorCostmap,
            "features_extractor_kwargs": {"features_dim": 256},
        }
    elif obs_mode == "humap":
        from hunav_rl.feature_extractor_humap import (
            CustomCombinedExtractorHumap
        )
        policy_kwargs = {
            "features_extractor_class": CustomCombinedExtractorHumap,
            "features_extractor_kwargs": {"features_dim": 256},
        }
    elif obs_mode == "laser":
        policy_kwargs = {}
    if node._training_mode == "training":
        # Create a new model using the training environment
        if algorithm == "PPO":
            model = PPO(
                "MultiInputPolicy",
                train_env,
                verbose=1,
                tensorboard_log=log_dir,
                policy_kwargs=policy_kwargs,
                n_steps=n_steps,
                batch_size=batch_size
            )
        elif algorithm == "RecurrentPPO":
            model = RecurrentPPO(
                "MultiInputLstmPolicy",
                train_env,
                verbose=1,
                tensorboard_log=log_dir,
                policy_kwargs=policy_kwargs,
                n_steps=n_steps,
                batch_size=batch_size
            )
        elif algorithm == "SAC":
            model = SAC(
                "MultiInputPolicy",
                train_env,
                verbose=1,
                tensorboard_log=log_dir,
                policy_kwargs=policy_kwargs,
                batch_size=batch_size,
                buffer_size= 100000,
                learning_starts= n_steps_total,
            )

    try:
        model.learn(
            total_timesteps=int(40000000),
            reset_num_timesteps=False,
            callback=callbacks,
            tb_log_name="test"
        )
    except (KeyboardInterrupt, SystemExit):
        # Save the model and replay buffer on interruption
        model.save(os.path.join(trained_models_dir, "test"))
        if (save_buffer
            and hasattr(model, 'replay_buffer')
            and model.replay_buffer is not None):
            buffer_path = os.path.join(
                replay_buffer_dir,
                "replay_buffer_final.pkl"
            )
            with open(buffer_path, 'wb') as f:
                pickle.dump(model.replay_buffer, f)
            node.get_logger().info(
                f"Final replay buffer saved to {buffer_path}"
            )
    # Save the model and replay buffer after training
    model.save(os.path.join(trained_models_dir, "test"))
    if (save_buffer and hasattr(model, 'replay_buffer')
        and model.replay_buffer is not None):
        buffer_path = os.path.join(
            replay_buffer_dir,
            "replay_buffer_final.pkl"
        )
        with open(buffer_path, 'wb') as f:
            pickle.dump(model.replay_buffer, f)
        node.get_logger().info(f"Final replay buffer saved to {buffer_path}")

    node.get_logger().info("The training is finished, stopping ROS node.")
    node.destroy_node()
    rclpy.shutdown()
    os._exit(0)
    return 0

if __name__ == "__main__":
    sys.exit(main())

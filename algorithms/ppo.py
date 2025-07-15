"""
PPO 封装（stable-baselines3）
功能：
  1. 训练时自动写 TensorBoard
  2. 保存 / 加载模型
  3. 评估并录屏
用法：
    agent = PPOAgent(env)
    agent.learn(total_timesteps=100_000)
    agent.evaluate(episodes=5)
"""
import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.evaluation import evaluate_policy
class PPOAgent:
    def __init__(self, env: gym.Env, log_dir: str = "runs/ppo",
                 model_path: str = "models/ppo_lunar"):
        self.env = env
        self.log_dir = log_dir
        self.model_path = model_path
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # 并行环境加速训练（可选）
        def _make_env():
            return gym.make("LunarLander-v3", render_mode="rgb_array")

        self.vec_env = make_vec_env(_make_env, n_envs=4)

        # 若已有模型则加载，否则新建
        if os.path.exists(f"{self.model_path}.zip"):
            self.model = PPO.load(self.model_path, env=self.vec_env)
            print("✅ 已加载已有模型")
        else:
            self.model = PPO(
                "MlpPolicy",
                self.vec_env,
                batch_size=64,
                n_steps=2048,
                n_epochs=4,
                gamma=0.999,
                gae_lambda=0.98,
                verbose=1,
                ent_coef=0.01,
                tensorboard_log=self.log_dir,
                device="auto"  # 自动使用 GPU（如有）
            )

    def learn(self, total_timesteps: int):
        """训练 + 自动保存最优模型"""
        eval_env = make_vec_env(lambda: self.env, n_envs=1)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=self.model_path + "_best",
            log_path=self.log_dir,
            eval_freq=max(total_timesteps // 20, 1),
            deterministic=True,
            render=False
        )
        self.model.learn(total_timesteps=total_timesteps,
                         callback=eval_callback)
        self.model.save(self.model_path)

        mean_reward, std_reward = evaluate_policy(self.model, eval_env, n_eval_episodes=10, deterministic=True)
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")

        print(f"💾 模型已保存到 {self.model_path}.zip")

        

    def evaluate(self, episodes: int = 10):
        """评估 + 录屏"""
        
        eval_env = make_vec_env(lambda: self.env, n_envs=1)
        # 录屏
        eval_env = VecVideoRecorder(
            eval_env,
            video_folder="videos/ppo_eval",
            record_video_trigger=lambda x: x == 0,
            video_length=1000,
            name_prefix="ppo"
        )
        
        obs = eval_env.reset()
        for _ in range(episodes):
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _ = eval_env.step(action)
        eval_env.close()
        print("🎬 评估视频已保存到 videos/ppo_eval/")
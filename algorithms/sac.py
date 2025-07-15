"""
SAC 封装（stable-baselines3）
功能：
  1. 训练时自动写 TensorBoard
  2. 保存 / 加载模型
  3. 评估并录屏
用法：
    agent = SACAgent(env)
    agent.learn(total_timesteps=300_000)
    agent.evaluate(episodes=5)
"""
import os
import gymnasium as gym
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder
from stable_baselines3.common.evaluation import evaluate_policy


class SACAgent:
    def __init__(self, env: gym.Env,
                 log_dir: str = "runs/sac",
                 model_path: str = "models/sac_lunar"):
        self.env = env
        self.log_dir = log_dir
        self.model_path = model_path
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)

        # SAC 只支持单环境（或 DummyVecEnv 1 个）
        def _make_env():
            return gym.make("LunarLanderContinuous-v2", render_mode="rgb_array")

        self.vec_env = make_vec_env(_make_env, n_envs=1)

        # 若已有模型则加载，否则新建
        if os.path.exists(f"{self.model_path}.zip"):
            self.model = SAC.load(self.model_path, env=self.vec_env)
            print("✅ 已加载已有 SAC 模型")
        else:
            self.model = SAC(
                "MlpPolicy",
                self.vec_env,
                learning_rate=3e-4,
                buffer_size=300_000,
                learning_starts=10_000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=1,
                gradient_steps=1,
                ent_coef="auto",
                verbose=1,
                tensorboard_log=self.log_dir,
                device="auto"
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

        # 训练结束再整体测一次
        mean_reward, std_reward = evaluate_policy(
            self.model, eval_env, n_eval_episodes=10, deterministic=True
        )
        print(f"mean_reward={mean_reward:.2f} +/- {std_reward}")
        print(f"💾 模型已保存到 {self.model_path}.zip")

    def evaluate(self, episodes: int = 10):
        """评估 + 录屏"""
        eval_env = make_vec_env(lambda: self.env, n_envs=1)
        eval_env = VecVideoRecorder(
            eval_env,
            video_folder="videos/sac_eval",
            record_video_trigger=lambda x: x == 0,
            video_length=1000,
            name_prefix="sac"
        )

        obs = eval_env.reset()
        for _ in range(episodes):
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, done, _ = eval_env.step(action)
        eval_env.close()
        print("🎬 评估视频已保存到 videos/sac_eval/")
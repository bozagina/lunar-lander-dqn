from stable_baselines3 import DQN
import gymnasium as gym
from utils.recorder import VideoRecorder, MultiAlgoRecorder
import os

class DQNAgentSB3:
    def __init__(self, env: gym.Env, **sb3_kwargs):
        # SB3 需要 observation_space / action_space
        self.env = env
        self.recorder = VideoRecorder(env, fps=30)  # <-- 新增
        self.model = DQN(
            policy="MlpPolicy",
            env=env,
            learning_rate=1e-3,
            buffer_size=50_000,
            learning_starts=1_000,
            batch_size=32,
            gamma=0.99,
            target_update_interval=250,
            verbose=1,
            **sb3_kwargs
        )
        self.model_path = "models/dqn_sb3_model"  # 保存模型的路径
        os.makedirs("models", exist_ok=True)  # 确保保存目录存在

    # --------------------------------------------------
    # 1. 训练 + 录制（可选）
    # --------------------------------------------------
    def learn(self, total_timesteps: int, record: bool = False, record_episodes: int = 3):
        """
        record: 是否开启视频录制
        record_episodes: 录制前 N 个 episode
        """
        if record:
            self.recorder.start_recording("dqn_train")

        # 使用 SB3 自带的回调来在每 step 触发
        from stable_baselines3.common.callbacks import BaseCallback

        class RecordCallback(BaseCallback):
            def __init__(self, recorder: VideoRecorder, record_episodes: int):
                super().__init__(verbose=0)
                self.recorder = recorder
                self.record_episodes = record_episodes
                self.episode_count = 0

            def _on_step(self) -> bool:
                # 只在需要录制的前 record_episodes 个回合记录
                if self.episode_count < self.record_episodes:
                    self.recorder.add_frame()
                return True

            def _on_rollout_end(self) -> None:
                # rollout 结束即一个 episode 结束
                self.episode_count += 1
                if self.episode_count == self.record_episodes:
                    self.recorder.end_recording()

        callback = RecordCallback(self.recorder, record_episodes) if record else None
        self.model.learn(total_timesteps=total_timesteps, callback=callback)
        self.save()  # 保存模型

    # --------------------------------------------------
    # 2. 评估 + 录制（默认录一次）
    # --------------------------------------------------
    def evaluate(self, episodes: int = 5, record: bool = True):
        if record:
            self.recorder.start_recording("dqn_eval")

        from stable_baselines3.common.evaluation import evaluate_policy
        mean_reward, _ = evaluate_policy(
            self.model,
            self.env,
            n_eval_episodes=episodes,
            deterministic=True,
            render=False,  # 禁用自带的 render
        )

        if record:
            # 手动录制每一帧
            for episode in range(episodes):
                obs, _ = self.env.reset()
                done = False
                while not done:
                    action, _ = self.model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, _ = self.env.step(action)
                    self.recorder.add_frame(self.env.render())  # 手动添加帧
            self.recorder.end_recording()

        print(f"[SB3-DQN] 平均回报: {mean_reward:.2f}")

    # --------------------------------------------------
    # 3. 保存模型
    # --------------------------------------------------
    def save(self, path: str = None):
        """
        保存训练好的模型
        :param path: 保存路径，默认为 self.model_path
        """
        if path is None:
            path = self.model_path
        self.model.save(path)
        print(f"Model saved to: {path}")
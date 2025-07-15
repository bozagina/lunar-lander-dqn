# utils/recorder.py
import cv2
import numpy as np
import gymnasium as gym
from typing import List, Tuple
import os, datetime, imageio

class VideoRecorder:
    """
    用法：
        recorder = VideoRecorder(env, fps=30)
        recorder.start_recording("dqn")
        ...
        recorder.add_frame(obs)          # 每步把当前观测加进去
        ...
        recorder.end_recording()
    """
    def __init__(self, env: gym.Env, fps: int = 30):
        self.env = env
        self.fps = fps
        self.frames = []
        self.enabled = False
        os.makedirs("videos", exist_ok=True)

    def start_recording(self, tag: str):
        self.tag = tag
        self.enabled = True
        self.frames.clear()

    def add_frame(self, obs=None):
        if not self.enabled:
            return
        # 如果 env 支持 render()，优先用 render() 拿 RGB
        try:
            rgb = self.env.render()
        except:
            # 否则用 obs 做占位（CartPole 的 obs 是 4 维向量，画成条形图）
            rgb = self._obs_to_img(obs)
        self.frames.append(rgb)

    def end_recording(self):
        if not self.enabled:
            return
        ts = datetime.datetime.now().strftime("%m%d_%H%M%S")
        filename = f"videos/{self.tag}_{ts}.mp4"
        imageio.mimsave(filename, self.frames, fps=self.fps)
        print(f"Video saved: {filename}")
        self.enabled = False
        self.frames.clear()

    # —— 把 4 维观测画成 64×64 的条形图，仅作 fallback ——
    def _obs_to_img(self, obs):
        img = np.zeros((64, 64, 3), dtype=np.uint8)
        if obs is None:
            return img
        for i, val in enumerate(obs):
            h = int(abs(val) * 6)
            img[-h:, i*16:(i+1)*16] = [255, 255, 255]
        return img


class MultiAlgoRecorder(VideoRecorder):
    """
    并排对比多个算法
    用法：
        mar = MultiAlgoRecorder(env, fps=30, grid=(1,2))  # 1行2列
        mar.start_recording(["dqn", "ppo"])
        ...
        mar.add_frames([obs_dqn, obs_ppo])
        ...
        mar.end_recording()
    """
    def __init__(self, env, fps=30, grid: Tuple[int, int] = (1, 2)):
        super().__init__(env, fps)
        self.grid = grid   # (rows, cols)
        self.tags = []

    def start_recording(self, tags: List[str]):
        self.tags = tags
        super().start_recording("_vs_".join(tags))

    def add_frames(self, obss: List[np.ndarray]):
        imgs = []
        for obs in obss:
            try:
                imgs.append(self.env.render())
            except:
                imgs.append(self._obs_to_img(obs))
        # 拼成网格
        rows, cols = self.grid
        h, w, _ = imgs[0].shape
        canvas = np.zeros((h*rows, w*cols, 3), dtype=np.uint8)
        for idx, img in enumerate(imgs):
            r, c = divmod(idx, cols)
            canvas[r*h:(r+1)*h, c*w:(c+1)*w] = img
        self.frames.append(canvas)
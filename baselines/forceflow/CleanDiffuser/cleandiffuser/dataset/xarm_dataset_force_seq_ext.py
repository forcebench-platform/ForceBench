import os
import json
import numpy as np
import torch
import zarr
from typing import Optional
from cleandiffuser.utils import MinMaxNormalizer, create_indices
from cleandiffuser.utils.codecs import jpeg  # noqa


class xArmDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, To: int = 1, Ta: int = 64, T_force: int = 10, normalizer_path: Optional[str] = None):
        super().__init__()
        self.root = zarr.open(file_path, "r")

        self.To, self.Ta, self.T_force = To, Ta, T_force
        self._episode_ends = self.root.meta.episode_ends[:]

        self.indices = create_indices(
            episode_ends=self._episode_ends,
            sequence_length=Ta + To - 1,
            pad_before=To - 1,
            pad_after=Ta - 1,
        )
        self.episode_idx = np.empty((len(self.indices),), dtype=int)
        for i in range(len(self.indices)):
            end_idx = self.indices[i][-1]
            self.episode_idx[i] = np.searchsorted(self._episode_ends, end_idx)

        self.size = self.root.data.pos.shape[0]

        # initialize normalizers
        self._init_normalizers(normalizer_path)

        self._obs_meta = [
            "rgb_arm",
            "rgb_fix",
            "pos",
            "force",
            "gripper_state",
        ]

    def _init_normalizers(self, normalizer_path: Optional[str]):
        """Initialize MinMax normalizers for pos, action, and force from JSON if provided.
        Expected JSON schema:
        {
          "pos": {"max": [..6..], "min": [..6..]},
          "action": {"max": [..6..], "min": [..6..]},
          "force": {"max": [..6..], "min": [..6..]}
        }
        Note: delta_force normalizer is no longer needed as we directly predict predicted_force.
        """
        if normalizer_path and os.path.exists(normalizer_path):
            with open(normalizer_path, "r") as f:
                info = json.load(f)
            pos_max = np.array(info["pos"]["max"], dtype=np.float32)
            pos_min = np.array(info["pos"]["min"], dtype=np.float32)
            act_max = np.array(info["action"]["max"], dtype=np.float32)
            act_min = np.array(info["action"]["min"], dtype=np.float32)
            force_max = np.array(info["force"]["max"], dtype=np.float32)
            force_min = np.array(info["force"]["min"], dtype=np.float32)

            assert pos_max.shape[0] == 6 and pos_min.shape[0] == 6, "pos normalizer must be 6-dim"
            assert act_max.shape[0] == 6 and act_min.shape[0] == 6, "action normalizer must be 6-dim"
            assert force_max.shape[0] == 6 and force_min.shape[0] == 6, "force normalizer must be 6-dim"

            self.pos_normalizer = MinMaxNormalizer(X_max=pos_max, X_min=pos_min)
            self.action_normalizer = MinMaxNormalizer(X_max=act_max, X_min=act_min)
            self.force_normalizer = MinMaxNormalizer(X_max=force_max, X_min=force_min)
            return
        else:
            raise ValueError(f"Normalizer file {normalizer_path} does not exist")
    


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        (
            buffer_start_idx,
            buffer_end_idx,
            sample_start_idx,
            sample_end_idx,
            end_idx,
        ) = self.indices[idx]

        e_Ta = self.Ta - (self.To + self.Ta - 1 - sample_end_idx)
        action_6d = self.root.data.action[buffer_end_idx - e_Ta : buffer_end_idx]
        action_6d = self.action_normalizer.normalize(action_6d)
        
        # 获取predicted_force：执行动作后下一时刻的实际接触力
        # 对于每个动作时间步t，我们需要t+1时刻的force作为标签
        # action序列: [buffer_end_idx - e_Ta : buffer_end_idx]
        # predicted_force序列: [buffer_end_idx - e_Ta + 1 : buffer_end_idx + 1]
        next_force_start_idx = buffer_end_idx - e_Ta + 1
        next_force_end_idx = buffer_end_idx + 1
        
        # 找到当前样本所在的episode结束位置
        episode_idx = self.episode_idx[idx]
        episode_end_idx = self._episode_ends[episode_idx]
        
        # 读取下一时刻的force，确保不超出episode边界
        # 如果next_force_end_idx超出episode边界，则用最后一个force值填充
        if next_force_end_idx > episode_end_idx:
            # 读取到episode结束
            next_force_raw = self.root.data.force[next_force_start_idx : episode_end_idx]
            # 用最后一个值填充不足的部分
            pad_length = next_force_end_idx - episode_end_idx
            last_force = self.root.data.force[episode_end_idx - 1:episode_end_idx]
            next_force_raw = np.concatenate([next_force_raw] + [last_force] * pad_length, axis=0)
        else:
            next_force_raw = self.root.data.force[next_force_start_idx : next_force_end_idx]
        
        # 获取夹爪动作数据
        gripper_action = self.root.data.gripper_action[buffer_end_idx - e_Ta : buffer_end_idx]
        
        # 对next_force进行归一化（使用force的归一化器）
        predicted_force = self.force_normalizer.normalize(next_force_raw)
        
        # 拼接6维位移动作 + 1维夹爪动作 + 6维predicted_force = 13维
        action = np.concatenate((action_6d, gripper_action, predicted_force), axis=-1)
        if self.To + self.Ta - 1 > sample_end_idx:
            action = np.pad(
                action,
                ((0, self.To + self.Ta - 1 - sample_end_idx), (0, 0)),
                mode="edge",
            )
        assert action.shape[0] == self.Ta, f"{action.shape[0]} != {self.Ta}"

        observation = dict()
        for obs_name in self._obs_meta:
            # force使用特殊处理，跳过这里
            if obs_name == "force":
                continue
                
            this_start_idx = buffer_start_idx
            x = self.root.data[obs_name][
                this_start_idx : buffer_start_idx + self.To - sample_start_idx
            ]
            if sample_start_idx > 0:
                if obs_name == "rgb_arm" or obs_name == "rgb_fix":
                    x = np.pad(x, ((sample_start_idx, 0), (0, 0), (0, 0), (0, 0)), mode="edge")
                else:
                    x = np.pad(x, ((sample_start_idx, 0), (0, 0)), mode="edge")

            if obs_name == "pos":
                x = self.pos_normalizer.normalize(x).astype(np.float32)
            elif obs_name == "rgb_arm" or obs_name == "rgb_fix":
                x = x.astype(np.float32) / 255.0
            elif obs_name == "gripper_state":
                x = x.astype(np.float32)  # 夹爪状态不需要归一化，已经是二元化的
            observation[obs_name] = x

        # 处理force的历史序列 (T_force步)
        # force序列的结束位置应该对应当前观测的最后一帧的下一个位置
        # 这样force_seq[:-1]是历史，force_seq[-1]对应最近的观测时刻（如果To=2，则force历史取到第2帧的位置）
        # 实际上我们希望force历史结束在当前观测结束的位置
        force_end_idx = buffer_start_idx + self.To - sample_start_idx
        
        # 向前取T_force步
        force_start_idx = force_end_idx - self.T_force
        
        # 找到当前样本所在的episode起始位置
        episode_idx = self.episode_idx[idx]
        episode_start_idx = 0 if episode_idx == 0 else self._episode_ends[episode_idx - 1]
        
        # 确保force序列不跨越episode边界（不能早于episode开始）
        if force_start_idx < episode_start_idx:
            force_start_idx = episode_start_idx
        
        # 读取force历史序列
        force_seq = self.root.data.force[force_start_idx:force_end_idx]
        
        # 如果force序列长度不足T_force（例如episode刚开始，历史不足10步）
        # 在前面用最早的force值进行edge padding
        if force_seq.shape[0] < self.T_force:
            pad_length = self.T_force - force_seq.shape[0]
            # mode="edge" 会用第一个值（最早的force）来填充前面的缺失部分
            force_seq = np.pad(force_seq, ((pad_length, 0), (0, 0)), mode="edge")
        
        # 归一化force序列
        force_seq = self.force_normalizer.normalize(force_seq).astype(np.float32)
        
        # 确保输出形状正确
        assert force_seq.shape == (self.T_force, 6), f"Force sequence shape mismatch: {force_seq.shape} vs expected ({self.T_force}, 6)"

        # 将夹爪状态拼接到位姿后面，形成7维的pos_with_gripper
        pos_with_gripper = np.concatenate((observation["pos"], observation["gripper_state"]), axis=-1)
        
        return {
            "obs": {
                "pos": torch.tensor(pos_with_gripper, dtype=torch.float32),  # (To, 7)：6维位姿 + 1维夹爪状态
                "rgb_arm": torch.tensor(observation["rgb_arm"], dtype=torch.float32),  # (To, C, H, W)
                "rgb_fix": torch.tensor(observation["rgb_fix"], dtype=torch.float32),  # (To, C, H, W)
                "force": torch.tensor(force_seq, dtype=torch.float32),  # (T_force, 6)：历史10步的力序列
            },
            "action": torch.tensor(action.astype(np.float32), dtype=torch.float32),  # (Ta, 13)
        }

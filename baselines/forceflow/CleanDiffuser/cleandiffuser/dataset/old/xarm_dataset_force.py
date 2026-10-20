import os
import json
import numpy as np
import torch
import zarr
from typing import Optional
from cleandiffuser.utils import MinMaxNormalizer, create_indices
from cleandiffuser.utils.codecs import jpeg  # noqa


class xArmDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, To: int = 1, Ta: int = 64, normalizer_path: Optional[str] = None):
        super().__init__()
        self.root = zarr.open(file_path, "r")

        self.To, self.Ta= To, Ta
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
            "rgb",
            "pos",
            "gripper_state",
            "force",
        ]
        
        # 检查是否存在预计算的delta_force数据
        self.has_precomputed_delta_force = 'delta_force' in self.root.data
        if self.has_precomputed_delta_force:
            print(f"✅ 使用预计算的delta_force数据，形状: {self.root.data.delta_force.shape}")
        else:
            print(f"⚠️ 未找到预计算的delta_force数据，将实时计算（影响性能）")

    def _init_normalizers(self, normalizer_path: Optional[str]):
        """Initialize MinMax normalizers for pos, action, force, and delta_force from JSON if provided.
        Expected JSON schema:
        {
          "pos": {"max": [..6..], "min": [..6..]},
          "action": {"max": [..6..], "min": [..6..]},
          "force": {"max": [..6..], "min": [..6..]},
          "delta_force": {"max": [..6..], "min": [..6..]}
        }
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
            delta_force_max = np.array(info["delta_force"]["max"], dtype=np.float32)
            delta_force_min = np.array(info["delta_force"]["min"], dtype=np.float32)

            assert pos_max.shape[0] == 6 and pos_min.shape[0] == 6, "pos normalizer must be 6-dim"
            assert act_max.shape[0] == 6 and act_min.shape[0] == 6, "action normalizer must be 6-dim"
            assert force_max.shape[0] == 6 and force_min.shape[0] == 6, "force normalizer must be 6-dim"
            assert delta_force_max.shape[0] == 6 and delta_force_min.shape[0] == 6, "delta_force normalizer must be 6-dim"

            self.pos_normalizer = MinMaxNormalizer(X_max=pos_max, X_min=pos_min)
            self.action_normalizer = MinMaxNormalizer(X_max=act_max, X_min=act_min)
            self.force_normalizer = MinMaxNormalizer(X_max=force_max, X_min=force_min)
            self.delta_force_normalizer = MinMaxNormalizer(X_max=delta_force_max, X_min=delta_force_min)
            return
        else:
            raise ValueError(f"Normalizer file {normalizer_path} does not exist")

    def _compute_expect_force(self, start_idx: int, end_idx: int) -> np.ndarray:
        """计算期望力：未来三步force的加权平均（返回原始值）
        
        Args:
            start_idx: 开始索引
            end_idx: 结束索引
            
        Returns:
            期望力数组（原始值），形状为(end_idx - start_idx, 6)
        """
        sequence_length = end_idx - start_idx
        expect_force = np.zeros((sequence_length, 6), dtype=np.float32)
        
        # 设置未来三步力的权重：0.7, 0.2, 0.1，加权作为期望力
        weights = np.array([0.7, 0.2, 0.1], dtype=np.float32)
        
        for i in range(sequence_length):
            current_step = start_idx + i
            
            # 找到当前步骤所在的episode边界
            current_episode_end = None
            for episode_end in self._episode_ends:
                if current_step < episode_end:
                    current_episode_end = episode_end
                    break
            
            # 获取未来三步的force数据并计算加权平均
            future_forces = np.zeros((3, 6), dtype=np.float32)
            
            for j in range(3):
                future_idx = current_step + j + 1
                
                # 检查是否超出当前episode边界或数据集边界
                if (current_episode_end is not None and 
                    future_idx < current_episode_end and 
                    future_idx < self.root.data.force.shape[0]):
                    future_forces[j] = self.root.data.force[future_idx]
                else:
                    # 如果超出边界，使用当前episode的最后一步force值
                    if current_episode_end is not None:
                        boundary_idx = min(current_episode_end - 1, self.root.data.force.shape[0] - 1)
                    else:
                        boundary_idx = self.root.data.force.shape[0] - 1
                    future_forces[j] = self.root.data.force[boundary_idx]
            
            # 计算加权平均作为期望力（保持原始值，不归一化）
            expect_force[i] = np.average(future_forces, axis=0, weights=weights)
        
        return expect_force
    


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
        action_gripper = self.root.data.gripper_action[buffer_end_idx - e_Ta : buffer_end_idx]
        
        # 获取delta_force：优先使用预计算数据，否则实时计算
        if self.has_precomputed_delta_force:
            # 直接读取预计算的delta_force（原始值）
            delta_force_raw = self.root.data.delta_force[buffer_end_idx - e_Ta : buffer_end_idx]
        else:
            # 实时计算delta_force (当前原始力 - 期望原始力)
            current_force_raw = self.root.data.force[buffer_end_idx - e_Ta : buffer_end_idx]
            expect_force_raw = self._compute_expect_force(buffer_end_idx - e_Ta, buffer_end_idx)
            delta_force_raw = current_force_raw - expect_force_raw
        
        # 对原始delta_force进行归一化
        delta_force = self.delta_force_normalizer.normalize(delta_force_raw)
        
        action = np.concatenate((action_6d, action_gripper, delta_force), axis=-1)
        if self.To + self.Ta - 1 > sample_end_idx:
            action = np.pad(
                action,
                ((0, self.To + self.Ta - 1 - sample_end_idx), (0, 0)),
                mode="edge",
            )
        assert action.shape[0] == self.Ta, f"{action.shape[0]} != {self.Ta}"

        observation = dict()
        for obs_name in self._obs_meta:
            this_start_idx = buffer_start_idx
            x = self.root.data[obs_name][
                this_start_idx : buffer_start_idx + self.To - sample_start_idx
            ]
            if sample_start_idx > 0:
                if obs_name == "rgb":
                    x = np.pad(x, ((sample_start_idx, 0), (0, 0), (0, 0), (0, 0)), mode="edge")
                else:
                    x = np.pad(x, ((sample_start_idx, 0), (0, 0)), mode="edge")

            if obs_name == "pos":
                x = self.pos_normalizer.normalize(x).astype(np.float32)
            elif obs_name == "gripper_state":
                x = x.astype(np.float32)
            elif obs_name == "rgb":
                x = x.astype(np.float32) / 255.0
            elif obs_name == "force":
                x = self.force_normalizer.normalize(x).astype(np.float32)
            observation[obs_name] = x

        return {
            "obs": {
                "pos": torch.tensor(observation["pos"], dtype=torch.float32),
                "gripper_state": torch.tensor(observation["gripper_state"], dtype=torch.float32),
                "rgb": torch.tensor(observation["rgb"], dtype=torch.float32),
                "force": torch.tensor(observation["force"], dtype=torch.float32),
            },
            "action": torch.tensor(action.astype(np.float32), dtype=torch.float32),
        }

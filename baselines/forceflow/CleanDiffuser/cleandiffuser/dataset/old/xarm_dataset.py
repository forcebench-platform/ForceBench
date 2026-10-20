import os
import json
import numpy as np
import torch
import zarr
from typing import Optional
from cleandiffuser.utils import MinMaxNormalizer, create_indices
from cleandiffuser.utils.codecs import jpeg  # noqa


class xArmDataset(torch.utils.data.Dataset):
    def __init__(self, file_path, To: int = 1, Ta: int = 64, Ts: int = 10, normalizer_path: Optional[str] = None):
        super().__init__()
        self.root = zarr.open(file_path, "r")

        self.To, self.Ta, self.Ts = To, Ta, Ts
        self._episode_ends = self.root.meta.episode_ends[:]

        self.indices = create_indices(
            episode_ends=self._episode_ends,
            sequence_length=Ta + To - 1,  # 使用To而不是Ts
            pad_before=To - 1,            # 使用To而不是Ts
            pad_after=Ta - 1,
        )
        self.episode_idx = np.empty((len(self.indices),), dtype=int)
        for i in range(len(self.indices)):
            end_idx = self.indices[i][-1]
            self.episode_idx[i] = np.searchsorted(self._episode_ends, end_idx)

        self.size = self.root.data.pos.shape[0]

        # initialize normalizers
        self._init_normalizers(normalizer_path)
        
        """
        self.pos_normalizer = MinMaxNormalizer(
            X_max=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),  # 根据实际数据调整
            X_min=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
        )

        self.action_normalizer = MinMaxNormalizer(
            X_max=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),  # 根据实际数据调整
            X_min=np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0]),
        )
        """

        self._obs_meta = [
            "rgb",
            "pos",
            "gripper_state",
        ]

    def _init_normalizers(self, normalizer_path: Optional[str]):
        """Initialize MinMax normalizers for pos and action from JSON if provided.
        Expected JSON schema:
        {
          "pos": {"max": [..6..], "min": [..6..]},
          "action": {"max": [..6..], "min": [..6..]}
        }
        """
        if normalizer_path and os.path.exists(normalizer_path):
            with open(normalizer_path, "r") as f:
                info = json.load(f)
            pos_max = np.array(info["pos"]["max"], dtype=np.float32)
            pos_min = np.array(info["pos"]["min"], dtype=np.float32)
            act_max = np.array(info["action"]["max"], dtype=np.float32)
            act_min = np.array(info["action"]["min"], dtype=np.float32)

            assert pos_max.shape[0] == 6 and pos_min.shape[0] == 6, "pos normalizer must be 6-dim"
            assert act_max.shape[0] == 6 and act_min.shape[0] == 6, "action normalizer must be 6-dim"

            self.pos_normalizer = MinMaxNormalizer(X_max=pos_max, X_min=pos_min)
            self.action_normalizer = MinMaxNormalizer(X_max=act_max, X_min=act_min)
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
        action_gripper = self.root.data.gripper_action[buffer_end_idx - e_Ta : buffer_end_idx]
        action = np.concatenate((action_6d, action_gripper), axis=-1)
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
            observation[obs_name] = x

        return {
            "obs": {
                "pos": torch.tensor(observation["pos"], dtype=torch.float32),
                "gripper_state": torch.tensor(observation["gripper_state"], dtype=torch.float32),
                "rgb": torch.tensor(observation["rgb"], dtype=torch.float32),
            },
            "action": torch.tensor(action.astype(np.float32), dtype=torch.float32),
        }

import json

import h5py
import numpy as np
import pytest

from forcebench.recording import EpisodeBuffer, HDF5EpisodeWriter, validate_hdf5_episode


def test_hdf5_records_both_holes_and_peg(tmp_path, monkeypatch):
    monkeypatch.setenv("FORCEBENCH_EXPERT_COLLECTION_CONTRACT", "forcebench_expert_collection")
    monkeypatch.setenv("FORCEBENCH_EXPERT_BACKEND", "scripted")
    monkeypatch.setenv("FORCEBENCH_TASK", "peg_insert")
    episode = EpisodeBuffer()
    for frame in range(2):
        episode.append(
            state=np.zeros(7),
            action=np.zeros(7),
            wrench_raw=np.full(6, frame + 1, dtype=np.float32),
            wrench_ema=np.full(6, 10 + frame, dtype=np.float32),
            timestamp=(frame + 1) / 15,
            image=np.zeros((10, 12, 3), dtype=np.uint8),
            wrist_image=np.zeros((8, 8, 3), dtype=np.uint8),
            joint_position=np.zeros(9),
            joint_velocity=np.zeros(9),
            tcp_pose=np.zeros(7),
            privileged={
                "objects/target_hole/pose": np.asarray([0.65, 0.0, 0.003, 0.0, 0.0, 0.0, 1.0]),
                "objects/pickup_hole/pose": np.asarray([0.50, 0.0, 0.003, 0.0, 0.0, 0.0, 1.0]),
                "objects/peg/pose": np.asarray([0.50, 0.0, 0.003, 0.0, 0.0, 0.0, 1.0]),
            },
            phase_id=frame,
        )
    path = HDF5EpisodeWriter(tmp_path).write(
        episode,
        {
            "task": "peg_insert",
            "success": True,
            "seed": 42,
            "initial_layout": {"initial_separation_m": 0.15},
            "scene_profile": "contact_lab",
            "images_present": True,
            "control_source": "zero_input",
            "phase_schema": {"scripted_prefix": 0, "force_rl": 1, "scripted_suffix": 2},
        },
        high_rate={
            "timestamp": np.arange(4, dtype=np.float64) / 60 + 1 / 60,
            "eef_wrench_joint_raw": np.zeros((4, 6), dtype=np.float32),
        },
    )[0]

    assert path.name == "seed_042_episode_01.hdf5"
    report = validate_hdf5_episode(path)
    assert report["success"]
    with h5py.File(path, "r") as handle:
        assert handle.attrs["schema_version"] == "forcebench_hdf5"
        assert handle["privileged/objects/pickup_hole/pose"].shape == (2, 7)
        assert handle["observations/images/main"].shape == (2, 10, 12, 3)
        assert handle["observations/images/wrist"].shape == (2, 8, 8, 3)
        np.testing.assert_array_equal(handle["phases/id"][:], [0, 1])
        np.testing.assert_allclose(handle["privileged/objects/pickup_hole/pose"][0, :3], [0.50, 0.0, 0.003])
        assert handle["forces/control/eef_wrench_joint_raw"].attrs["frame"] == "incoming_joint"
        assert handle["forces/control/eef_wrench_joint_raw"].attrs["joint"] == "panda_hand_joint"
        assert not handle["forces/control/eef_wrench_joint_raw"].attrs["gravity_compensated"]
        assert handle["forces/control/eef_wrench_joint_raw"].attrs["filter"] == "none"
        assert handle["forces/control/eef_wrench_joint_ema"].attrs["filter"] == "ema_alpha_0.25"
        np.testing.assert_array_equal(
            handle["forces/control/eef_wrench_joint_raw"][:, 0], [1.0, 2.0]
        )
        assert set(handle["forces/control"].keys()) == {
            "eef_wrench_joint_raw",
            "eef_wrench_joint_ema",
        }
        assert "teleop" not in handle
        metadata = json.loads(handle.attrs["metadata_json"])
        assert metadata["expert_policy_wrench_field"].endswith("eef_wrench_joint_raw")
        assert metadata["expert_recorded_wrench_fields"] == [
            "forces/control/eef_wrench_joint_raw",
            "forces/control/eef_wrench_joint_ema",
        ]
        assert not any(str(key).startswith("gello") for key in metadata)

    with h5py.File(path, "r+") as handle:
        handle["timestamps/physics"][:] = np.arange(1, 5, dtype=np.float64) / 120
    with pytest.raises(ValueError, match="physics timestamps"):
        validate_hdf5_episode(path)


def test_hdf5_episode_numbers_are_independent_per_seed(tmp_path):
    def write(seed):
        episode = EpisodeBuffer()
        episode.append(
            state=np.zeros(7),
            action=np.zeros(7),
            wrench_raw=np.zeros(6),
            wrench_ema=np.zeros(6),
            timestamp=1 / 15,
            image=np.zeros((2, 2, 3), dtype=np.uint8),
            wrist_image=np.zeros((2, 2, 3), dtype=np.uint8),
            joint_position=np.zeros(9),
            joint_velocity=np.zeros(9),
            tcp_pose=np.zeros(7),
            privileged={"objects/peg/pose": np.zeros(7)},
        )
        return HDF5EpisodeWriter(tmp_path).write(
            episode,
            {
                "task": "test",
                "success": True,
                "seed": seed,
                "initial_layout": {},
                "scene_profile": "contact_lab",
                "control_source": "test",
                "images_present": True,
            },
            high_rate={
                "timestamp": np.asarray([1 / 60], dtype=np.float64),
                "eef_wrench_joint_raw": np.zeros((1, 6), dtype=np.float32),
            },
        )[0]

    assert write(3).name == "seed_003_episode_01.hdf5"
    assert write(4).name == "seed_004_episode_01.hdf5"
    assert write(3).name == "seed_003_episode_02.hdf5"
    assert HDF5EpisodeWriter(tmp_path).successful_episode_count() == 3

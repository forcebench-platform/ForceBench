# 新增 ForceBench demo

新增 demo 不需要修改 registry、共享环境、recording writer 或启动器。

## 最小目录

```text
forcebench/tasks/my_demo/
  __init__.py   # 纯 Python MANIFEST；不得导入 Isaac Sim
  spec.py       # 可选：随机分布和成功判定的纯数学定义
  runtime.py    # TaskRuntime 实现
```

`__init__.py`：

```python
from forcebench.task_api import TaskManifest

MANIFEST = TaskManifest(
    name="my_demo",
    prompt="完成任务。",
    runtime_class="forcebench.tasks.my_demo.runtime:MyDemoTask",
    dataset_subdir="my_demo",
    scene_profile="contact_lab",
    description="一句话任务说明。",
)
```

`TaskRuntime` 必须实现：

- `validate_assets()`
- `scene_entities()`
- `sample_initial_state(seed)`
- `reset_scene(env, env_ids, layout)`
- `evaluate_final_state(env)`
- `privileged_observations(env)`
- `wrench_reference_position(env)`

可选 hook：`spawn_scene_prims`、`after_scene_setup`、`on_control_step`、
`episode_metadata`、`instruction`、`failure_reasons`、`debug_state`。

约束：

- runtime 的 `manifest` 必须与包导出的 `MANIFEST` 完全一致。
- `dataset_subdir` 不能绝对路径、不能包含 `..`，且不能和其他任务重复。
- `scene_profile` 必须是 `forcebench.scene_profiles` 已注册的名称。
- 任务不得覆写 canonical camera，也不得自行创建 recorder 或环境副本。
- 成功判定应由 `evaluate_final_state()` 集中完成；正式 writer 只保存成功 episode。

若任务需要自动专家数据，在 `scripts/expert/` 中增加该任务独立的生成脚本，
并在 `forcebench/expert_adapters/` 中增加对应插件。装配类任务可以读取声明过的
`privileged_observations` 做闭环轨迹求解，但不得 teleport 物体、overwrite
机器人状态或绕过统一 SmoothPosition 控制器。

验证发现和导入：

```bash
cd /home/pc/ForceBench
python -c 'from forcebench.registry import task_names; print(task_names())'
FORCEBENCH_TASK=my_demo ./run_native.sh smoke
```

# 策略接入

ForceBench 的核心设计是：**换模型，不换 harness**。评测循环只认识一个观测类型和
一个动作类型，模型特有的一切（张量布局、传输协议、相对/绝对动作转换）都被隔离在
适配器里。

代码真值：[`forcebench/policy_api.py`](../forcebench/policy_api.py)、
[`forcebench/policy_adapters/`](../forcebench/policy_adapters/)。

## 1. 契约总览

```
PolicyObservation  ──►  你的适配器  ──►  PolicyActionChunk
   （15 Hz 一帧）        （你写的）        （15 Hz 绝对 TCP 目标）
```

评测器在执行任何一帧之前都会校验这个契约。适配器返回的东西不符合契约就直接报错，
不会静默地跑出一个无意义的结果。

关键常量（`policy_api.py`）：

| 常量 | 值 |
|---|---|
| `POLICY_CONTRACT` | `"forcebench_policy"` |
| `CANONICAL_ACTION_SPACE` | `"world_tcp_xyz_rpy_gripper_open_minus1_plus1"` |
| `STATE_SEMANTICS` | 同上 |
| `WRENCH_SEMANTICS` | `"panda_hand_joint_parent_on_child_fx_fy_fz_tx_ty_tz"` |
| `WRENCH_FILTER` | `"none"`（策略拿到的是 raw，不是 EMA） |
| `POLICY_CONTROL_RATE_HZ` | `15.0` |

## 2. 输入：`PolicyObservation`

评测器构造的每一帧包含：

| 字段 | 类型 / 形状 | 含义 |
|---|---|---|
| `step` | `int >= 0` | 控制步序号 |
| `rollout_id` | 非空 `str` | 本次 rollout 的标识 |
| `state` | `float32 [7]` | 测得的世界 TCP `[x, y, z, roll, pitch, yaw, gripper]` |
| `main_image` | `uint8 [H, W, 3]` | 主相机 RGB |
| `wrist_image` | `uint8 [H, W, 3]` | 腕部相机 RGB |
| `eef_wrench` | `float32 [6]` | **raw** 六维 wrench `[Fx, Fy, Fz, Tx, Ty, Tz]` |
| `prompt` | 非空 `str` | 任务语言指令 |

关于各字段的语义细节：

- **`state`**：物理步进**之后**测得的量，不是目标值。`gripper` 为 `-1` 闭合、
  `+1` 张开。TCP 的 RPY 是**连续的**——环境已经处理好 `±pi` 分支跳变，
  你拿到的角度不会在 `+pi`/`-pi` 之间突跳。
- **`eef_wrench`**：通过 `panda_hand_joint` 作用在 `force_sensor` link 上的入向固定
  关节反作用力，表达在入向关节坐标系中，父作用于子。**不做坐标变换，不做重力补偿，
  不做滤波。** HDF5 里另存了一路 `alpha=0.25` 的 EMA 通道，但那是给离线分析用的，
  **不是**策略输入。
- **`prompt`**：来自任务 manifest，见各[任务卡片](./tasks/README.md)的"任务目标"一节。

### 可选的历史

`PolicyObservation` 还支持四个可选的历史字段：`state_history [T,7]`、
`main_image_history [T,H,W,3]`、`wrist_image_history [T,H,W,3]`、
`eef_wrench_history [T,6]`。

**要么四个都给，要么四个都不给**——只给一部分会被拒绝。给了的话还必须满足：

- 四者的时间长度 `T` 一致；
- 历史图像的 `H, W, 3` 与当前图像一致；
- **每个历史序列的最后一帧必须等于对应的当前帧**（`state`、两张图、`eef_wrench`）。

这些不变式在 `__post_init__` 里强制检查，目的是防止历史和当前帧错位这种很难查的 bug。

## 3. 输出：`PolicyActionChunk`

```python
PolicyActionChunk(actions=...)   # actions: float32 [H, 7], H >= 1
```

| 字段 | 要求 |
|---|---|
| `actions` | `float32 [H, 7]`，`H >= 1`，全部有限（无 NaN/Inf） |
| `action_space` | 必须是 `world_tcp_xyz_rpy_gripper_open_minus1_plus1` |
| `control_rate_hz` | 必须是 `15.0` |

每一行是一个**绝对**的世界坐标系 TCP 目标：
`[target_x, target_y, target_z, target_roll, target_pitch, target_yaw, target_gripper]`。

**适配器内部想怎么转换都可以**——你的模型输出相对位移、关节角、或者别的动作空间都行，
但**返回给 ForceBench 的必须是 15 Hz 的世界系绝对 TCP 目标**。这是整个 benchmark
可比性的基础：所有模型经过完全相同的控制器和物理，差异只来自策略本身。

评测器默认每次推理执行动作块的前 **50** 帧（`FORCEBENCH_EXECUTE_CHUNK`），
然后再请求一次推理。

## 4. 写一个适配器

在 `forcebench/policy_adapters/` 下新增一个**不以下划线开头**的模块，导出：

1. 一个名为 `MANIFEST` 的 `PolicyAdapterManifest`；
2. 一个被 `MANIFEST.adapter_class` 引用的 `PolicyAdapter` 子类。

注册表会**自动发现**它，不需要改评测循环或任何 switch。

```python
# forcebench/policy_adapters/my_model_ws.py
from forcebench.policy_api import (
    PolicyAdapter, PolicyAdapterManifest, PolicyActionChunk, PolicyObservation,
)

MANIFEST = PolicyAdapterManifest(
    name="my_model_ws",
    adapter_class="forcebench.policy_adapters.my_model_ws:MyModelAdapter",
    description="一句话说明。",
)


class MyModelAdapter(PolicyAdapter):
    manifest = MANIFEST

    def __init__(self, host: str, port: int, **options):
        ...  # 建立到你的推理服务的连接

    @property
    def endpoint_uri(self) -> str:
        return f"ws://{self._host}:{self._port}"

    def infer(self, observation: PolicyObservation) -> PolicyActionChunk:
        # 1. 把 observation 重排成你的模型要的格式
        # 2. 发给推理服务，拿回结果
        # 3. 转换成 [H, 7] 的世界系绝对 TCP 目标
        return PolicyActionChunk(actions=my_actions)

    def close(self) -> None:
        ...  # 释放传输资源
```

必须实现的三个成员：`endpoint_uri`（属性）、`infer()`、`close()`。

选用适配器：

```bash
export FORCEBENCH_POLICY_ADAPTER=my_model_ws
export FORCEBENCH_POLICY_ADAPTER_OPTIONS='{"temperature": 0.0}'   # 可选，JSON
```

`FORCEBENCH_POLICY_ADAPTER_OPTIONS` 里的 JSON 会作为关键字参数传给适配器构造函数。

## 5. 内置适配器

| 名称 | 说明 |
|---|---|
| `openpi_ws` | π0 / π0.5 系列 |
| `forcevla_ws` | ForceVLA（力融合 MoE） |
| `forcebench_ws` | ForceBench 原生契约 |

**新写的推理服务建议直接对接 `forcebench_ws` 的原生契约**，另外两个是为了兼容既有
模型服务而存在的。原生契约的线上负载可以用
`PolicyObservation.canonical_request()` 生成，形如：

```python
{
  "contract": "forcebench_policy",
  "step": 0,
  "rollout_id": "...",
  "control_rate_hz": 15.0,
  "observation": {
    "state": <float32[7]>,
    "state_semantics": "world_tcp_xyz_rpy_gripper_open_minus1_plus1",
    "images": {"main": <uint8[H,W,3]>, "wrist": <uint8[H,W,3]>},
    "eef_wrench": <float32[6]>,
    "wrench_semantics": "panda_hand_joint_parent_on_child_fx_fy_fz_tx_ty_tz",
    "wrench_filter": "none",
    "prompt": "..."
  }
}
```

（给了历史时会多一个 `observation.history` 子字典。）

## 6. 力信息强度分级

ForceBench 的设计目标之一是把"策略拿到多少力信息"做成一个**可分级的变量**，
而不是一个二值开关：

| 档位 | 策略拿到什么 |
|---|---|
| 仅本体感知 | `state`（其中 `gripper` 间接反映负载），不用 `eef_wrench` |
| 原始 6 维 wrench | 完整的 `eef_wrench`，作为额外输入通道拼接进去 |
| 架构级融合 | 在模型结构层面融合力信号（如 ForceVLA 的力融合 MoE） |

**这个分级由适配器实现，而不是由环境实现**——环境永远提供完整的 `eef_wrench`，
是适配器决定把多少信息喂给模型。这样"同一个模型的有力 / 无力版本"跑的是完全相同的
物理和判据，差异只在于信息通道。

> **注意：** 上表描述的是设计意图和当前可行的做法。ForceBench 目前**没有**提供一个
> 现成的"力信息屏蔽"开关或标准化的分级配置文件——要跑无力对照，你需要在自己的适配器
> 或模型侧做屏蔽。把这个分级固化成配置是 v1.0 的待办项。

## 7. 报告结果时

请务必遵守[评测协议](./eval_protocol.md)：固定 seed `200`–`299`、环境间距 6 m、
按任务的步数上限、成功率必须附 95% Clopper-Pearson 区间。特别注意协议第 3 节关于
**步数上限会翻转结论**的警告，以及第 5 节关于**不要写总体性"力有没有用"结论**的要求。

# ForceBench 网站（web/）

ForceBench 的中文前端站点，含 leaderboard。**纯静态**：原生 HTML + CSS + 少量 vanilla JS，
没有构建步骤、没有 npm 依赖、不加载任何外部 CDN 资源（字体使用系统字体栈，图表是手写 SVG），
可以直接由 GitHub Pages 托管。

> 这么选型的理由：这是研究项目的原型站，维护者是研究者不是前端工程师。
> 一个「改一个数字要先 `npm install`」的站点会立刻腐烂。

---

## 目录结构

```
web/
├── index.html               首页：项目定位、评测进度、八任务概览
├── leaderboard.html         Leaderboard（核心页面）
├── tasks.html               八个任务的说明，区分两类任务
├── about.html               项目背景、与 ForceVLA 论文的区别、引用、许可证
├── assets/
│   ├── style.css            全部样式（浅色为主，附深色适配）
│   ├── data.js              数据加载 + 共享常量/工具（全局对象 FB）
│   └── render.js            各页面的渲染逻辑，按 <body data-page> 分发
├── sample_leaderboard.json  ★ 样例数据，仅供开发预览，数字全部虚构 ★
└── README.md                本文件
```

---

## 本地预览

站点会读取**仓库根目录**下的 `results/leaderboard.json`，所以必须从**仓库根目录**起服务，
而不是从 `web/` 目录起。

```bash
# 在仓库根目录执行
python -m http.server 8000
```

然后打开：

- 首页 <http://localhost:8000/web/index.html>
- Leaderboard <http://localhost:8000/web/leaderboard.html>

直接用 `file://` 双击打开 HTML **不行** —— 浏览器的同源策略会拦掉 `fetch()`，
页面会显示「数据尚未生成」。必须走 HTTP。

### 用样例数据预览页面样式

如果 `results/leaderboard.json` 还没生成，或者你只想看看页面长什么样，在网址后面加 `?data=sample`：

<http://localhost:8000/web/leaderboard.html?data=sample>

此时页面会加载 `web/sample_leaderboard.json`，并在顶部常驻一条**醒目的红色横幅**，
说明当前显示的是虚构数据。

> **`sample_leaderboard.json` 里的所有数字都是编出来的**，只为覆盖
> `complete` / `partial` / `missing` / `invalid` 四种状态的渲染，
> 不代表任何模型的实际表现，任何情况下都不得引用，也不要把它复制进 `results/`。

---

## 数据从哪来 / 要改数字改哪个文件

| 你想改什么 | 改哪里 |
|---|---|
| **榜单上的成功率、置信区间、状态** | 仓库根目录的 `results/leaderboard.json` |
| 任务的中文名、中文描述、`varies_zh` | 同上（`tasks[]` 里的 `display_name_zh` / `summary_zh` / `varies_zh`） |
| 模型的中文名、`force_input`、模型增减 | 同上（`models[]`） |
| 评测协议参数（seed 范围、rollout 数等） | 同上（`eval_protocol`） |
| p 值、显著性判定、检验说明 | 同上（`head_to_head`） |
| 页面文案、版式、配色 | `web/` 下的 HTML / `assets/style.css` |

任务列表、模型列表、表格列数、覆盖率分母**全部由数据文件决定**。
增删一个模型只需改 `models[]` 和对应的 `results[]` 行，页面会自动多出/少掉一列，
不需要动任何 HTML。

**页面里没有任何硬编码的评测数字。** 所有成绩都是运行时 `fetch('../results/leaderboard.json')`
读出来的。改完 JSON 刷新浏览器即可，不需要重新构建任何东西。

数据格式契约是 `forcebench-results-v1`。字段定义以生成端的 schema 文档为准，
简要说明见下方「数据契约要点」。

### 唯一的例外：任务说明的兜底文案

`assets/data.js` 里的 `TASK_FALLBACK` 存了八个任务的中文名和一句话说明，
**只在 `results/leaderboard.json` 读不到时**用于渲染「任务说明」页，
内容译自仓库 README 的任务表，**不含任何评测数字**。
数据文件一旦可用，页面就完全以数据文件里的 `display_name_zh` / `summary_zh` 为准。

---

## 数据契约要点（`forcebench-results-v1`）

顶层字段：`schema_version`、`benchmark_version`、`generated_at`、`eval_protocol`、
`models[]`、`tasks[]`、`results[]`。

- `models[].force_input` ∈ `none` / `wrench6_concat` / `moe_fusion`，
  模型 id 固定为 `pi05`、`pi05_force6`、`forcevla`。
- `tasks[].force_category` ∈ `force_classification`（5 个）/ `contact_rich`（3 个）。
  这个区分在前端始终可见，不要去掉。
- `results[]` 每个 (task, model) 组合都要有一条，**包括没跑过的**，
  用 `status` 标注：`complete` / `partial` / `missing` / `invalid`。
  前端不靠「缺行」来表示缺数据。
- 置信区间字段 `ci95_low` / `ci95_high` 必须是 **Clopper-Pearson 精确二项区间**，
  在生成数据时算好写进文件，前端只负责展示，不做任何统计计算。

### `head_to_head`（逐任务显著性检验）

顶层可选字段。有它时，leaderboard 表格会多出一列 p 值 + 显著性标记，
并在表格下方渲染一整节完整的逐任务对比表。

```json
"head_to_head": {
  "method": "Fisher exact test (two-sided), per task, pi05 vs forcevla",
  "family_size": 6,
  "bonferroni_alpha": 0.00833,
  "notes_zh": "……",
  "comparisons": [
    {"task": "weight_sort", "baseline": "pi05", "challenger": "forcevla",
     "baseline_rate": 0.91, "challenger_rate": 0.72, "diff_pp": -19.0,
     "p_value": 0.0009, "significant_at_005": true,
     "significant_after_bonferroni": true}
  ]
}
```

**p 值和显著性判定全部由生成端算好写入，前端不做任何统计计算**，
只根据 `significant_at_005` / `significant_after_bonferroni` 两个布尔值选择标记：
`★` 校正后仍显著 / `△` p&lt;0.05 但校正后不显著 / `○` 不显著。
`notes_zh` 会原样渲染在对比表上方，**请不要在里面写超出数据的结论**。

---

## 前端遵守的展示规则（改页面时不要破坏）

这几条是这个 benchmark 公信力的核心，实现时已刻意遵守：

1. **每个成功率旁必须显示 95% 置信区间。** 区间以文字（`95% CI 16.9–34.7%`）给出，
   宽屏另有误差棒 SVG 作为补充；窄屏隐藏图形但**文字始终保留**。
   n = 100 时 25% 的区间宽度接近 18 个百分点，只放点估计会误导读者。
2. **`missing` 与 `invalid` 必须在视觉上明确区别于「低分」。**
   `missing` 用虚线边框 + 斜纹底 + `○` + 文字「尚未评测」；
   `invalid` 用红色 45° 斜纹 + `⚠` + 文字「结果无效」，并把跑出来的数字加删除线、
   附脚注说明原因。两者都**不会**渲染成 0% 或空白。
3. **覆盖率必须显式呈现。** 首页和 leaderboard 顶部都有覆盖率面板：
   「N / 总格数 已完成」+ 进度条 + 四种状态的计数 + 任务 × 模型的状态矩阵。
   分母由数据文件决定，不写死。
4. **不下任何方向的因果结论，并且不让并排的百分数误导读者。**
   凡是把两个模型成功率并排展示的地方（表格行、对比表），
   **同一行必须同时给出该任务的 p 值和显著性标记**，
   并明确区分「p&lt;0.05」和「多重比较校正后仍显著」。
   页面不说「力有帮助」，也不说「力没用」；
   方向清点只在**达到显著的那几组**里做（否则会把噪声也算进方向）；
   并且**不提供跨任务平均成功率和总排名** —— 任务间差异太大，平均值会掩盖真实信号。
5. **原型状态声明**常驻首页和 leaderboard 顶部：v0.1 原型、覆盖不完整、数字会变、请勿引用。

另外：颜色不是唯一的信息载体（每种状态都同时有字形、文字标签和底纹），
表格用 `<th scope>`、`<caption>`、分组 `<tbody>`，页面有跳转正文链接。

---

## 开启 GitHub Pages

1. 打开仓库 **Settings → Pages**。
2. **Source** 选 `Deploy from a branch`。
3. **Branch** 选 `main`，目录选 **`/ (root)`**（不要选 `/docs`）。
4. 保存，等一两分钟。

站点地址：

```
https://pkuhanxuan.github.io/ForceBench/web/index.html
```

数据文件会被解析到 `https://pkuhanxuan.github.io/ForceBench/results/leaderboard.json`。

> **必须选 `/ (root)`。** 页面用 `../results/leaderboard.json` 相对路径读数据，
> 如果只发布 `web/` 子目录，`results/` 就不在站点里，leaderboard 会一直显示「数据尚未生成」。

想让 <https://pkuhanxuan.github.io/ForceBench/> 直接落到首页，可以在仓库根目录放一个
跳转用的 `index.html`（本目录不负责这件事，需要时另行添加）。

---

## 读不到数据时会怎样

如果 `results/leaderboard.json` 不存在或解析失败，页面显示明确的
**「评测数据尚未生成」**面板，附上请求的 URL 和失败原因，
并提供一个「仅查看页面样式（载入样例数据）」的按钮。

**页面不会静默回退到样例数据，也不会用 0 或空白填补缺失的格子。**
只有用户显式点了那个按钮（或自己在网址上加 `?data=sample`）才会加载样例数据，
且此时顶部一定有醒目的样例横幅。

---

## 浏览器兼容

用的都是老特性：`fetch`、`Array.prototype.forEach/filter/map`、CSS Grid/Flexbox、
CSS 自定义属性、`position: sticky`。没有 ES 模块、没有可选链、没有构建产物。
现代 Chrome / Edge / Firefox / Safari 都可以直接跑。

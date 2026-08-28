/* ForceBench 站点 —— 数据加载与共享工具
 * ---------------------------------------------------------------
 * 数据源唯一真相：仓库根目录的 results/leaderboard.json
 * 格式契约：forcebench-results-v1（见 web/README.md）
 *
 * 重要约定：
 *   真实数据文件读不到时，页面显示"数据尚未生成"，
 *   绝不静默回退到 sample_leaderboard.json。
 *   只有用户显式访问 ?data=sample 才会加载样例数据，
 *   且此时页面顶部必须常驻醒目的样例横幅。
 * --------------------------------------------------------------- */
'use strict';

var FB = (function () {

  /* 相对于 web/*.html 的路径 */
  var REAL_URL = '../results/leaderboard.json';
  var SAMPLE_URL = './sample_leaderboard.json';

  var EXPECTED_SCHEMA = 'forcebench-results-v1';

  /* ---------- 静态常量（仓库事实，非评测结果） ---------- */

  /* 任务的中文名与说明来自 results/leaderboard.json 的
     display_name_zh / summary_zh / varies_zh；
     下面这份仅在数据文件尚未生成时用于渲染"任务说明"页，
     内容取自仓库 README 的任务表，不包含任何评测数字。 */
  var TASK_FALLBACK = [
    {
      id: 'weight_sort',
      display_name_zh: '按重量分拣箱子',
      force_category: 'force_classification',
      summary_zh: '拿起外观完全相同的箱子，靠重量判断轻重，轻箱放黑色容器、重箱放蓝色容器。',
      varies_zh: '箱子质量 0.06 kg / 0.18 kg，外观不可区分',
      dataset_episodes: 200
    },
    {
      id: 'rattle_sort',
      display_name_zh: '按内部配重分拣包裹',
      force_category: 'force_classification',
      summary_zh: '分辨包裹内部配重是居中还是偏置（总质量固定 0.18 kg），据此分拣。',
      varies_zh: '质心偏移量；总质量固定',
      dataset_episodes: 200
    },
    {
      id: 'stove_turn_on',
      display_name_zh: '旋转灶台旋钮越过点火卡扣',
      force_category: 'force_classification',
      summary_zh: '旋转灶台旋钮，靠力矩变化判断是否已越过点火卡扣（detent）。',
      varies_zh: '非线性卡位力矩，峰值 0.12 N·m',
      dataset_episodes: 200
    },
    {
      id: 'faucet_turn',
      display_name_zh: '扳动龙头把手越过卡扣',
      force_category: 'force_classification',
      summary_zh: '扳动水龙头把手，同样只能靠卡位力矩判断是否到位。',
      varies_zh: '非线性卡位力矩，峰值 0.20 N·m（固定）',
      dataset_episodes: 200
    },
    {
      id: 'radmanso_drawer_sort_table',
      display_name_zh: '拉开抽屉并在阻力上升处停手',
      force_category: 'force_classification',
      summary_zh: '拉开抽屉整理桌面物品，需要在阻力上升时判断何时停止拉动。',
      varies_zh: '起拉阻力随机 5.5–9.0 N',
      dataset_episodes: 200
    },
    {
      id: 'peg_insert',
      display_name_zh: '销轴插孔',
      force_category: 'contact_rich',
      summary_zh: '将销轴插入孔中。当前成功判定只检查最终位姿，spec 中没有力/质量常量。',
      varies_zh: '插入对准与接触几何',
      dataset_episodes: 189
    },
    {
      id: 'gear_mesh',
      display_name_zh: '齿轮啮合装配',
      force_category: 'contact_rich',
      summary_zh: '把齿轮放到位并与另一齿轮啮合。质量仅为接触真实感而设，不参与判分。',
      varies_zh: '齿轮就位与啮合几何',
      dataset_episodes: 62
    },
    {
      id: 'nut_thread',
      display_name_zh: '螺母拧入螺栓',
      force_category: 'contact_rich',
      summary_zh: '将螺母拧上螺栓。当前判分不检查拧入过程中的力。',
      varies_zh: '螺纹啮合几何',
      dataset_episodes: 200
    }
  ];

  /* 模型列顺序。名单以数据文件的 models[] 为准，这里只决定排列先后；
     未列出的模型会排在后面而不会被丢掉。 */
  var MODEL_ORDER = ['pi05', 'pi05_force6', 'forcevla', 'forceflow'];

  var MODEL_FALLBACK = {
    pi05: { id: 'pi05', display_name_zh: 'π0.5', short_name: 'pi0.5', force_input: 'none' },
    pi05_force6: { id: 'pi05_force6', display_name_zh: 'π0.5 + 力6D', short_name: 'pi0.5+F6', force_input: 'wrench6_concat' },
    forcevla: { id: 'forcevla', display_name_zh: 'ForceVLA', short_name: 'ForceVLA', force_input: 'moe_fusion' }
  };

  var CATEGORY = {
    force_classification: {
      label: '力分类任务',
      glyph: '◆',
      desc: '决定"正确行为"的物体属性在视觉上完全不可观测，只能靠力/力矩推断。'
    },
    contact_rich: {
      label: '接触密集任务',
      glyph: '■',
      desc: '接触丰富，但当前成功判定只看最终位姿、不读取力信号。MANIFEST 里 "force-sensitive" 的措辞是前瞻性的，不代表现在检验了力感知。'
    }
  };

  var FORCE_INPUT_DESC = {
    none: '不输入任何力信号',
    wrench6_concat: '6 维力/力矩直接拼进 state 向量',
    moe_fusion: '架构级力融合（FVLMoE）'
  };

  var STATUS = {
    complete: { label: '已完成', glyph: '✓', cls: 'st-complete',
      desc: '完整 100 次 rollout 的有效结果。' },
    partial: { label: '部分评测', glyph: '◐', cls: 'st-partial',
      desc: '跑过但 rollout 数不足 100，置信区间相应更宽。' },
    missing: { label: '尚未评测', glyph: '○', cls: 'st-missing',
      desc: '这一格从未跑过评测。没有数字，不是 0 分。' },
    invalid: { label: '结果无效', glyph: '⚠', cls: 'st-invalid',
      desc: '跑完了但结果不可信（例如三个模型同时全 0，疑似评测链路故障），不作为分数使用。' }
  };

  /* ---------- 站外链接 ----------
     只登记确实存在的地址。尚未发布的东西（论文、数据集）不给链接，
     只给状态文字 —— 死链比没有链接更糟。 */
  var REPO = 'https://github.com/pkuhanxuan/ForceBench';
  var LINKS = {
    repo: REPO,
    docs: REPO + '/tree/main/docs',
    docsIndex: REPO + '/blob/main/docs/README.md',
    evalProtocol: REPO + '/blob/main/docs/eval_protocol.md',
    policyInterface: REPO + '/blob/main/docs/policy_interface.md',
    datasetDoc: REPO + '/blob/main/docs/dataset.md',
    reproduce: REPO + '/blob/main/docs/reproduce.md',
    resultsJson: REPO + '/blob/main/results/leaderboard.json',
    issues: REPO + '/issues',
    commits: REPO + '/commits/main',
    taskDoc: function (id) { return REPO + '/blob/main/docs/tasks/' + id + '.md'; },
    commit: function (hash) { return REPO + '/commit/' + hash; }
  };

  /* 尚未存在的产物 —— 页面上只显示状态，不生成链接 */
  var PENDING = {
    paper: { label: '论文', status: '准备中，尚未发表',
      detail: '论文尚未投稿/发表，因此本站不提供论文入口，也没有可引用的 arXiv 编号。' },
    dataset: { label: '数据集与 checkpoint', status: '尚未发布',
      detail: '专家数据与训练好的 checkpoint 目前都只在项目内部的机器上，尚未上传到任何公开托管（包括 HuggingFace）。' +
        '计划在论文投稿之后再开放下载。在那之前本站不提供下载链接 —— 指向一个还不存在的仓库只会浪费你的时间。' }
  };

  /* ---------- 更新时间线 ----------
     每一条都对应本仓库 git 历史里真实存在的一次提交，附 commit 短哈希可自行核对
     （git log --oneline，或点条目上的哈希跳到 GitHub）。
     本项目至今没有发布过任何 release/tag，所以这里不写"版本发布"。 */
  var NEWS = [
    { date: '2026-08-28', commit: null, kind: 'web',
      title: '站点补齐提交记录、多视角排序与更新时间线',
      detail: '参照成熟 benchmark 站点的结构，补上贡献者/评测日期、任务范围与排序切换、' +
        '文档入口和这条时间线；同时把置信区间、缺格标注与显著性检验原样保留。' },
    { date: '2026-08-27', commit: '8a1bd1b', kind: 'fix',
      title: '兼容旧版 HDF5 schema 标签，并把 episode 计数改成与磁盘一致',
      detail: '文档里的专家数据条数此前与实际落盘的数据对不上，已按磁盘实际情况更正。' },
    { date: '2026-08-27', commit: '367c68a', kind: 'results',
      title: 'results/leaderboard.json 正式落盘，并修正任务覆盖口径',
      detail: '在此之前 leaderboard 的权威数据文件并不在仓库里，站点无数可读。' },
    { date: '2026-08-27', commit: '9d4df40', kind: 'web',
      title: '上线纯静态中文站点与 leaderboard 页面',
      detail: '无构建步骤、无 npm、不依赖任何外部 CDN，所有数字运行时从数据文件读取。' },
    { date: '2026-08-27', commit: 'e9c52ed', kind: 'docs',
      title: '把「数据托管状态」与「下载脚本状态」在文档中拆开陈述',
      detail: '两件事此前被混为一谈，容易让人以为数据已经可以下载。' },
    { date: '2026-08-27', commit: '27b0365', kind: 'docs',
      title: '顶层文档改写为中文' },
    { date: '2026-08-27', commit: 'c9cd720', kind: 'docs',
      title: '新增面向使用者的中文指南、任务索引与 results 目录骨架' },
    { date: '2026-08-27', commit: '9db0c81', kind: 'results',
      title: '汇总 08-25 / 08-26 两批已核对的 baseline 评测结果',
      detail: '只收录环境间距为 6 m、且评测 arming 缺陷修复之后的运行；更早的结果全部作废。' },
    { date: '2026-08-27', commit: '4e103a4', kind: 'docs',
      title: '内部文档迁入 docs/developer/ 并翻译为中文' },
    { date: '2026-08-27', commit: 'cd70dd2', kind: 'repo',
      title: '全仓库更名：coribench → forcebench' },
    { date: '2026-08-26', commit: '9384626', kind: 'repo',
      title: '初始骨架：8 个力感知仿真任务' },
    { date: '2026-08-26', commit: '04fd445', kind: 'repo',
      title: '仓库创建' }
  ];

  var NEWS_KIND = {
    results: { label: '评测结果', glyph: '▣' },
    eval:    { label: '评测运行', glyph: '▣' },
    web:     { label: '站点', glyph: '◧' },
    docs:    { label: '文档', glyph: '◫' },
    fix:     { label: '修复', glyph: '◈' },
    repo:    { label: '仓库', glyph: '◇' }
  };

  /* ---------- 贡献者 ----------
     数据契约 forcebench-results-v1 目前没有 contributor 字段。
     没有就如实显示"项目自测"，不编造投稿人。
     将来 models[] 或 results[] 里出现 contributor / submitted_by 时会自动采用。 */
  var SELF_CONTRIBUTOR = 'ForceBench 团队';
  var SELF_CONTRIBUTOR_NOTE =
    '当前 leaderboard 上的全部结果都由项目自己评测，不是外部提交。' +
    '数据文件（forcebench-results-v1）目前没有 contributor 字段，' +
    '因此这一列统一显示为「' + SELF_CONTRIBUTOR + '」；' +
    '等数据文件里出现 contributor / submitted_by 字段后，这里会直接显示提交者。';

  function contributorOf(model, rows) {
    var c = model && (model.contributor || model.submitted_by);
    for (var i = 0; !c && i < rows.length; i++) {
      c = rows[i].contributor || rows[i].submitted_by;
    }
    if (!c) return { name: SELF_CONTRIBUTOR, url: null, external: false };
    if (typeof c === 'string') return { name: c, url: null, external: true };
    return { name: c.name || String(c), url: c.url || null, external: true };
  }

  /* ---------- 小工具 ---------- */

  function esc(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function pct(x, digits) {
    if (x == null || isNaN(x)) return '—';
    return (x * 100).toFixed(digits == null ? 1 : digits) + '%';
  }

  function pctNum(x, digits) {
    if (x == null || isNaN(x)) return '—';
    return (x * 100).toFixed(digits == null ? 1 : digits);
  }

  function statusOf(r) {
    if (!r || !r.status || !STATUS[r.status]) return STATUS.missing;
    return STATUS[r.status];
  }

  /* 误差棒 SVG：横轴 0 → axisMax，画 95% Clopper-Pearson 区间与点估计。
     纯装饰性补充，数值本身在旁边有文字，故 aria-hidden。 */
  function errorBarSVG(rate, lo, hi, axisMax) {
    var W = 132, H = 22, PAD = 3;
    var span = W - PAD * 2;
    var sc = function (v) { return PAD + Math.max(0, Math.min(1, v / axisMax)) * span; };
    var xlo = sc(lo), xhi = sc(hi), xp = sc(rate);
    var mid = 11;
    return '<svg class="errbar" width="' + W + '" height="' + H +
      '" viewBox="0 0 ' + W + ' ' + H + '" aria-hidden="true" focusable="false">' +
      '<line x1="' + PAD + '" y1="' + mid + '" x2="' + (W - PAD) + '" y2="' + mid +
        '" stroke="var(--border)" stroke-width="1"/>' +
      '<line x1="' + xlo + '" y1="' + mid + '" x2="' + xhi + '" y2="' + mid +
        '" stroke="var(--accent)" stroke-width="3" stroke-linecap="butt" opacity="0.42"/>' +
      '<line x1="' + xlo + '" y1="' + (mid - 5) + '" x2="' + xlo + '" y2="' + (mid + 5) +
        '" stroke="var(--accent)" stroke-width="1.6"/>' +
      '<line x1="' + xhi + '" y1="' + (mid - 5) + '" x2="' + xhi + '" y2="' + (mid + 5) +
        '" stroke="var(--accent)" stroke-width="1.6"/>' +
      '<circle cx="' + xp + '" cy="' + mid + '" r="3.6" fill="var(--accent)"/>' +
      '</svg>';
  }

  /* 供误差棒共用的横轴上界：取所有 ci95_high 的最大值向上取整到 10%，
     夹在 30%–100% 之间。全站所有误差棒共用同一刻度，可横向比较。 */
  function axisMaxFor(results) {
    var m = 0;
    results.forEach(function (r) {
      if (r.status === 'complete' || r.status === 'partial') {
        if (typeof r.ci95_high === 'number') m = Math.max(m, r.ci95_high);
      }
    });
    var v = Math.ceil((m + 0.001) * 10) / 10;
    if (!isFinite(v) || v <= 0) v = 0.3;
    return Math.max(0.3, Math.min(1, v));
  }

  /* ---------- 数据索引 ---------- */

  function index(data) {
    var byKey = {};
    (data.results || []).forEach(function (r) {
      byKey[r.task + '|' + r.model] = r;
    });
    var tasks = (data.tasks && data.tasks.length) ? data.tasks.slice() : TASK_FALLBACK.slice();
    var models = (data.models && data.models.length)
      ? data.models.slice()
      : MODEL_ORDER.map(function (id) { return MODEL_FALLBACK[id]; })
          .filter(function (m) { return !!m; });

    /* 模型按契约固定顺序排列 */
    models.sort(function (a, b) {
      var ia = MODEL_ORDER.indexOf(a.id), ib = MODEL_ORDER.indexOf(b.id);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
    });

    /* 任务按类别分组：力分类在前 */
    var order = TASK_FALLBACK.map(function (t) { return t.id; });
    tasks.sort(function (a, b) {
      var ca = a.force_category === 'force_classification' ? 0 : 1;
      var cb = b.force_category === 'force_classification' ? 0 : 1;
      if (ca !== cb) return ca - cb;
      var ia = order.indexOf(a.id), ib = order.indexOf(b.id);
      return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib);
    });

    return {
      raw: data,
      tasks: tasks,
      models: models,
      get: function (taskId, modelId) {
        return byKey[taskId + '|' + modelId] ||
          { task: taskId, model: modelId, status: 'missing', successes: null, success_rate: null, ci95_low: null, ci95_high: null };
      },
      counts: function () {
        var c = { complete: 0, partial: 0, missing: 0, invalid: 0, total: 0 };
        tasks.forEach(function (t) {
          models.forEach(function (m) {
            var r = byKey[t.id + '|' + m.id];
            var s = (r && STATUS[r.status]) ? r.status : 'missing';
            c[s]++; c.total++;
          });
        });
        return c;
      },
      allResults: function () {
        var out = [];
        tasks.forEach(function (t) {
          models.forEach(function (m) { out.push(byKey[t.id + '|' + m.id] || { status: 'missing' }); });
        });
        return out;
      },
      /* head_to_head：逐任务的两模型显著性检验，由数据生成端算好写入。
         前端只展示，不自己做统计。 */
      h2h: h2h(data)
    };
  }

  function h2h(data) {
    var hh = data.head_to_head;
    if (!hh || !hh.comparisons || !hh.comparisons.length) return null;
    var by = {};
    hh.comparisons.forEach(function (c) { by[c.task] = c; });
    return {
      meta: hh,
      by: by,
      list: hh.comparisons,
      baseline: hh.comparisons[0].baseline,
      challenger: hh.comparisons[0].challenger
    };
  }

  /* ---------- 提交记录 ----------
     一个模型 = 一次"提交"。贡献者与评测日期都从数据里取，取不到就如实说明缺。
     这里刻意不给排名：本站不做跨任务平均分，理由见 leaderboard 页面。 */
  function submissions(idx) {
    return idx.models.map(function (m) {
      var rows = [], done = 0, dates = [];
      idx.tasks.forEach(function (t) {
        var r = idx.get(t.id, m.id);
        rows.push(r);
        if (r.status === 'complete' || r.status === 'partial') done++;
        if (r.evaluated_at) dates.push(String(r.evaluated_at).slice(0, 10));
      });
      dates.sort();
      return {
        model: m,
        rows: rows,
        done: done,
        total: idx.tasks.length,
        firstEval: dates.length ? dates[0] : null,
        lastEval: dates.length ? dates[dates.length - 1] : null,
        contributor: contributorOf(m, rows)
      };
    });
  }

  /* 从 results[] 的 evaluated_at 里还原"哪天跑了哪些格子"，用于时间线。
     完全由数据驱动 —— 数据文件里没有的日期，时间线上也不会出现。 */
  function evalMilestones(idx) {
    var by = {};
    (idx.raw.results || []).forEach(function (r) {
      if (!r.evaluated_at) return;
      if (r.status !== 'complete' && r.status !== 'partial') return;
      var d = String(r.evaluated_at).slice(0, 10);
      if (!by[d]) by[d] = { date: d, cells: [], tasks: {}, models: {} };
      by[d].cells.push(r);
      by[d].tasks[r.task] = 1;
      by[d].models[r.model] = 1;
    });
    return Object.keys(by).sort().reverse().map(function (d) {
      var e = by[d];
      return {
        date: d,
        cells: e.cells.length,
        tasks: Object.keys(e.tasks),
        models: Object.keys(e.models)
      };
    });
  }

  /* ---------- 查询串状态 ----------
     视角/筛选写进地址栏，方便把某个视角直接发给别人。
     ?data=sample 等既有参数会原样保留。 */
  function qsGet() {
    var out = {};
    var s = location.search.replace(/^\?/, '');
    if (!s) return out;
    s.split('&').forEach(function (kv) {
      if (!kv) return;
      var i = kv.indexOf('=');
      var k = i < 0 ? kv : kv.slice(0, i);
      var v = i < 0 ? '' : kv.slice(i + 1);
      try { out[decodeURIComponent(k)] = decodeURIComponent(v.replace(/\+/g, ' ')); }
      catch (e) { out[k] = v; }
    });
    return out;
  }

  function qsSet(patch) {
    var p = qsGet();
    Object.keys(patch).forEach(function (k) {
      if (patch[k] == null || patch[k] === '') delete p[k];
      else p[k] = patch[k];
    });
    var parts = Object.keys(p).map(function (k) {
      return encodeURIComponent(k) + '=' + encodeURIComponent(p[k]);
    });
    var url = location.pathname + (parts.length ? '?' + parts.join('&') : '') + location.hash;
    if (history && history.replaceState) history.replaceState(null, '', url);
  }

  /* 显著性分级 —— 颜色之外同时给字形和文字，不靠颜色单独承载信息 */
  function significance(c) {
    if (!c) return null;
    if (c.significant_after_bonferroni) {
      return { key: 'strong', glyph: '★', cls: 'sig-strong',
        label: '多重比较校正后仍显著',
        desc: '通过 Bonferroni 校正，是这批对比中唯一站得住的差异级别。' };
    }
    if (c.significant_at_005) {
      return { key: 'weak', glyph: '△', cls: 'sig-weak',
        label: 'p < 0.05，但校正后不显著',
        desc: '单看这一个任务达到 p < 0.05，但在同时做多个任务对比时，这一水平不足以排除偶然。' };
    }
    return { key: 'ns', glyph: '○', cls: 'sig-ns',
      label: '差异不显著',
      desc: '现有样本量下，这个差距无法与随机波动区分开。' };
  }

  function fmtP(p) {
    if (p == null || isNaN(p)) return '—';
    if (p < 0.0001) return 'p < 0.0001';
    return 'p = ' + Number(p).toFixed(4);
  }

  function fmtDiff(pp) {
    if (pp == null || isNaN(pp)) return '—';
    var s = (pp > 0 ? '+' : (pp < 0 ? '−' : '')) + Math.abs(pp).toFixed(1);
    return s + ' pp';
  }

  /* ---------- 加载 ---------- */

  function wantsSample() {
    return /(^|[?&])data=sample($|&)/.test(location.search);
  }

  /* load(onReady, onEmpty)
   *   onReady(idx, meta)  —— meta = { sample: bool, url: string, warning: string|null }
   *   onEmpty(info)       —— info = { url, reason }
   */
  function load(onReady, onEmpty) {
    var sample = wantsSample();
    var url = sample ? SAMPLE_URL : REAL_URL;

    fetch(url, { cache: 'no-cache' })
      .then(function (resp) {
        if (!resp.ok) throw new Error('HTTP ' + resp.status + ' ' + resp.statusText);
        return resp.json();
      })
      .then(function (data) {
        var warning = null;
        if (data.schema_version && data.schema_version !== EXPECTED_SCHEMA) {
          warning = '数据文件的 schema_version 是 "' + data.schema_version +
            '"，本站按 "' + EXPECTED_SCHEMA + '" 渲染，字段可能对不上。';
        }
        if (!sample && data.is_sample === true) {
          /* 真实路径上出现了样例数据 —— 必须如实标注，不得当真数据展示 */
          sample = true;
          warning = '真实数据路径下的文件自称样例数据（is_sample: true）。';
        }
        onReady(index(data), { sample: sample, url: url, warning: warning });
      })
      .catch(function (err) {
        onEmpty({ url: url, reason: String(err && err.message ? err.message : err), sample: sample });
      });
  }

  /* ---------- 通用 UI 片段 ---------- */

  function sampleBannerHTML() {
    return '<div class="sample-banner" role="alert">' +
      '<strong>⚠ 当前显示的是「样例数据」，不是真实评测结果。</strong>' +
      '<p>本页正在读取 <code>web/sample_leaderboard.json</code>，其中的所有数字都是为了开发和演示页面而虚构的，' +
      '不代表任何模型在 ForceBench 上的实际表现，任何情况下都不得引用。' +
      '去掉网址里的 <code>?data=sample</code> 即可切回真实数据源。</p></div>';
  }

  function emptyStateHTML(info) {
    return '<div class="datastate" role="status">' +
      '<span class="icon" aria-hidden="true">○</span>' +
      '<h2>评测数据尚未生成</h2>' +
      '<p class="muted">本站不内置任何评测数字。真实结果来自仓库中的 ' +
      '<code>results/leaderboard.json</code>，该文件目前读取不到，' +
      '因此这里不显示任何成绩 —— 而不是回退到样例数据或补零。</p>' +
      '<p class="muted small">如果你是在本地预览：请确认已从仓库根目录启动 HTTP 服务，' +
      '并且 <code>results/leaderboard.json</code> 已经生成。</p>' +
      '<div class="actions">' +
      '<a class="btn secondary" href="?data=sample">仅查看页面样式（载入样例数据）</a>' +
      '</div>' +
      '<p class="detail">请求：' + esc(info.url) + ' — ' + esc(info.reason) + '</p>' +
      '</div>';
  }

  /* 高亮当前导航项 */
  function markNav() {
    var page = document.body.getAttribute('data-page');
    var links = document.querySelectorAll('.site-nav a[data-nav]');
    for (var i = 0; i < links.length; i++) {
      if (links[i].getAttribute('data-nav') === page) {
        links[i].setAttribute('aria-current', 'page');
      }
    }
  }

  return {
    REAL_URL: REAL_URL,
    SAMPLE_URL: SAMPLE_URL,
    TASK_FALLBACK: TASK_FALLBACK,
    MODEL_ORDER: MODEL_ORDER,
    MODEL_FALLBACK: MODEL_FALLBACK,
    CATEGORY: CATEGORY,
    FORCE_INPUT_DESC: FORCE_INPUT_DESC,
    STATUS: STATUS,
    LINKS: LINKS,
    PENDING: PENDING,
    NEWS: NEWS,
    NEWS_KIND: NEWS_KIND,
    SELF_CONTRIBUTOR: SELF_CONTRIBUTOR,
    SELF_CONTRIBUTOR_NOTE: SELF_CONTRIBUTOR_NOTE,
    contributorOf: contributorOf,
    submissions: submissions,
    evalMilestones: evalMilestones,
    qsGet: qsGet,
    qsSet: qsSet,
    esc: esc,
    pct: pct,
    pctNum: pctNum,
    statusOf: statusOf,
    significance: significance,
    fmtP: fmtP,
    fmtDiff: fmtDiff,
    errorBarSVG: errorBarSVG,
    axisMaxFor: axisMaxFor,
    index: index,
    load: load,
    wantsSample: wantsSample,
    sampleBannerHTML: sampleBannerHTML,
    emptyStateHTML: emptyStateHTML,
    markNav: markNav
  };
})();

document.addEventListener('DOMContentLoaded', FB.markNav);

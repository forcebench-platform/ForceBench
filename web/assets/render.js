/* ForceBench 站点 —— 各页面渲染逻辑
 * 依赖 data.js（全局对象 FB）。按 <body data-page="..."> 分发。 */
'use strict';

(function () {

  var E = FB.esc;

  /* ============ 共享片段 ============ */

  function catBadge(cat) {
    var c = FB.CATEGORY[cat];
    if (!c) return '';
    return '<span class="cat ' + E(cat) + '"><span class="glyph" aria-hidden="true">' +
      c.glyph + '</span>' + E(c.label) + '</span>';
  }

  function taskCard(t, opts) {
    opts = opts || {};
    var c = FB.CATEGORY[t.force_category] || {};
    var h = '<article class="task-card">' +
      '<h3>' + E(t.display_name_zh || t.id) + ' ' + catBadge(t.force_category) + '</h3>' +
      '<span class="tid">' + E(t.id) + '</span>' +
      '<p>' + E(t.summary_zh || '') + '</p>';
    if (opts.detail) {
      h += '<dl>';
      if (t.varies_zh) h += '<dt>物理上变化的量</dt><dd>' + E(t.varies_zh) + '</dd>';
      if (t.dataset_episodes != null) {
        /* 0 条不能只显示一个"0" —— 没有专家数据是这个任务当前最重要的事实 */
        h += '<dt>训练集 episode 数</dt><dd>' +
          (Number(t.dataset_episodes) > 0
            ? E(t.dataset_episodes)
            : '<span class="flag f-missing"><span class="glyph" aria-hidden="true">○</span>' +
              '当前没有专家数据</span>') +
          '</dd>';
      }
      h += '<dt>为什么归到「' + E(c.label || '') + '」</dt><dd>' + E(c.desc || '') + '</dd>';
      h += '</dl>';
      h += '<p class="card-links"><a href="' + E(FB.LINKS.taskDoc(t.id)) + '">任务卡片文档 ↗</a></p>';
    }
    h += '</article>';
    return h;
  }

  /* 覆盖率面板：显式回答"整张网格里做完了几格"。
     格数、行列数一律由数据算出，页面里不写死任何数字。 */
  function coveragePanel(idx) {
    var c = idx.counts();
    var done = c.complete;
    var pctDone = c.total ? Math.round(done / c.total * 100) : 0;
    var nT = idx.tasks.length, nM = idx.models.length;
    /* 列数由数据决定；具体列宽仍交给 CSS（含窄屏断点） */
    var gridStyle = '--mcols:' + nM;

    var legend = '<ul class="status-legend">';
    ['complete', 'partial', 'invalid', 'missing'].forEach(function (k) {
      var s = FB.STATUS[k];
      legend += '<li><span class="swatch ' + s.cls + '" aria-hidden="true">' + s.glyph + '</span>' +
        '<span>' + E(s.label) + '</span><span class="count">' + c[k] + ' 格</span></li>';
    });
    legend += '</ul>';

    /* 状态矩阵：任务 × 模型 */
    var mtx = '<div class="matrix" role="group" aria-label="评测覆盖矩阵：' + nT + ' 个任务 × ' + nM + ' 个模型">';
    mtx += '<div class="matrix-row matrix-head" style="' + gridStyle + '"><span class="rowlab">任务 \\ 模型</span>';
    idx.models.forEach(function (m) {
      mtx += '<span class="cell headcell">' + E(m.short_name || m.id) + '</span>';
    });
    mtx += '</div>';
    idx.tasks.forEach(function (t) {
      mtx += '<div class="matrix-row" style="' + gridStyle + '">' +
        '<span class="rowlab" title="' + E(t.id) + '">' + E(t.id) + '</span>';
      idx.models.forEach(function (m) {
        var r = idx.get(t.id, m.id);
        var s = FB.statusOf(r);
        mtx += '<span class="cell ' + s.cls + '" title="' + E(t.id + ' × ' + m.id + '：' + s.label) + '">' +
          '<span aria-hidden="true">' + s.glyph + '</span>' +
          '<span class="visually-hidden">' + E(t.id + ' 搭配 ' + m.id + '：' + s.label) + '</span></span>';
      });
      mtx += '</div>';
    });
    mtx += '</div>';

    return '<section class="coverage" aria-labelledby="cov-h">' +
      '<div>' +
      '<h2 id="cov-h" style="margin:0 0 6px;font-size:1.05rem">评测覆盖率</h2>' +
      '<p class="coverage-headline">' + done + ' <span class="of">/ ' + c.total + ' 格已完成</span></p>' +
      '<div class="coverage-bar" role="img" aria-label="' + c.total + ' 格中 ' + done +
      ' 格已完成，约 ' + pctDone + '%">' +
      '<span style="width:' + pctDone + '%"></span></div>' +
      legend +
      '<p class="small muted" style="margin-top:12px">' +
      '共 ' + nT + ' 个任务 × ' + nM + ' 个模型 = ' + c.total +
      ' 个 (任务, 模型) 组合。空缺不是 0 分，是没跑。' +
      '下方表格的筛选只影响显示范围，这个面板始终按完整网格统计。</p>' +
      '</div>' +
      '<div>' + mtx + '</div>' +
      '</section>';
  }

  /* ============ 提交记录（贡献者 / 评测日期） ============ */

  function submissionsSection(idx) {
    var subs = FB.submissions(idx);
    var anyExternal = subs.some(function (s) { return s.contributor.external; });

    var rows = subs.map(function (s) {
      var m = s.model;
      var fi = m.force_input || '';
      var cname = s.contributor.external
        ? (s.contributor.url
            ? '<a href="' + E(s.contributor.url) + '">' + E(s.contributor.name) + '</a>'
            : E(s.contributor.name))
        : E(s.contributor.name) + ' <span class="tagline">项目自测</span>';

      var when;
      if (s.lastEval) {
        when = '<span class="mono">' + E(s.lastEval) + '</span>';
        if (s.firstEval && s.firstEval !== s.lastEval) {
          when += '<span class="subline">最早 ' + E(s.firstEval) + '</span>';
        }
      } else {
        when = '<span class="flag f-missing"><span class="glyph" aria-hidden="true">○</span>尚无评测</span>';
      }

      var covCls = s.done > 0 ? '' : ' nodata';
      var cov = '<div class="res' + covCls + '" style="min-width:0">' +
        (s.done > 0
          ? '<span class="rate" style="font-size:1.05rem">' + s.done + ' / ' + s.total + '</span>' +
            '<span class="nk">个任务有有效结果</span>'
          : '<span class="flag f-missing"><span class="glyph" aria-hidden="true">○</span>0 / ' + s.total + '</span>' +
            '<span class="why">整列从未跑过，不是全部 0%。</span>') +
        '</div>';

      return '<tr>' +
        '<th scope="row">' + E(m.display_name_zh || m.id) +
          '<span class="tid">' + E(m.id) + '</span></th>' +
        '<td><span class="finput-pill" title="' + E(FB.FORCE_INPUT_DESC[fi] || '') + '">' +
          E(fi || '未标注') + '</span>' +
          '<span class="subline">' + E(FB.FORCE_INPUT_DESC[fi] || '数据文件未标注该模型的力输入方式') + '</span></td>' +
        '<td>' + cname + '</td>' +
        '<td>' + when + '</td>' +
        '<td>' + cov + '</td>' +
        '</tr>';
    }).join('');

    var h = '<h2 id="subs-h">提交记录</h2>';
    h += '<p class="muted">每个模型是一次「提交」。贡献者与评测日期都取自数据文件；' +
      '成熟 benchmark 的榜单通常还会给一个总排名，' +
      '<b>本站刻意不给</b> —— 覆盖还不完整，任务之间的难度差距又极大，' +
      '跨任务平均会把这两件事一起抹平。</p>';
    h += '<div class="table-scroll" tabindex="0" role="region" aria-labelledby="subs-h">' +
      '<table class="lb subs-table">' +
      '<caption>按力信号进入策略的程度排列，不是成绩排名。' +
      '「已完成任务数」统计的是该模型在多少个任务上拿到了有效结果。</caption>' +
      '<thead><tr>' +
      '<th scope="col">模型</th>' +
      '<th scope="col">力信号接入方式<span class="mid">force_input</span></th>' +
      '<th scope="col">贡献者</th>' +
      '<th scope="col">最近评测日期<span class="mid">evaluated_at</span></th>' +
      '<th scope="col">已完成任务数</th>' +
      '</tr></thead><tbody>' + rows + '</tbody></table></div>';

    if (!anyExternal) {
      h += '<p class="small muted">' + E(FB.SELF_CONTRIBUTOR_NOTE) + '</p>';
    }

    h += '<div class="note info"><span class="note-title">想把自己的结果放上来？</span>' +
      '<p>本站的榜单结构已经为外部提交留好了位置（贡献者、评测日期、逐任务成绩），' +
      '但<b>正式的提交流程还没有定下来</b> —— 目前既没有提交模板，也没有复核规则。' +
      '在那之前，如果你跑了 ForceBench 并希望结果被收录，请先到仓库开一个 issue，' +
      '附上按 <code>forcebench-results-v1</code> 格式组织的结果和评测命令，' +
      '我们会人工核对。请注意评测协议必须一致：固定 seed、' +
      '环境间距 6 m、以及与本表相同的步数上限。</p>' +
      '<p><a class="btn secondary" href="' + E(FB.LINKS.issues) + '">到 GitHub 提 issue</a> ' +
      '<a class="btn secondary" href="' + E(FB.LINKS.evalProtocol) + '">评测协议文档</a></p>' +
      '</div>';

    return h;
  }

  /* ============ 首页 ============ */

  function modelCard(idx, m) {
    /* 该模型是否已有任何有效结果 —— 决定要不要打「尚未评测」标记 */
    var any = false;
    idx.tasks.forEach(function (t) {
      var r = idx.get(t.id, m.id);
      if (r.status === 'complete' || r.status === 'partial') any = true;
    });
    var fi = m.force_input || '';
    return '<div class="card">' +
      '<h3 style="margin-top:0">' + E(m.display_name_zh || m.id) +
      (any ? '' : ' <span class="flag f-missing" style="font-size:.7rem;vertical-align:middle">' +
        '<span class="glyph" aria-hidden="true">○</span>尚未评测</span>') + '</h3>' +
      '<p class="small mono muted">' + E(m.id) + ' · force_input: ' + E(fi || '未标注') + '</p>' +
      '<p class="small">' + E(m.description_zh || FB.FORCE_INPUT_DESC[fi] || '数据文件未提供该模型的中文说明。') + '</p>' +
      '</div>';
  }

  /* ============ 更新时间线 ============
     两个来源，都不是手写的宣传文案：
       1) FB.NEWS —— 本仓库真实的 git 提交，每条附 commit 短哈希；
       2) 数据文件 results[] 里的 evaluated_at —— 哪天真的跑了哪些格子。
     两者合并后按日期倒序。没有第三种来源，也不写"版本发布"（本项目还没发过 release）。 */

  function newsItemHTML(it) {
    var k = FB.NEWS_KIND[it.kind] || { label: it.kind || '更新', glyph: '·' };
    var h = '<li class="news-item">' +
      '<div class="news-when"><time datetime="' + E(it.date) + '">' + E(it.date) + '</time></div>' +
      '<div class="news-body">' +
      '<span class="news-kind k-' + E(it.kind || 'repo') + '">' +
      '<span class="glyph" aria-hidden="true">' + k.glyph + '</span>' + E(k.label) + '</span>' +
      '<p class="news-title">' + E(it.title) + '</p>';
    if (it.detail) h += '<p class="news-detail">' + E(it.detail) + '</p>';
    if (it.commit) {
      h += '<p class="news-src">提交 <a class="mono" href="' + E(FB.LINKS.commit(it.commit)) + '">' +
        E(it.commit) + '</a></p>';
    } else if (it.srcNote) {
      h += '<p class="news-src">' + it.srcNote + '</p>';
    }
    h += '</div></li>';
    return h;
  }

  function newsHTML(idx) {
    var items = FB.NEWS.slice();

    /* 评测里程碑：直接从数据文件的 evaluated_at 还原，页面不写死日期 */
    if (idx) {
      FB.evalMilestones(idx).forEach(function (e) {
        var names = e.tasks.map(function (id) {
          var t = null;
          idx.tasks.forEach(function (x) { if (x.id === id) t = x; });
          return t ? (t.display_name_zh || t.id) : id;
        });
        var mnames = e.models.map(function (id) { return modelName(idx, id); });
        items.push({
          date: e.date,
          kind: 'eval',
          commit: null,
          title: '评测运行：' + e.cells + ' 格结果（' + mnames.join(' / ') + '）',
          detail: '当天取得有效结果的任务：' + names.join('、') + '。',
          srcNote: '来源：<code>results/leaderboard.json</code> 的 <code>evaluated_at</code> 字段'
        });
      });
    }

    items.sort(function (a, b) {
      if (a.date === b.date) return 0;
      return a.date < b.date ? 1 : -1;
    });

    return '<ol class="news">' + items.map(newsItemHTML).join('') + '</ol>' +
      '<p class="small muted">时间线上的每一条要么对应仓库里一次真实的 git 提交（可点哈希核对），' +
      '要么直接由 <code>results/leaderboard.json</code> 的 <code>evaluated_at</code> 还原。' +
      '本项目至今没有发布过任何 release 或版本标签，所以这里不会出现"发布 vX.Y"这样的条目。' +
      '完整提交历史见 <a href="' + FB.LINKS.commits + '">GitHub commits</a>。</p>';
  }

  function paintNews(idx) {
    var slot = document.getElementById('home-news');
    if (slot) slot.innerHTML = newsHTML(idx);
  }

  function renderHome() {
    var cov = document.getElementById('home-coverage');
    var cards = document.getElementById('home-tasks');
    var mods = document.getElementById('home-models');

    if (cards) {
      cards.innerHTML = FB.TASK_FALLBACK.map(function (t) { return taskCard(t, { detail: false }); }).join('');
    }
    /* 数据文件读不到时，时间线里的 git 部分照常显示，只是少了评测里程碑 */
    paintNews(null);

    FB.load(function (idx, meta) {
      if (meta.sample) {
        document.getElementById('sample-slot').innerHTML = FB.sampleBannerHTML();
      }
      paintNews(idx);
      if (cov) cov.innerHTML = coveragePanel(idx);
      if (cards) {
        cards.innerHTML = idx.tasks.map(function (t) { return taskCard(t, { detail: false }); }).join('');
      }
      if (mods) {
        mods.innerHTML = idx.models.map(function (m) { return modelCard(idx, m); }).join('');
      }
    }, function (info) {
      if (mods) {
        mods.innerHTML = '<p class="muted small">模型列表来自 <code>results/leaderboard.json</code> 的 ' +
          '<code>models[]</code> 字段，该文件当前读取不到，因此这里不列出任何模型。</p>';
      }
      if (cov) {
        cov.innerHTML = '<div class="datastate" role="status">' +
          '<span class="icon" aria-hidden="true">○</span>' +
          '<h2 style="font-size:1.1rem;margin:0 0 8px">评测数据尚未生成</h2>' +
          '<p class="muted small" style="margin:0">仓库中的 <code>results/leaderboard.json</code> 还读不到，' +
          '所以本页不显示任何覆盖率或成绩数字。</p>' +
          '<p class="detail">' + E(info.url) + ' — ' + E(info.reason) + '</p></div>';
      }
    });
  }

  /* ============ Leaderboard ============ */

  function resultCell(r, axisMax, footnotes) {
    var s = FB.statusOf(r);

    if (r.status === 'complete' || r.status === 'partial') {
      var n = r.num_rollouts != null ? r.num_rollouts : '?';
      var h = '<div class="res">';
      if (r.status === 'partial') {
        h += '<span class="flag f-partial"><span class="glyph" aria-hidden="true">◐</span>部分评测</span>';
      }
      h += '<span class="rate">' + FB.pct(r.success_rate) + '</span>';
      if (r.ci95_low != null && r.ci95_high != null) {
        h += '<span class="ci">95% CI ' + FB.pctNum(r.ci95_low) + '–' + FB.pctNum(r.ci95_high) + '%</span>';
        h += FB.errorBarSVG(r.success_rate, r.ci95_low, r.ci95_high, axisMax);
      } else {
        h += '<span class="ci">95% CI 缺失（数据文件未提供）</span>';
      }
      h += '<span class="nk">' + (r.successes != null ? r.successes : '?') + ' / ' + n + ' 次成功</span>';
      h += '</div>';
      return h;
    }

    if (r.status === 'invalid') {
      var mark = '';
      if (r.notes_zh) {
        footnotes.push({ key: r.task + ' × ' + r.model, text: r.notes_zh });
        mark = '<sup class="fn-ref">[' + footnotes.length + ']</sup>';
      }
      var observed = '';
      if (r.successes != null && r.num_rollouts != null) {
        observed = '<span class="nk">跑出 <span class="struck">' + r.successes + ' / ' + r.num_rollouts +
          '</span>，不作为分数采用</span>';
      }
      return '<div class="res nodata">' +
        '<span class="flag f-invalid" title="' + E(s.desc) + '">' +
        '<span class="glyph" aria-hidden="true">⚠</span>结果无效</span>' + mark +
        observed +
        '<span class="why">评测跑完了，但结果不可信，不等同于低分。</span>' +
        '</div>';
    }

    /* missing */
    return '<div class="res nodata">' +
      '<span class="flag f-missing" title="' + E(s.desc) + '">' +
      '<span class="glyph" aria-hidden="true">○</span>尚未评测</span>' +
      '<span class="why">这一格从未跑过，没有数字 —— 不是 0%。</span>' +
      '</div>';
  }

  function metaStrip(data, meta) {
    var p = data.eval_protocol || {};
    var bits = [];
    if (data.benchmark_version) bits.push('<span>benchmark 版本：<b>' + E(data.benchmark_version) + '</b></span>');
    if (data.generated_at) bits.push('<span>数据生成时间：<b>' + E(String(data.generated_at).replace('T', ' ').slice(0, 16)) + '</b></span>');
    if (p.num_rollouts) bits.push('<span>每格 rollout：<b>' + E(p.num_rollouts) + ' 次</b></span>');
    if (p.seed_start != null && p.seed_end != null) bits.push('<span>seed：<b>' + E(p.seed_start) + '–' + E(p.seed_end) + '</b></span>');
    if (p.control_rate_hz) bits.push('<span>控制频率：<b>' + E(p.control_rate_hz) + ' Hz</b></span>');
    bits.push('<span>数据源：<b class="mono">' + E(meta.url) + '</b></span>');
    return '<div class="meta-strip">' + bits.join('') + '</div>' +
      (p.notes_zh ? '<p class="small muted" style="margin:-14px 0 22px">' + E(p.notes_zh) + '</p>' : '');
  }

  /* ---- 多视角：任务范围 × 排序方式 × 模型列 ----
     全部数据驱动。将来数据文件里多出一个指标，只要在 SORTS 里加一条，
     表格结构本身不用动。 */

  var SCOPES = [
    { id: 'all', label: '全部任务',
      match: function () { return true; } },
    { id: 'force_classification', label: '只看力分类任务',
      match: function (t) { return t.force_category === 'force_classification'; } },
    { id: 'contact_rich', label: '只看接触密集任务',
      match: function (t) { return t.force_category === 'contact_rich'; } }
  ];

  /* 每个排序视角：group 表示是否保留"按类别分组"的表头行；
     needsRef 表示这个排序要挑一个参照模型。 */
  var SORTS = [
    { id: 'category', label: '按任务类别分组', group: true, needsRef: false,
      note: '力分类任务在前，接触密集任务在后 —— 这两类任务的性质不同，默认不混排。' },
    { id: 'name', label: '按任务名称（字母序）', group: false, needsRef: false,
      note: '纯按 task id 排列，方便和仓库里的目录对照。' },
    { id: 'coverage', label: '按已完成模型数（多→少）', group: false, needsRef: false,
      note: '先看哪些任务的横向对比最完整 —— 只完成一两格的任务，横向比较意义有限。' },
    { id: 'rate_desc', label: '按成功率（高→低）', group: false, needsRef: true,
      note: '按所选参照模型的成功率点估计排序。请记住排序只用了点估计，' +
        '而相邻两行的置信区间常常大幅重叠。' },
    { id: 'rate_asc', label: '按成功率（低→高）', group: false, needsRef: true,
      note: '同上，反向排列 —— 用来快速找出所有模型都做不好的任务。' }
  ];

  var COLMODES = [
    { id: 'all', label: '显示全部模型列（含尚未评测）' },
    { id: 'evaluated', label: '只显示已有结果的模型列' }
  ];

  function pickById(list, id, dflt) {
    for (var i = 0; i < list.length; i++) if (list[i].id === id) return list[i];
    return dflt || list[0];
  }

  function refCandidates(idx) {
    return idx.models.filter(function (m) {
      return idx.tasks.some(function (t) {
        var r = idx.get(t.id, m.id);
        return r.status === 'complete' || r.status === 'partial';
      });
    });
  }

  function viewTasks(idx, view) {
    var scope = pickById(SCOPES, view.scope);
    var sort = pickById(SORTS, view.sort);
    var tasks = idx.tasks.filter(scope.match);

    /* 「没跑过」永远不参与数值排序，一律沉底 —— 把缺格当 0 来排序会直接
       制造出"这个任务最难"的假象。 */
    function rateOf(t) {
      var r = idx.get(t.id, view.ref);
      if (r.status !== 'complete' && r.status !== 'partial') return null;
      return typeof r.success_rate === 'number' ? r.success_rate : null;
    }
    function doneCount(t) {
      var n = 0;
      idx.models.forEach(function (m) {
        var r = idx.get(t.id, m.id);
        if (r.status === 'complete' || r.status === 'partial') n++;
      });
      return n;
    }

    if (sort.id === 'name') {
      tasks.sort(function (a, b) { return a.id < b.id ? -1 : (a.id > b.id ? 1 : 0); });
    } else if (sort.id === 'coverage') {
      tasks.sort(function (a, b) {
        var d = doneCount(b) - doneCount(a);
        return d || (a.id < b.id ? -1 : 1);
      });
    } else if (sort.id === 'rate_desc' || sort.id === 'rate_asc') {
      var sign = sort.id === 'rate_desc' ? -1 : 1;
      tasks.sort(function (a, b) {
        var ra = rateOf(a), rb = rateOf(b);
        if (ra == null && rb == null) return a.id < b.id ? -1 : 1;
        if (ra == null) return 1;   /* 无数据一律沉底 */
        if (rb == null) return -1;
        if (ra === rb) return a.id < b.id ? -1 : 1;
        return sign * (ra > rb ? 1 : -1);
      });
    }
    /* sort.id === 'category' 时保持 index() 已经排好的顺序 */
    return tasks;
  }

  function viewModels(idx, view) {
    if (view.cols !== 'evaluated') return idx.models;
    var keep = refCandidates(idx);
    return keep.length ? keep : idx.models;
  }

  function renderTable(idx, axisMax, footnotes, view) {
    var hasH2H = !!idx.h2h;
    var tasks = viewTasks(idx, view);
    var models = viewModels(idx, view);
    var sort = pickById(SORTS, view.sort);
    var grouped = sort.group;
    var ncols = models.length + 1 + (hasH2H ? 1 : 0);

    if (!tasks.length) {
      return '<table class="lb"><caption>当前筛选下没有任务可显示。</caption></table>';
    }

    var h = '<table class="lb">';
    h += '<caption>成功率越高越好。每个数字后面的 95% CI 是 Clopper-Pearson 精确二项区间；' +
      'n = 100 时区间宽度接近 18 个百分点，请不要只看点估计。' +
      (hasH2H ? '最右一列给出该任务上两个模型差异的 p 值，' +
        '避免只凭并排的百分数过度解读。' : '') +
      '排序与筛选只改变显示顺序和范围，不改变任何数字。</caption>';

    h += '<thead><tr><th scope="col">任务</th>';
    models.forEach(function (m) {
      var fi = m.force_input || '';
      var isRef = sort.needsRef && m.id === view.ref;
      h += '<th scope="col"' + (isRef ? ' class="ref-col"' : '') + '>' +
        '<span class="mname">' + E(m.display_name_zh || m.id) +
        (isRef ? '<span class="ref-mark" title="当前排序的参照模型">排序参照</span>' : '') + '</span>' +
        '<span class="mid">' + E(m.id) + '</span>' +
        '<span class="finput" title="' + E(FB.FORCE_INPUT_DESC[fi] || '') + '">force_input: ' + E(fi) + '</span>' +
        '</th>';
    });
    if (hasH2H) {
      h += '<th scope="col" class="h2h-col">' +
        '<span class="mname">' + E(modelName(idx, idx.h2h.baseline)) + ' vs. ' +
        E(modelName(idx, idx.h2h.challenger)) + '</span>' +
        '<span class="mid">Fisher 精确检验（双侧）</span></th>';
    }
    h += '</tr></thead>';

    var lastCat = null;
    var opened = false;
    tasks.forEach(function (t) {
      if (grouped && t.force_category !== lastCat) {
        if (opened) h += '</tbody>';
        lastCat = t.force_category;
        var c = FB.CATEGORY[lastCat] || { label: lastCat, desc: '' };
        h += '<tbody><tr class="group-head"><th scope="rowgroup" colspan="' + ncols + '">' +
          '<span class="glabel">' + catBadge(lastCat) +
          '<span class="muted small" style="font-weight:400">' + E(c.desc) + '</span></span></th></tr>';
        opened = true;
      } else if (!opened) {
        h += '<tbody>';
        opened = true;
      }
      /* 打乱分组顺序时，类别标记必须跟到每一行上 ——
         力分类 / 接触密集的区分在任何视角下都不能消失。 */
      h += '<tr><th scope="row">' + E(t.display_name_zh || t.id) +
        (grouped ? '' : ' ' + catBadge(t.force_category)) +
        '<span class="tid">' + E(t.id) + '</span></th>';
      models.forEach(function (m) {
        h += '<td>' + resultCell(idx.get(t.id, m.id), axisMax, footnotes) + '</td>';
      });
      if (hasH2H) h += h2hCell(idx, t.id);
      h += '</tr>';
    });
    if (opened) h += '</tbody>';

    h += '</table>';
    return h;
  }

  /* 视角控制条。无 JS 时这段不会出现，表格仍按默认视角完整渲染。 */
  function controlsHTML(idx, view) {
    var refs = refCandidates(idx);
    var sort = pickById(SORTS, view.sort);

    function sel(id, label, options, current, hint) {
      var o = options.map(function (x) {
        return '<option value="' + E(x.id) + '"' + (x.id === current ? ' selected' : '') + '>' +
          E(x.label) + '</option>';
      }).join('');
      return '<div class="ctl">' +
        '<label for="' + id + '">' + E(label) + '</label>' +
        '<select id="' + id + '" data-ctl="' + E(id) + '">' + o + '</select>' +
        (hint ? '<span class="ctl-hint">' + hint + '</span>' : '') +
        '</div>';
    }

    var h = '<form class="lb-controls" id="lb-controls" aria-labelledby="ctl-h" ' +
      'onsubmit="return false">' +
      '<span class="ctl-title" id="ctl-h">看这张表的角度</span>';
    h += sel('ctl-scope', '任务范围', SCOPES, view.scope);
    h += sel('ctl-sort', '排序方式', SORTS, view.sort);
    if (refs.length > 1) {
      h += sel('ctl-ref', '排序参照模型',
        refs.map(function (m) { return { id: m.id, label: m.display_name_zh || m.id }; }),
        view.ref,
        sort.needsRef ? '' : '（仅在按成功率排序时生效）');
    }
    h += sel('ctl-cols', '模型列', COLMODES, view.cols);
    h += '<button type="button" class="btn secondary ctl-reset" data-ctl="reset">恢复默认视角</button>';
    h += '</form>';
    return h;
  }

  function viewSummaryHTML(idx, view) {
    var scope = pickById(SCOPES, view.scope);
    var sort = pickById(SORTS, view.sort);
    var tasks = viewTasks(idx, view);
    var models = viewModels(idx, view);

    var bits = '当前视角：<b>' + E(scope.label) + '</b>（' + tasks.length + ' / ' +
      idx.tasks.length + ' 个任务）· <b>' + E(sort.label) + '</b>';
    if (sort.needsRef) bits += '（参照 <b>' + E(modelName(idx, view.ref)) + '</b>）';
    bits += ' · 显示 ' + models.length + ' / ' + idx.models.length + ' 个模型列';

    var h = '<p class="view-summary">' + bits + '</p>';
    h += '<p class="small muted view-note">' + E(sort.note);
    if (sort.needsRef) {
      h += ' 标着「尚未评测」「结果无效」的任务在这个排序下<b>一律排在最后</b>，' +
        '因为它们没有数值可比 —— 这不代表它们成绩最低。';
    }
    if (view.cols === 'evaluated') {
      h += ' 当前隐藏了尚未评测的模型列；隐藏不等于这些模型不存在，' +
        '完整缺口见上方覆盖率面板。';
    }
    h += '</p>';
    return h;
  }

  /* ---- head_to_head：逐任务显著性检验 ---- */

  function modelName(idx, id) {
    var out = id;
    idx.models.forEach(function (m) { if (m.id === id) out = m.display_name_zh || m.id; });
    return out;
  }

  /* 小型分叉条：以 0 为中心，向左/右表示差值方向。装饰性，数值在旁边有文字。 */
  function diffBarSVG(pp, maxPP) {
    var W = 108, H = 18, cx = W / 2, half = (W / 2) - 4;
    var len = Math.max(1.5, Math.min(half, Math.abs(pp) / maxPP * half));
    var x = pp >= 0 ? cx : cx - len;
    return '<svg width="' + W + '" height="' + H + '" viewBox="0 0 ' + W + ' ' + H +
      '" aria-hidden="true" focusable="false" style="vertical-align:middle">' +
      '<rect x="' + x + '" y="4" width="' + len + '" height="10" rx="2" fill="var(--accent)" opacity="0.55"/>' +
      '<line x1="' + cx + '" y1="1" x2="' + cx + '" y2="' + (H - 1) +
        '" stroke="var(--border-strong)" stroke-width="1"/>' +
      '</svg>';
  }

  function sigBadge(sg) {
    if (!sg) return '';
    return '<span class="sig ' + sg.cls + '" title="' + E(sg.desc) + '">' +
      '<span class="glyph" aria-hidden="true">' + sg.glyph + '</span>' + E(sg.label) + '</span>';
  }

  /* 表格里每一行的对比单元格：并排展示两个成功率的地方，必须同时给出 p 值与显著性 */
  function h2hCell(idx, taskId) {
    if (!idx.h2h) return '';
    var c = idx.h2h.by[taskId];
    if (!c) {
      return '<td><div class="res nodata"><span class="flag f-missing">' +
        '<span class="glyph" aria-hidden="true">○</span>无对比</span>' +
        '<span class="why">该任务缺少可比较的两组有效结果。</span></div></td>';
    }
    var sg = FB.significance(c);
    var maxPP = idx._maxPP || 20;
    var dir = c.diff_pp > 0 ? modelName(idx, c.challenger) + ' 更高'
      : (c.diff_pp < 0 ? modelName(idx, c.baseline) + ' 更高' : '持平');
    return '<td><div class="res h2h">' +
      '<span class="diff">' + E(FB.fmtDiff(c.diff_pp)) + '</span>' +
      '<span class="nk">' + E(dir) + '</span>' +
      diffBarSVG(c.diff_pp, maxPP) +
      '<span class="pval">' + E(FB.fmtP(c.p_value)) + '</span>' +
      sigBadge(sg) +
      '</div></td>';
  }

  /* 页面下方的完整 head_to_head 板块 */
  function h2hSection(idx) {
    if (!idx.h2h) return '';
    var hh = idx.h2h.meta;
    var maxPP = idx._maxPP || 20;
    var baseName = modelName(idx, idx.h2h.baseline);
    var chalName = modelName(idx, idx.h2h.challenger);

    /* 方向只在「达到 p<0.05」的那几组里清点 —— 否则会把不显著的差异也算进方向里 */
    var nSig = 0, nBonf = 0, sigBase = 0, sigChal = 0;
    idx.h2h.list.forEach(function (c) {
      if (c.significant_after_bonferroni) nBonf++;
      if (!c.significant_at_005) return;
      nSig++;
      if (c.diff_pp < 0) sigBase++; else if (c.diff_pp > 0) sigChal++;
    });

    var rows = idx.h2h.list.map(function (c) {
      var sg = FB.significance(c);
      var t = null;
      idx.tasks.forEach(function (x) { if (x.id === c.task) t = x; });
      var dir = c.diff_pp > 0 ? chalName + ' 更高' : (c.diff_pp < 0 ? baseName + ' 更高' : '持平');
      return '<tr>' +
        '<th scope="row">' + E(t ? (t.display_name_zh || t.id) : c.task) +
        '<span class="tid">' + E(c.task) + '</span></th>' +
        '<td class="num">' + E(FB.pct(c.baseline_rate)) + '</td>' +
        '<td class="num">' + E(FB.pct(c.challenger_rate)) + '</td>' +
        '<td class="num"><span class="diff">' + E(FB.fmtDiff(c.diff_pp)) + '</span> ' +
          diffBarSVG(c.diff_pp, maxPP) + '<span class="visually-hidden">，' + E(dir) + '</span></td>' +
        '<td class="num">' + E(FB.fmtP(c.p_value)) + '</td>' +
        '<td>' + sigBadge(sg) + '</td>' +
        '</tr>';
    }).join('');

    var h = '<h2 id="h2h-h">' + E(baseName) + ' vs. ' + E(chalName) + '：逐任务显著性检验</h2>';

    var dirSentence = '';
    if (nSig > 0) {
      if (sigBase > 0 && sigChal > 0) {
        dirSentence = '而且达到显著的那 ' + nSig + ' 组里，方向<b>并不一致</b> —— ' +
          E(baseName) + ' 更高的有 ' + sigBase + ' 组，' + E(chalName) + ' 更高的有 ' + sigChal + ' 组。';
      } else {
        dirSentence = '达到显著的 ' + nSig + ' 组全部是 ' +
          E(sigChal > 0 ? chalName : baseName) + ' 更高。';
      }
    }

    h += '<div class="note caution"><span class="note-title">⚠ 为什么并排看两个百分数会骗人</span>' +
      '<p>这 ' + idx.h2h.list.length + ' 组对比里，达到 <code>p &lt; 0.05</code> 的只有 <b>' + nSig +
      ' 组</b>；在做完多重比较校正之后，只剩 <b>' + nBonf + ' 组</b>还站得住。' +
      dirSentence +
      '所以看到两个并排的百分数时，请先看同一行的 p 值和显著性标记再下判断。</p>';
    if (hh.notes_zh) h += '<p>' + E(hh.notes_zh) + '</p>';
    h += '</div>';

    h += '<p class="scroll-hint">表格可左右滑动。</p>';
    h += '<div class="table-scroll" tabindex="0" role="region" aria-labelledby="h2h-h">' +
      '<table class="lb h2h-table">' +
      '<caption>每一行都把两个模型的成功率、差值、p 值和显著性放在一起。' +
      '差值单位 pp（percentage point，百分点）。</caption>' +
      '<thead><tr>' +
      '<th scope="col">任务</th>' +
      '<th scope="col">' + E(baseName) + '<span class="mid">' + E(idx.h2h.baseline) + '</span></th>' +
      '<th scope="col">' + E(chalName) + '<span class="mid">' + E(idx.h2h.challenger) + '</span></th>' +
      '<th scope="col">差值</th>' +
      '<th scope="col">p 值</th>' +
      '<th scope="col">显著性</th>' +
      '</tr></thead><tbody>' + rows + '</tbody></table></div>';

    var meta = [];
    if (hh.method) meta.push('检验方法：<code>' + E(hh.method) + '</code>');
    if (hh.family_size != null) meta.push('对比族大小：<b>' + E(hh.family_size) + '</b>');
    if (hh.bonferroni_alpha != null) meta.push('Bonferroni 校正后阈值：<b>α = ' + E(hh.bonferroni_alpha) + '</b>');
    if (meta.length) h += '<p class="small muted">' + meta.join(' · ') + '</p>';

    h += '<ul class="legend"><li><span class="swatch sig-strong" aria-hidden="true">★</span>' +
      '<span><b>多重比较校正后仍显著</b> — 通过 Bonferroni 校正</span></li>' +
      '<li><span class="swatch sig-weak" aria-hidden="true">△</span>' +
      '<span><b>p &lt; 0.05，但校正后不显著</b> — 同时比较多个任务时不足以排除偶然</span></li>' +
      '<li><span class="swatch sig-ns" aria-hidden="true">○</span>' +
      '<span><b>差异不显著</b> — 现有样本量下无法与随机波动区分</span></li></ul>';

    h += '<p class="small muted">所有 p 值和显著性判定都由数据生成端算好写入 ' +
      '<code>results/leaderboard.json</code> 的 <code>head_to_head</code> 字段，前端只负责展示，不做统计计算。' +
      '本站不由此给出「力有帮助」或「力没用」的总体结论。</p>';

    return h;
  }

  function legendHTML(axisMax) {
    var items = ['complete', 'partial', 'invalid', 'missing'].map(function (k) {
      var s = FB.STATUS[k];
      return '<li><span class="swatch ' + s.cls + '" aria-hidden="true">' + s.glyph + '</span>' +
        '<span><b>' + E(s.label) + '</b> — ' + E(s.desc) + '</span></li>';
    }).join('');
    return '<ul class="legend">' + items +
      '<li><span class="swatch" style="background:none;border:0" aria-hidden="true">' +
      '<svg width="22" height="14" viewBox="0 0 22 14" aria-hidden="true">' +
      '<line x1="3" y1="7" x2="19" y2="7" stroke="var(--accent)" stroke-width="3" opacity="0.42"/>' +
      '<line x1="3" y1="2.5" x2="3" y2="11.5" stroke="var(--accent)" stroke-width="1.6"/>' +
      '<line x1="19" y1="2.5" x2="19" y2="11.5" stroke="var(--accent)" stroke-width="1.6"/>' +
      '<circle cx="9" cy="7" r="3" fill="var(--accent)"/></svg></span>' +
      '<span><b>误差棒</b> — 点为成功率，两端为 95% CI，横轴 0–' + Math.round(axisMax * 100) +
      '%（全表统一刻度，窄屏隐藏，CI 数值始终以文字给出）</span></li>' +
      '</ul>';
  }

  function defaultView(idx) {
    var refs = refCandidates(idx);
    return {
      scope: 'all',
      sort: 'category',
      ref: refs.length ? refs[0].id : (idx.models[0] ? idx.models[0].id : ''),
      cols: 'all'
    };
  }

  function viewFromQuery(idx) {
    var d = defaultView(idx);
    var q = FB.qsGet();
    var refs = refCandidates(idx);
    var v = {
      scope: pickById(SCOPES, q.scope, SCOPES[0]).id,
      sort: pickById(SORTS, q.sort, SORTS[0]).id,
      cols: pickById(COLMODES, q.cols, COLMODES[0]).id,
      ref: d.ref
    };
    if (q.ref) {
      refs.forEach(function (m) { if (m.id === q.ref) v.ref = q.ref; });
    }
    return v;
  }

  function renderLeaderboard() {
    var root = document.getElementById('lb-root');
    if (!root) return;

    FB.load(function (idx, meta) {
      var axisMax = FB.axisMaxFor(idx.allResults());
      var view = viewFromQuery(idx);
      var d = defaultView(idx);

      /* 分叉条的共用刻度：所有 diff_pp 绝对值的最大值，向上取整到 5 */
      if (idx.h2h) {
        var mx = 0;
        idx.h2h.list.forEach(function (c) { mx = Math.max(mx, Math.abs(c.diff_pp || 0)); });
        idx._maxPP = Math.max(5, Math.ceil(mx / 5) * 5);
      }

      /* 表格区随视角重绘；页面其余部分（覆盖率、显著性检验）保持不变。 */
      function paintTable() {
        var slot = document.getElementById('lb-table-slot');
        if (!slot) return;
        var footnotes = [];
        var tableHTML = renderTable(idx, axisMax, footnotes, view);
        var h = viewSummaryHTML(idx, view);
        h += '<p class="scroll-hint">表格可左右滑动查看全部模型列。</p>';
        h += '<div class="table-scroll" tabindex="0" role="region" aria-labelledby="table-h">' +
          tableHTML + '</div>';
        h += legendHTML(axisMax);
        if (footnotes.length) {
          h += '<h3>无效结果说明</h3><ul class="footnotes">';
          footnotes.forEach(function (f, i) {
            h += '<li><span class="marker">[' + (i + 1) + ']</span><code>' + E(f.key) +
              '</code> — ' + E(f.text) + '</li>';
          });
          h += '</ul>';
        }
        slot.innerHTML = h;
      }

      var h = '';
      if (meta.sample) h += FB.sampleBannerHTML();
      if (meta.warning) {
        h += '<div class="note caution"><span class="note-title">数据格式提醒</span><p>' + E(meta.warning) + '</p></div>';
      }

      h += metaStrip(idx.raw, meta);
      h += coveragePanel(idx);
      h += submissionsSection(idx);
      h += '<h2 id="table-h">逐任务成功率</h2>';
      h += controlsHTML(idx, view);
      h += '<div id="lb-table-slot"></div>';
      h += h2hSection(idx);

      root.innerHTML = h;
      paintTable();

      var form = document.getElementById('lb-controls');
      if (form) {
        form.addEventListener('change', function (ev) {
          var el = ev.target;
          var key = el && el.getAttribute && el.getAttribute('data-ctl');
          if (!key) return;
          if (key === 'ctl-scope') view.scope = el.value;
          else if (key === 'ctl-sort') view.sort = el.value;
          else if (key === 'ctl-ref') view.ref = el.value;
          else if (key === 'ctl-cols') view.cols = el.value;
          else return;
          FB.qsSet({
            scope: view.scope === d.scope ? null : view.scope,
            sort: view.sort === d.sort ? null : view.sort,
            ref: view.ref === d.ref ? null : view.ref,
            cols: view.cols === d.cols ? null : view.cols
          });
          paintTable();
        });
        form.addEventListener('click', function (ev) {
          var el = ev.target;
          if (!el || !el.getAttribute || el.getAttribute('data-ctl') !== 'reset') return;
          view.scope = d.scope; view.sort = d.sort; view.ref = d.ref; view.cols = d.cols;
          ['ctl-scope', 'ctl-sort', 'ctl-ref', 'ctl-cols'].forEach(function (id) {
            var s = document.getElementById(id);
            if (s) s.value = view[id.replace('ctl-', '')];
          });
          FB.qsSet({ scope: null, sort: null, ref: null, cols: null });
          paintTable();
        });
      }
    }, function (info) {
      root.innerHTML = FB.emptyStateHTML(info);
    });
  }

  /* ============ 任务说明页 ============ */

  function renderTasks() {
    var fcRoot = document.getElementById('tasks-fc');
    var crRoot = document.getElementById('tasks-cr');
    var srcNote = document.getElementById('tasks-source');

    function paint(tasks, fromData) {
      var fc = tasks.filter(function (t) { return t.force_category === 'force_classification'; });
      var cr = tasks.filter(function (t) { return t.force_category === 'contact_rich'; });
      fcRoot.innerHTML = fc.map(function (t) { return taskCard(t, { detail: true }); }).join('');
      crRoot.innerHTML = cr.map(function (t) { return taskCard(t, { detail: true }); }).join('');
      if (srcNote) {
        srcNote.innerHTML = fromData
          ? '任务的中文名与说明读取自 <code>results/leaderboard.json</code>。'
          : '<b>注意：</b><code>results/leaderboard.json</code> 当前读取不到，' +
            '下面展示的是站点内置的任务说明（译自仓库 README 的任务表，不含任何评测数字）。' +
            '数据文件生成后，将以其中的 <code>display_name_zh</code> / <code>summary_zh</code> 为准。';
      }
    }

    paint(FB.TASK_FALLBACK, false);

    FB.load(function (idx, meta) {
      if (meta.sample) document.getElementById('sample-slot').innerHTML = FB.sampleBannerHTML();
      paint(idx.tasks, true);
    }, function () { /* 保持内置说明，提示已在 paint 中给出 */ });
  }

  /* ============ 关于页 ============ */

  function renderAbout() {
    var slot = document.getElementById('about-protocol');
    if (!slot) return;
    FB.load(function (idx, meta) {
      if (meta.sample) {
        slot.innerHTML = FB.sampleBannerHTML() +
          '<p class="muted small">样例模式下不展示评测协议参数。</p>';
        return;
      }
      var p = idx.raw.eval_protocol || {};
      var rows = [
        ['seed 范围', (p.seed_start != null ? p.seed_start + ' – ' + p.seed_end : '—')],
        ['每格 rollout 次数', p.num_rollouts != null ? p.num_rollouts : '—'],
        ['action chunk 执行步数', p.execute_chunk != null ? p.execute_chunk : '—'],
        ['物理步频', p.physics_rate_hz != null ? p.physics_rate_hz + ' Hz' : '—'],
        ['控制频率', p.control_rate_hz != null ? p.control_rate_hz + ' Hz' : '—']
      ];
      var h = '<div class="tbl-wrap"><table class="plain"><caption class="visually-hidden">评测协议参数</caption>' +
        '<tbody>' + rows.map(function (r) {
          return '<tr><th scope="row">' + E(r[0]) + '</th><td>' + E(r[1]) + '</td></tr>';
        }).join('') + '</tbody></table></div>';
      if (p.notes_zh) h += '<p class="small muted">' + E(p.notes_zh) + '</p>';
      h += '<p class="small muted">以上参数读取自 <code>results/leaderboard.json</code> 的 <code>eval_protocol</code> 字段。</p>';
      slot.innerHTML = h;
    }, function (info) {
      slot.innerHTML = '<p class="muted">评测协议参数存放在 <code>results/leaderboard.json</code> 的 ' +
        '<code>eval_protocol</code> 字段中，该文件当前读取不到，因此这里不填任何数字。</p>' +
        '<p class="detail small muted mono">' + E(info.url) + ' — ' + E(info.reason) + '</p>';
    });
  }

  /* ============ 分发 ============ */

  document.addEventListener('DOMContentLoaded', function () {
    var page = document.body.getAttribute('data-page');
    if (page === 'home') renderHome();
    else if (page === 'leaderboard') renderLeaderboard();
    else if (page === 'tasks') renderTasks();
    else if (page === 'about') renderAbout();
  });

})();

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
      if (t.dataset_episodes != null) h += '<dt>训练集 episode 数</dt><dd>' + E(t.dataset_episodes) + '</dd>';
      h += '<dt>为什么归到「' + E(c.label || '') + '」</dt><dd>' + E(c.desc || '') + '</dd>';
      h += '</dl>';
    }
    h += '</article>';
    return h;
  }

  /* 覆盖率面板：显式回答"24 格里做完了几格" */
  function coveragePanel(idx) {
    var c = idx.counts();
    var done = c.complete;
    var pctDone = c.total ? Math.round(done / c.total * 100) : 0;

    var legend = '<ul class="status-legend">';
    ['complete', 'partial', 'invalid', 'missing'].forEach(function (k) {
      var s = FB.STATUS[k];
      legend += '<li><span class="swatch ' + s.cls + '" aria-hidden="true">' + s.glyph + '</span>' +
        '<span>' + E(s.label) + '</span><span class="count">' + c[k] + ' 格</span></li>';
    });
    legend += '</ul>';

    /* 状态矩阵：任务 × 模型 */
    var mtx = '<div class="matrix" role="group" aria-label="评测覆盖矩阵：8 个任务 × 3 个模型">';
    mtx += '<div class="matrix-row matrix-head"><span class="rowlab">任务 \\ 模型</span>';
    idx.models.forEach(function (m) {
      mtx += '<span class="cell" style="background:none;border:0">' + E(m.short_name || m.id) + '</span>';
    });
    mtx += '</div>';
    idx.tasks.forEach(function (t) {
      mtx += '<div class="matrix-row"><span class="rowlab" title="' + E(t.id) + '">' + E(t.id) + '</span>';
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
      '<div class="coverage-bar" role="img" aria-label="24 格中 ' + done + ' 格已完成，约 ' + pctDone + '%">' +
      '<span style="width:' + pctDone + '%"></span></div>' +
      legend +
      '<p class="small muted" style="margin-top:12px">' +
      '共 ' + idx.tasks.length + ' 个任务 × ' + idx.models.length + ' 个模型 = ' + c.total +
      ' 个 (任务, 模型) 组合。空缺不是 0 分，是没跑。</p>' +
      '</div>' +
      '<div>' + mtx + '</div>' +
      '</section>';
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

  function renderHome() {
    var cov = document.getElementById('home-coverage');
    var cards = document.getElementById('home-tasks');
    var mods = document.getElementById('home-models');

    if (cards) {
      cards.innerHTML = FB.TASK_FALLBACK.map(function (t) { return taskCard(t, { detail: false }); }).join('');
    }

    FB.load(function (idx, meta) {
      if (meta.sample) {
        document.getElementById('sample-slot').innerHTML = FB.sampleBannerHTML();
      }
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

  function renderTable(idx, axisMax, footnotes) {
    var hasH2H = !!idx.h2h;
    var ncols = idx.models.length + 1 + (hasH2H ? 1 : 0);

    var h = '<table class="lb">';
    h += '<caption>成功率越高越好。每个数字后面的 95% CI 是 Clopper-Pearson 精确二项区间；' +
      'n = 100 时区间宽度接近 18 个百分点，请不要只看点估计。' +
      (hasH2H ? '最右一列给出该任务上两个模型差异的 p 值，' +
        '避免只凭并排的百分数过度解读。' : '') + '</caption>';

    h += '<thead><tr><th scope="col">任务</th>';
    idx.models.forEach(function (m) {
      var fi = m.force_input || '';
      h += '<th scope="col">' +
        '<span class="mname">' + E(m.display_name_zh || m.id) + '</span>' +
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
    idx.tasks.forEach(function (t) {
      if (t.force_category !== lastCat) {
        if (lastCat !== null) h += '</tbody>';
        lastCat = t.force_category;
        var c = FB.CATEGORY[lastCat] || { label: lastCat, desc: '' };
        h += '<tbody><tr class="group-head"><th scope="rowgroup" colspan="' + ncols + '">' +
          '<span class="glabel">' + catBadge(lastCat) +
          '<span class="muted small" style="font-weight:400">' + E(c.desc) + '</span></span></th></tr>';
      }
      h += '<tr><th scope="row">' + E(t.display_name_zh || t.id) +
        '<span class="tid">' + E(t.id) + '</span></th>';
      idx.models.forEach(function (m) {
        h += '<td>' + resultCell(idx.get(t.id, m.id), axisMax, footnotes) + '</td>';
      });
      if (hasH2H) h += h2hCell(idx, t.id);
      h += '</tr>';
    });
    if (lastCat !== null) h += '</tbody>';

    h += '</table>';
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

  function renderLeaderboard() {
    var root = document.getElementById('lb-root');
    if (!root) return;

    FB.load(function (idx, meta) {
      var footnotes = [];
      var axisMax = FB.axisMaxFor(idx.allResults());
      var h = '';

      if (meta.sample) h += FB.sampleBannerHTML();
      if (meta.warning) {
        h += '<div class="note caution"><span class="note-title">数据格式提醒</span><p>' + E(meta.warning) + '</p></div>';
      }

      /* 分叉条的共用刻度：所有 diff_pp 绝对值的最大值，向上取整到 5 */
      if (idx.h2h) {
        var mx = 0;
        idx.h2h.list.forEach(function (c) { mx = Math.max(mx, Math.abs(c.diff_pp || 0)); });
        idx._maxPP = Math.max(5, Math.ceil(mx / 5) * 5);
      }

      h += metaStrip(idx.raw, meta);
      h += coveragePanel(idx);
      h += '<h2 id="table-h">逐任务成功率</h2>';
      h += '<p class="scroll-hint">表格可左右滑动查看全部 3 个模型。</p>';

      var tableHTML = renderTable(idx, axisMax, footnotes);
      h += '<div class="table-scroll" tabindex="0" role="region" aria-labelledby="table-h">' + tableHTML + '</div>';
      h += legendHTML(axisMax);

      if (footnotes.length) {
        h += '<h3>无效结果说明</h3><ul class="footnotes">';
        footnotes.forEach(function (f, i) {
          h += '<li><span class="marker">[' + (i + 1) + ']</span><code>' + E(f.key) + '</code> — ' + E(f.text) + '</li>';
        });
        h += '</ul>';
      }

      h += h2hSection(idx);

      root.innerHTML = h;
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

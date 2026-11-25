/* 关联抽样工作台前端逻辑（无构建步骤，原生 JS）。 */
"use strict";

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const SVG_NS = "http://www.w3.org/2000/svg";

const state = {
  runs: [],
  current: null,
};

// ---------------------------------------------------------------- 主题
function initTheme() {
  const saved = localStorage.getItem("join-sampler-theme") || "auto";
  $("#theme-select").value = saved;
  applyTheme(saved);
  $("#theme-select").addEventListener("change", (e) => {
    localStorage.setItem("join-sampler-theme", e.target.value);
    applyTheme(e.target.value);
    if (state.current) renderAll(state.current);
  });
}
function applyTheme(mode) {
  if (mode === "auto") document.documentElement.removeAttribute("data-theme");
  else document.documentElement.setAttribute("data-theme", mode);
}

// ---------------------------------------------------------------- Tabs
$$(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    $$(".tab").forEach((t) => {
      t.classList.toggle("active", t === tab);
      t.setAttribute("aria-selected", t === tab ? "true" : "false");
    });
    $$(".tab-panel").forEach((p) =>
      p.classList.toggle("hidden", p.dataset.panel !== tab.dataset.tab)
    );
  });
});

// ---------------------------------------------------------------- 工具
function fmtInt(n) {
  return Number(n).toLocaleString("zh-CN");
}
function fmtBytes(n) {
  if (n == null) return "—";
  const units = ["B", "KB", "MB", "GB"];
  let v = n, u = 0;
  while (v >= 1024 && u < units.length - 1) { v /= 1024; u++; }
  return `${v.toFixed(v >= 100 || u === 0 ? 0 : 1)} ${units[u]}`;
}
function fmtProb(p) {
  if (p == null || !isFinite(p)) return "—";
  if (p === 0) return "0";
  if (p < 1e-4) return p.toExponential(2);
  return p.toFixed(5);
}
function cssVar(name) {
  return getComputedStyle($("#app")).getPropertyValue(name).trim();
}
async function api(path, options = {}) {
  const resp = await fetch(path, {
    headers: options.body instanceof FormData ? {} : { "Content-Type": "application/json" },
    ...options,
  });
  const data = await resp.json();
  if (!resp.ok) throw new Error(data.error || `HTTP ${resp.status}`);
  return data;
}

// ---------------------------------------------------------------- 新建运行
function collectConfig() {
  const numOrNull = (v) => (v === "" || v == null ? null : Number(v));
  return {
    sample_size: Number($("#cfg-n").value),
    mode: $("#cfg-mode").value,
    seed: numOrNull($("#cfg-seed").value),
    max_pairs_per_key: Number($("#cfg-max-pairs").value),
    estimator_probes: Number($("#cfg-probes").value),
    max_build_evaluations: numOrNull($("#cfg-max-eval").value),
    max_build_seconds: numOrNull($("#cfg-max-sec").value),
    max_memory_bytes: numOrNull($("#cfg-max-mem").value),
  };
}

$("#btn-run").addEventListener("click", async () => {
  $("#form-error").textContent = "";
  const activeTab = $(".tab.active").dataset.tab;
  try {
    let result;
    if (activeTab === "synthetic") {
      result = await api("/api/synthetic", {
        method: "POST",
        body: JSON.stringify({
          left_rows: Number($("#syn-left-rows").value),
          right_rows: Number($("#syn-right-rows").value),
          distinct_keys: Number($("#syn-keys").value),
          hot_key_multiplier: Number($("#syn-hot").value),
          null_key_fraction: Number($("#syn-null").value),
          data_seed: Number($("#syn-data-seed").value),
          condition: $("#cfg-condition").value || null,
          ...collectConfig(),
        }),
      });
    } else {
      const lf = $("#up-left").files[0], rf = $("#up-right").files[0];
      if (!lf || !rf) throw new Error("请同时选择左右两个 CSV 文件");
      const fd = new FormData();
      fd.append("left_file", lf);
      fd.append("right_file", rf);
      fd.append("key", $("#cfg-key").value);
      fd.append("condition", $("#cfg-condition").value || "");
      Object.entries(collectConfig()).forEach(([k, v]) => {
        if (v !== null) fd.append(k, String(v));
      });
      result = await api("/api/upload", { method: "POST", body: fd });
    }
    await refreshRuns();
    selectRun(result.id);
  } catch (err) {
    $("#form-error").textContent = err.message;
  }
});

// ---------------------------------------------------------------- 追加批次
$("#btn-append").addEventListener("click", async () => {
  if (!state.current) return;
  try {
    const result = await api(`/api/runs/${state.current.id}/append`, {
      method: "POST",
      body: JSON.stringify({ more: Number($("#append-n").value) }),
    });
    await refreshRuns();
    renderAll(result);
  } catch (err) {
    alert(err.message);
  }
});

// ---------------------------------------------------------------- 运行列表
async function refreshRuns() {
  const data = await api("/api/runs");
  state.runs = data.runs;
  const tbody = $("#table-runs tbody");
  tbody.innerHTML = "";
  for (const r of state.runs) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td><code>${r.id}</code></td>
      <td>${r.label ?? ""}</td>
      <td><span class="pill ${r.mode}">${r.mode}</span></td>
      <td class="num">${r.seed ?? "随机"}</td>
      <td class="num">${fmtInt(r.sample_size)}</td>
      <td><code>${r.snapshot_fingerprint.slice(0, 10)}…</code></td>
      <td><button data-id="${r.id}">查看</button></td>`;
    tr.querySelector("button").addEventListener("click", () => selectRun(r.id));
    tbody.appendChild(tr);
  }
  for (const sel of ["#cmp-base", "#cmp-other"]) {
    const node = $(sel);
    const prev = node.value;
    node.innerHTML = "";
    for (const r of state.runs) {
      const opt = document.createElement("option");
      opt.value = r.id;
      opt.textContent = `${r.id} · ${r.label ?? ""}（${r.sample_size} 对）`;
      node.appendChild(opt);
    }
    if (prev) node.value = prev;
  }
  $("#compare-panel").classList.toggle("hidden", state.runs.length < 2);
}

async function selectRun(runId) {
  const detail = await api(`/api/runs/${runId}`);
  state.current = detail;
  renderAll(detail);
  $("#run-panel").scrollIntoView({ behavior: "smooth", block: "start" });
}

// ---------------------------------------------------------------- 总渲染
function renderAll(detail) {
  $("#run-panel").classList.remove("hidden");
  $("#run-title").textContent = `· ${detail.label ?? detail.id}`;
  renderStats(detail);
  renderKeyChart(detail);
  renderKeyTable(detail);
  renderRecordTable(detail);
  renderMemory(detail);
}

function statTile(label, value, sub) {
  return `<div class="stat-tile"><div class="label">${label}</div>
    <div class="value">${value}</div>${sub ? `<div class="sub">${sub}</div>` : ""}</div>`;
}

function renderStats(d) {
  const bs = d.build_stats, sd = d.snapshot_describe, cfg = d.config;
  $("#stat-grid").innerHTML = [
    statTile("已抽关联对", fmtInt(d.sample_size), `模式 ${cfg.mode} · 种子 ${d.rng_seed ?? "随机"}`),
    statTile("全量关联对（不物化）", fmtInt(sd.total_pair_count), "仅计数，未生成结果集"),
    statTile("关联键数", fmtInt(bs.join_key_count), `精确 ${bs.exact_key_count} · 估计 ${bs.estimate_key_count}`),
    statTile("整键零结果", fmtInt(bs.zero_result_key_count), "条件使该键无任何接受对"),
    statTile("条件求值次数", fmtInt(bs.condition_evaluations), `构建耗时 ${bs.build_seconds}s`),
    statTile("校正后总权重", fmtInt(Math.round(bs.total_weight)), "可接受关联对总量（估计键为估计值）"),
    statTile("左表空键行", fmtInt(sd.left_null_key_rows), `左表总行 ${fmtInt(sd.left_rows)}`),
    statTile("右表空键行", fmtInt(sd.right_null_key_rows), `右表总行 ${fmtInt(sd.right_rows)}`),
  ].join("");
}

function renderMemory(d) {
  const m = d.memory_report;
  $("#memory-grid").innerHTML = [
    statTile("两侧表存储", fmtBytes(m.store_bytes), "分段列式内存"),
    statTile("抽样器结构", fmtBytes(m.sampler_bytes), "键统计 + Fenwick + 已选集合"),
    statTile("接受对缓存", fmtInt(m.accepted_pairs_cached), "受 max_accepted_cache_pairs 约束"),
    statTile("物化关联行", fmtInt(m.materialized_join_rows), "恒为 0：完整关联结果从不落盘/入内存"),
  ].join("");
}

// ---------------------------------------------------------------- SVG 柱状图
function renderKeyChart(d) {
  const tooltip = $("#tooltip");
  tooltip.hidden = true;
  const hits = new Map(d.key_selection_counts.map((x) => [String(x.key), x.count]));
  const rows = d.key_build
    .map((k) => ({ ...k, hit: hits.get(String(k.key)) ?? 0 }))
    .filter((k) => k.weight > 0)
    .sort((a, b) => b.weight - a.weight)
    .slice(0, 30);

  const cBlue = cssVar("--series-1"), cOrange = cssVar("--series-2");
  // 图例固定在独立容器中，重渲染只替换内容，不会堆积。
  $("#chart-legend").innerHTML = `
    <span class="key"><span class="swatch" style="background:${cBlue}"></span>上图：抽样命中次数</span>
    <span class="key"><span class="swatch" style="background:${cOrange}"></span>下图：键校正权重（可接受关联对数）</span>`;

  // 两个小倍数图：同组键、同一 X 顺序，各自单一 Y 刻度（禁止双轴）。
  renderSingleBarChart({
    svg: $("#chart-keys-hit"), rows, value: (r) => r.hit,
    color: cBlue, tickFmt: (v) => fmtInt(Math.round(v)),
    tooltip, tooltipHtml: (r) => tooltipHtml(r, "命中次数", fmtInt(r.hit)),
    showXLabels: false,
  });
  renderSingleBarChart({
    svg: $("#chart-keys-weight"), rows, value: (r) => r.weight,
    color: cOrange, tickFmt: compactNum,
    tooltip, tooltipHtml: (r) => tooltipHtml(r, "校正权重", fmtInt(Math.round(r.weight))),
    showXLabels: true,
  });
}

function tooltipHtml(r, metricName, metricValue) {
  return `
    <b>键 ${escapeHtml(String(r.key))}</b>
    <div class="row"><span>${metricName}</span><span>${metricValue}</span></div>
    <div class="row"><span>关联对数</span><span>${fmtInt(r.pair_count)}</span></div>
    <div class="row"><span>接受率</span><span>${(r.acceptance_rate * 100).toFixed(2)}%${
      r.rate_smoothed ? "（零命中平滑）" : r.rate_is_estimate ? "（估计）" : ""
    }</span></div>
    <div class="row"><span>左/右行数</span><span>${r.left_rows} / ${r.right_rows}</span></div>`;
}

function renderSingleBarChart({ svg, rows, value, color, tickFmt, tooltip, tooltipHtml, showXLabels }) {
  svg.innerHTML = "";
  const W = Math.max(720, rows.length * 34 + 80);
  const H = showXLabels ? 220 : 180;
  const M = { top: 16, right: 16, bottom: showXLabels ? 58 : 12, left: 64 };
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", String(W));
  svg.setAttribute("height", String(H));

  const maxV = Math.max(1e-9, ...rows.map(value));
  const plotH = H - M.top - M.bottom, plotW = W - M.left - M.right;
  const baseline = M.top + plotH;
  const ticks = 4;

  for (let t = 0; t <= ticks; t++) {
    const y = M.top + (plotH / ticks) * (ticks - t);
    svg.appendChild(el("line", { x1: M.left, x2: W - M.right, y1: y, y2: y, class: "gridline" }));
    const label = el("text", { x: M.left - 8, y: y + 4, "text-anchor": "end" });
    label.textContent = tickFmt((maxV / ticks) * t);
    svg.appendChild(label);
  }
  svg.appendChild(el("line", {
    x1: M.left, x2: W - M.right, y1: baseline, y2: baseline, class: "axis-line",
  }));

  const slotW = plotW / rows.length;
  const barW = Math.min(16, slotW * 0.62);
  rows.forEach((r, i) => {
    const x0 = M.left + slotW * i + slotW / 2;
    const h = (value(r) / maxV) * plotH;
    // 仅数据端 4px 圆角、基线端直角（dataviz 标记规范）。
    const bar = el("path", {
      d: topRoundedRect(x0 - barW / 2, baseline - h, barW, h, 4),
      fill: color,
    });
    const hitArea = el("rect", {
      x: x0 - slotW / 2, y: M.top, width: slotW, height: plotH, class: "bar-hit",
    });
    const g = el("g", {});
    g.append(bar, hitArea);

    if (showXLabels) {
      const xLabel = el("text", {
        x: x0, y: baseline + 16, "text-anchor": "end",
        transform: `rotate(-38 ${x0} ${baseline + 16})`,
      });
      const keyText = String(r.key);
      xLabel.textContent = keyText.length > 8 ? keyText.slice(0, 7) + "…" : keyText;
      g.appendChild(xLabel);
    }

    for (const node of [bar, hitArea]) {
      node.addEventListener("mousemove", (ev) => {
        tooltip.hidden = false;
        tooltip.innerHTML = tooltipHtml(r);
        const wrap = svg.parentElement.getBoundingClientRect();
        tooltip.style.left = `${ev.clientX - wrap.left + 14}px`;
        tooltip.style.top = `${ev.clientY - wrap.top + 14}px`;
      });
      node.addEventListener("mouseleave", () => { tooltip.hidden = true; });
    }
    svg.appendChild(g);
  });
}

function el(tag, attrs) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  return node;
}

// 顶边（数据端）圆角 r、底边（基线端）直角的柱形 path。
function topRoundedRect(x, y, w, h, r) {
  const rr = Math.min(r, w / 2, h);
  if (h <= rr) return `M${x},${y + h} L${x},${y} L${x + w},${y} L${x + w},${y + h} Z`;
  return [
    `M${x},${y + h}`,
    `L${x},${y + rr}`,
    `Q${x},${y} ${x + rr},${y}`,
    `L${x + w - rr},${y}`,
    `Q${x + w},${y} ${x + w},${y + rr}`,
    `L${x + w},${y + h}`,
    "Z",
  ].join(" ");
}
function compactNum(n) {
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(Math.round(n));
}
function escapeHtml(s) {
  return s.replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ---------------------------------------------------------------- 键表
function renderKeyTable(d) {
  const tbody = $("#table-keys tbody");
  tbody.innerHTML = "";
  const hits = new Map(d.key_selection_counts.map((x) => [String(x.key), x.count]));
  const rows = [...d.key_build].sort((a, b) => b.weight - a.weight);
  for (const k of rows) {
    const tr = document.createElement("tr");
    const zero = k.acceptance_rate === 0;
    const modePill = zero
      ? `<span class="pill zero">整键零结果</span>`
      : `<span class="pill ${k.weight_mode}">${
          { exact: "精确枚举", estimate: "探针估计", unconditioned: "无条件" }[k.weight_mode]
        }</span>`;
    tr.innerHTML = `
      <td>${escapeHtml(String(k.key))}</td>
      <td class="num">${fmtInt(k.left_rows)}</td>
      <td class="num">${fmtInt(k.right_rows)}</td>
      <td class="num">${fmtInt(k.pair_count)}</td>
      <td>${modePill}</td>
      <td class="num">${(k.acceptance_rate * 100).toFixed(3)}%</td>
      <td>${
        k.rate_smoothed
          ? "零命中平滑（" + k.probes_used + " 探针无命中）"
          : k.rate_is_estimate
            ? "估计（" + k.probes_used + " 探针）"
            : k.weight_mode === "unconditioned" ? "—" : "精确（全枚举 " + k.probes_used + "）"
      }</td>
      <td class="num">${fmtInt(Math.round(k.weight))}</td>
      <td class="num">${fmtInt(k.accepted_count)}</td>
      <td class="num">${fmtInt(k.probes_used)}</td>
      <td class="num">${fmtInt(hits.get(String(k.key)) ?? 0)}</td>`;
    tbody.appendChild(tr);
  }
}

// ---------------------------------------------------------------- 结果对表
function renderRecordTable(d) {
  const tbody = $("#table-records tbody");
  tbody.innerHTML = "";
  for (const r of d.records) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="num">${r.draw_index + 1}</td>
      <td>${escapeHtml(String(r.key))}</td>
      <td class="num">${r.left_row_id}</td>
      <td class="num">${r.right_row_id}</td>
      <td>${escapeHtml(JSON.stringify(r.left_row))}</td>
      <td>${escapeHtml(JSON.stringify(r.right_row))}</td>
      <td class="num">${fmtProb(r.p_select_key)}</td>
      <td class="num">${fmtProb(r.p_select_pair_within_key)}</td>
      <td class="num">${fmtProb(r.p_joint)}</td>
      <td><span class="pill ${r.weight_mode}">${
        { exact: "精确", estimate: "估计", unconditioned: "无条件" }[r.weight_mode]
      }</span>${r.rate_smoothed ? " 零命中平滑" : r.rate_is_estimate ? " *" : ""}</td>
      <td class="num">${r.rejection_attempts}</td>`;
    tbody.appendChild(tr);
  }
}

// ---------------------------------------------------------------- 比较
$("#btn-compare").addEventListener("click", async () => {
  try {
    const report = await api("/api/compare", {
      method: "POST",
      body: JSON.stringify({
        base_id: $("#cmp-base").value,
        other_id: $("#cmp-other").value,
      }),
    });
    const node = $("#compare-result");
    node.innerHTML = `
      <div class="compare-cards">
        ${statTile("基准样本量", fmtInt(report.base_sample_size), report.base_label)}
        ${statTile("对比样本量", fmtInt(report.other_sample_size), report.other_label)}
        ${statTile("共同关联对", fmtInt(report.shared_pair_count), "同一 (左行号, 右行号)")}
        ${statTile("基准独有", fmtInt(report.base_only_pair_count), "")}
        ${statTile("对比独有", fmtInt(report.other_only_pair_count), "")}
        ${statTile("同一快照", report.same_snapshot ? "是" : "否", "指纹比对")}
      </div>
      <div class="table-wrap" style="margin-top:12px">
        <table><thead><tr><th>键</th><th class="num">基准命中</th><th class="num">对比命中</th><th class="num">差值</th></tr></thead>
        <tbody>${report.key_distribution
          .map((k) => `<tr><td>${escapeHtml(String(k.key))}</td>
            <td class="num">${fmtInt(k.base_count)}</td>
            <td class="num">${fmtInt(k.other_count)}</td>
            <td class="num">${k.delta >= 0 ? "+" : ""}${fmtInt(k.delta)}</td></tr>`)
          .join("")}</tbody></table>
      </div>`;
  } catch (err) {
    alert(err.message);
  }
});

// ---------------------------------------------------------------- 启动
initTheme();
refreshRuns().catch((err) => {
  $("#form-error").textContent = `无法连接后端：${err.message}`;
});

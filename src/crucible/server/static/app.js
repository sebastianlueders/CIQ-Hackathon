let latestSummary = null;
let latestLiveSnap = null;
let yamlTab = "turnkey";

async function fetchSummary() {
  const resp = await fetch("/api/summary");
  return await resp.json();
}

async function fetchLiveStatus() {
  const resp = await fetch("/api/live-status");
  return await resp.json();
}

async function initBackendDefault() {
  try {
    const resp = await fetch("/api/discover");
    const info = await resp.json();
    const sel = document.getElementById("backend-select");
    if (info.discovery_report && info.discovery_report.fuzzball_cli_available) {
      sel.value = "fuzzball_mcp";
    } else {
      sel.value = "simulator";
    }
  } catch (_) {}
}

function renderNodeCards(liveSnap) {
  const wrap = document.getElementById("node-cards");
  const nodes = (liveSnap && liveSnap.nodes) || [];
  wrap.innerHTML = nodes
    .map((n) => {
      const running = n.state === "RUNNING";
      const wfText = running
        ? `ACTIVE: ${n.active_trial_id} · wf ${String(n.active_workflow_id || "").slice(0, 12)}…`
        : n.last_trial_id
        ? `LAST: ${n.last_trial_id} · wf ${String(n.last_workflow_id || "").slice(0, 12)}… (${n.trials_completed} runs)`
        : "Idle · Awaiting Fuzzball allocation";
      const badgeCls = running ? "status-ok" : "";
      return `
        <div class="node-card ${running ? "node-running" : ""}">
          <div class="node-top">
            <div>
              <div class="node-name">${n.hostname}</div>
              <div class="node-id"><code>${n.node_id}</code></div>
            </div>
            <span class="stage-status ${badgeCls}">${n.state}</span>
          </div>
          <div class="node-specs">
            1&times; <code>${n.gpu_product}</code> (<code>sm_121</code>) &middot;
            <code>${n.alloc_cores}/${n.physical_cores}c</code> (<code>${n.arch_device}</code>) &middot;
            <code>${n.alloc_memory_gib}/${n.physical_memory_gib} GiB</code>
          </div>
          <div class="node-wf"><code>${wfText}</code></div>
        </div>
      `;
    })
    .join("");
}

function renderLiveTelemetry(snap) {
  latestLiveSnap = snap;
  document.getElementById("pill-def-id").textContent = snap.definition_id || "team20";
  document.getElementById("live-stage-label").textContent = snap.stage_label || "Ready on Team 20";

  const stBadge = document.getElementById("live-run-state");
  stBadge.textContent = `${String(snap.state || "idle").toUpperCase()} · ${snap.backend}`;
  stBadge.className =
    snap.state === "running"
      ? "badge badge-accent"
      : snap.state === "completed"
      ? "badge badge-green"
      : "badge";

  const errBanner = document.getElementById("live-error-banner");
  if (snap.error) {
    errBanner.textContent = snap.error;
    errBanner.classList.remove("hidden");
  } else {
    errBanner.classList.add("hidden");
  }

  renderNodeCards(snap);

  const feed = document.getElementById("live-event-feed");
  const events = snap.events || [];
  feed.innerHTML = events
    .slice()
    .reverse()
    .map(
      (ev) =>
        `<div class="event-line"><span class="ev-ts">[${ev.time_utc}]</span>${ev.message}</div>`
    )
    .join("");

  if (snap.state === "running" && snap.live_curves && Object.keys(snap.live_curves).length > 0) {
    renderLossSvg(snap.live_curves);
  }
  if (snap.state === "running" && snap.live_firewall && snap.live_firewall.length > 0) {
    const lastRec = snap.live_firewall[snap.live_firewall.length - 1].record || {};
    const fwBox = document.getElementById("firewall-audit-box");
    fwBox.innerHTML =
      `<strong>Live Privacy Schema Firewall:</strong> ${snap.live_firewall.length} advisor payloads verified · ` +
      `<code>contains_raw_data=false</code> · SHA-256: <code>${String(lastRec.payload_sha256 || "").slice(0, 16)}…</code>`;
  }
  refreshYamlViewer();
}

function renderLossSvg(curves) {
  const wrap = document.getElementById("loss-chart-wrap");
  const legend = document.getElementById("loss-legend");
  const seriesDefs = [
    { key: "stageb-00-pytorch-baseline-1node", label: "PyTorch Baseline (1-Node SPARK39)", color: "#38bdf8", dash: "" },
    { key: "stageb-04-sm121-all-kernels-2node-ddp", label: "sm_121 Kernels + 2-Node SPARK39+40 DDP (Accepted)", color: "#a3e635", dash: "5,3" },
    { key: "stageb-05-oversized-global-batch-2node", label: "Oversized Batch 96 (Rejected: Curve Drift)", color: "#f87171", dash: "2,3" },
  ];

  const active = seriesDefs.filter((s) => Array.isArray(curves[s.key]) && curves[s.key].length > 1);
  if (active.length === 0) {
    const fallbackKeys = Object.keys(curves).slice(-3);
    const palette = ["#38bdf8", "#a3e635", "#c084fc"];
    fallbackKeys.forEach((k, idx) => {
      active.push({ key: k, label: k, color: palette[idx % palette.length], dash: "" });
    });
  }
  if (active.length === 0) return;

  const width = 620;
  const height = 198;
  const pad = { top: 14, right: 18, bottom: 26, left: 44 };
  let allVals = [];
  active.forEach((s) => {
    allVals = allVals.concat(curves[s.key]);
  });
  const yMin = Math.min(...allVals) - 0.03;
  const yMax = Math.max(...allVals) + 0.03;

  const xScale = (i, len) => pad.left + (i / Math.max(len - 1, 1)) * (width - pad.left - pad.right);
  const yScale = (v) => pad.top + (1 - (v - yMin) / Math.max(yMax - yMin, 1e-5)) * (height - pad.top - pad.bottom);

  let gridLines = "";
  for (let g = 0; g <= 4; g++) {
    const val = yMin + (g / 4) * (yMax - yMin);
    const y = yScale(val).toFixed(1);
    gridLines += `<line x1="${pad.left}" y1="${y}" x2="${width - pad.right}" y2="${y}" stroke="#1c273d" stroke-width="1"/>`;
    gridLines += `<text x="${pad.left - 6}" y="${Number(y) + 4}" fill="#7f91ad" font-size="10" text-anchor="end">${val.toFixed(2)}</text>`;
  }

  let polylines = "";
  active.forEach((s) => {
    const arr = curves[s.key];
    const pts = arr.map((v, i) => `${xScale(i, arr.length).toFixed(1)},${yScale(v).toFixed(1)}`).join(" ");
    const dashAttr = s.dash ? `stroke-dasharray="${s.dash}"` : "";
    polylines += `<polyline fill="none" stroke="${s.color}" stroke-width="2.2" ${dashAttr} points="${pts}"/>`;
  });

  wrap.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" width="100%" height="100%" preserveAspectRatio="none">
      ${gridLines}
      ${polylines}
      <text x="${(width + pad.left) / 2}" y="${height - 5}" fill="#7f91ad" font-size="10.5" text-anchor="middle">Training Optimization Step (Team 20 Verification Window)</text>
    </svg>
  `;

  legend.innerHTML = active
    .map(
      (s) =>
        `<span class="legend-item"><span class="legend-swatch" style="background:${s.color}"></span>${s.label}</span>`
    )
    .join("");
}

function refreshYamlViewer() {
  const viewer = document.getElementById("yaml-viewer");
  if (yamlTab === "trial" && latestLiveSnap && latestLiveSnap.latest_trial_fuzzfile) {
    viewer.textContent = latestLiveSnap.latest_trial_fuzzfile;
  } else if (latestLiveSnap && latestLiveSnap.turnkey_fuzzfile) {
    viewer.textContent = latestLiveSnap.turnkey_fuzzfile;
  } else if (latestSummary && latestSummary.turnkey_fuzzfile) {
    viewer.textContent = latestSummary.turnkey_fuzzfile;
  }
}

function renderDashboard(data) {
  latestSummary = data;
  document.getElementById("pill-fz-ver").textContent = data.cluster.fuzzfile_version;
  document.getElementById("pill-def-id").textContent = data.cluster.definition_id || "team20";

  const sa = data.stage_a;
  const sb = data.stage_b;
  const sc = data.stage_c;

  const tunedPpl = sa.winning_summary.held_out_perplexity.toFixed(3);
  document.getElementById("kpi-ppl").textContent = `${sa.base_held_out_perplexity.toFixed(3)} → ${tunedPpl}`;
  document.getElementById("meta-stage-a").textContent =
    `Winner: ${sa.winning_label} · ${sa.total_trials_launched} trials (${sa.early_stopped_count} pruned) · ${sa.two_node_wall_speedup}x faster across SPARK39+40`;

  const baseTrainTok = sb.outcomes[0].summary.mean_tokens_per_sec;
  const winTrainOut = sb.outcomes.find((o) => o.label === sb.winning_label) || sb.outcomes[0];
  const bestTrainTok = winTrainOut.summary.mean_tokens_per_sec;
  document.getElementById("kpi-train-speed").textContent =
    `${Math.round(bestTrainTok).toLocaleString()} tok/s (${sb.overall_speedup_x}x)`;
  document.getElementById("meta-stage-b").textContent =
    `${Math.round(baseTrainTok).toLocaleString()} → ${Math.round(bestTrainTok).toLocaleString()} tok/s · sm_121 kernels alone: ${sb.single_node_kernel_speedup_x}x (1-node)`;

  const winServeOut = sc.outcomes.find((o) => o.label === sc.winning_label) || sc.outcomes[0];
  document.getElementById("kpi-serve-speed").textContent =
    `${Math.round(winServeOut.throughput_tokens_per_sec).toLocaleString()} tok/s (${sc.serving_speedup_x}x)`;
  document.getElementById("meta-stage-c").textContent =
    `Winner: ${sc.winning_label} · PPL +${winServeOut.perplexity_regression_pct.toFixed(2)}% · P99 ${winServeOut.p99_latency_ms.toFixed(0)} ms`;

  document.getElementById("meta-stage-d").textContent =
    `definitionId: ${data.cluster.definition_id} · VAST '${data.cluster.vast.persistent_volume_name}' · ${data.dataset_manifest.held_out_files.length} sealed held-out files`;

  document.getElementById("curve-badge").textContent =
    `Winner Δ = ${winTrainOut.max_loss_curve_delta.toFixed(4)} (≤ 0.025)`;
  document.getElementById("util-badge").textContent =
    `${winTrainOut.summary.hardware_utilization_pct.toFixed(1)}% Cluster Peak`;

  renderLossSvg(data.trial_loss_curves || {});

  const maxTok = Math.max(...sb.outcomes.map((o) => o.summary.mean_tokens_per_sec), 1);
  document.getElementById("train-bars").innerHTML = sb.outcomes
    .map((o) => {
      const tok = o.summary.mean_tokens_per_sec;
      const pct = Math.max(6, (tok / maxTok) * 100).toFixed(1);
      const isWin = o.label === sb.winning_label;
      const cls = !o.accepted ? "bar-fill bar-fill-rejected" : isWin ? "bar-fill bar-fill-winner" : "bar-fill";
      const statusTag = o.accepted
        ? isWin
          ? `<span class="tag-ok">WINNER</span>`
          : `<span class="tag-ok">OK (Δ ${o.max_loss_curve_delta.toFixed(4)})</span>`
        : `<span class="tag-reject">REJECTED (Δ ${o.max_loss_curve_delta.toFixed(3)})</span>`;
      return `
        <div class="bar-row">
          <div class="bar-top">
            <span><code>${o.label}</code> (${o.summary.world_size}N · ${o.summary.peak_mem_gb.toFixed(0)} GiB)</span>
            <span><strong>${Math.round(tok).toLocaleString()} tok/s</strong> &nbsp;${statusTag}</span>
          </div>
          <div class="bar-track"><div class="${cls}" style="width:${pct}%"></div></div>
        </div>
      `;
    })
    .join("");

  document.querySelector("#serving-table tbody").innerHTML = sc.outcomes
    .map((s) => {
      const isWin = s.label === sc.winning_label;
      const badge = s.accepted
        ? `<span class="tag-ok">${isWin ? "SELECTED" : "PASS"}</span>`
        : `<span class="tag-reject" title="${s.rejection_reasons.join("; ")}">REJECTED</span>`;
      return `
        <tr class="${isWin ? "row-winner" : ""}">
          <td><code>${s.label}</code></td>
          <td>${s.precision}</td>
          <td>${s.replicas}&times;TP${s.tensor_parallel} (B=${s.max_num_seqs})</td>
          <td><strong>${Math.round(s.throughput_tokens_per_sec).toLocaleString()}</strong></td>
          <td>${s.p99_latency_ms.toFixed(0)} ms</td>
          <td>+${s.perplexity_regression_pct.toFixed(2)}%</td>
          <td>${badge}</td>
        </tr>
      `;
    })
    .join("");

  document.querySelector("#kernel-table tbody").innerHTML = data.kernel_benchmarks
    .map(
      (k) => `
      <tr>
        <td><code>${k.kernel_name}</code></td>
        <td><code>${k.max_abs_error.toExponential(2)}</code></td>
        <td>${k.memory_reduction_ratio.toFixed(2)}&times;</td>
        <td><span class="tag-ok">${k.correctness_passed ? "PASS" : "FAIL"}</span></td>
      </tr>
    `
    )
    .join("");

  document.getElementById("advisor-rationales").innerHTML = data.advisor_explanations
    .map((e) => `<div class="advisor-item"><strong>[${e.stage}]</strong> ${e.rationale}</div>`)
    .join("");

  const audits = data.privacy_audit || [];
  const lastDigest = audits.length ? audits[audits.length - 1].payload_sha256.slice(0, 16) : "n/a";
  const fwBox = document.getElementById("firewall-audit-box");
  fwBox.classList.remove("blocked-alert");
  fwBox.innerHTML =
    `<strong>Privacy Schema Firewall Verified:</strong> ${audits.length} advisor payloads inspected · ` +
    `<code>contains_raw_data=false</code> on all outbound requests · Latest SHA-256: <code>${lastDigest}…</code>`;

  refreshYamlViewer();
}

function connectLiveStream() {
  const es = new EventSource("/api/events");
  let prevState = null;
  es.onmessage = async (msg) => {
    const snap = JSON.parse(msg.data);
    renderLiveTelemetry(snap);
    if (prevState === "running" && snap.state === "completed") {
      const summary = await fetchSummary();
      renderDashboard(summary);
      document.getElementById("btn-run").disabled = false;
      document.getElementById("btn-run").textContent = "Launch on team20";
    } else if (snap.state === "failed") {
      document.getElementById("btn-run").disabled = false;
      document.getElementById("btn-run").textContent = "Launch on team20";
    }
    prevState = snap.state;
  };
}

document.getElementById("btn-run").addEventListener("click", async () => {
  const btn = document.getElementById("btn-run");
  const profile = document.getElementById("profile-select").value;
  const backend = document.getElementById("backend-select").value;
  btn.disabled = true;
  btn.textContent = `Running (${backend})…`;
  const resp = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      profile,
      backend,
      fuzzfile_version: "v4",
      definition_id: "team20",
      async_mode: true,
    }),
  });
  if (!resp.ok) {
    btn.disabled = false;
    btn.textContent = "Launch on team20";
    const err = await resp.json();
    const errBanner = document.getElementById("live-error-banner");
    errBanner.textContent = err.detail || "Failed to start run";
    errBanner.classList.remove("hidden");
  }
});

document.getElementById("btn-probe").addEventListener("click", async () => {
  const resp = await fetch("/api/firewall-probe", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ inject_cpp_snippet: true }),
  });
  const res = await resp.json();
  const fwBox = document.getElementById("firewall-audit-box");
  if (res.blocked) {
    fwBox.classList.add("blocked-alert");
    fwBox.innerHTML =
      `<strong>Firewall Guard Triggered (Zero Egress):</strong> Blocked simulated C++ code injection before serialization — ` +
      `<code>${res.violation}</code> (Audit SHA-256: <code>${res.record.payload_sha256.slice(0, 16)}…</code>)`;
  }
});

document.getElementById("btn-tab-turnkey").addEventListener("click", () => {
  yamlTab = "turnkey";
  document.getElementById("btn-tab-turnkey").classList.add("active-tab");
  document.getElementById("btn-tab-trial").classList.remove("active-tab");
  refreshYamlViewer();
});

document.getElementById("btn-tab-trial").addEventListener("click", () => {
  yamlTab = "trial";
  document.getElementById("btn-tab-trial").classList.add("active-tab");
  document.getElementById("btn-tab-turnkey").classList.remove("active-tab");
  refreshYamlViewer();
});

document.getElementById("btn-copy-yaml").addEventListener("click", async () => {
  const text = document.getElementById("yaml-viewer").textContent;
  await navigator.clipboard.writeText(text);
  const btn = document.getElementById("btn-copy-yaml");
  btn.textContent = "Copied!";
  setTimeout(() => (btn.textContent = "Copy YAML"), 1400);
});

(async () => {
  await initBackendDefault();
  const [summary, liveSnap] = await Promise.all([fetchSummary(), fetchLiveStatus()]);
  renderDashboard(summary);
  renderLiveTelemetry(liveSnap);
  setTimeout(connectLiveStream, 600);
})();

let latestSummary = null;
let latestLiveSnap = null;
let yamlMode = "turnkey";

async function fetchSummary() {
  const resp = await fetch("/api/summary");
  return await resp.json();
}

async function fetchLiveStatus() {
  const resp = await fetch("/api/live-status");
  return await resp.json();
}

async function initClusterDefaults() {
  try {
    const resp = await fetch("/api/discover");
    const info = await resp.json();
    if (info.cluster_config && info.cluster_config.definition_id) {
      document.getElementById("team-input").value = info.cluster_config.definition_id;
    }
    const sel = document.getElementById("backend-select");
    if (info.discovery_report && info.discovery_report.fuzzball_cli_available) {
      sel.value = "fuzzball_mcp";
    } else {
      sel.value = "simulator";
    }
  } catch (_) {}
}

function updateStepper(state, currentStage) {
  const order = ["stage_a_recipe", "stage_b_training_perf", "stage_c_inference_perf", "stage_d_emit"];
  const curIdx = order.indexOf(currentStage);
  document.querySelectorAll("#stage-stepper .step").forEach((el, idx) => {
    el.classList.remove("step-done", "step-active");
    if (state === "completed") {
      el.classList.add("step-done");
    } else if (state === "running") {
      if (curIdx > idx) el.classList.add("step-done");
      else if (curIdx === idx || (curIdx === -1 && idx === 0)) el.classList.add("step-active");
    }
  });
}

function renderNodeCards(snap) {
  const wrap = document.getElementById("node-cards");
  const nodes = (snap && snap.nodes) || [];
  wrap.innerHTML = nodes
    .map((n) => {
      const running = n.state === "RUNNING";
      const wfLine = running
        ? `ACTIVE: ${n.active_trial_id} (${String(n.active_workflow_id || "").slice(0, 8)}…)`
        : n.last_trial_id
        ? `LAST: ${n.last_trial_id} (${n.trials_completed} completed)`
        : "Idle · Ready";
      return `
        <div class="node-card">
          <div class="node-card-head">
            <span>${n.hostname}</span>
            <span class="${running ? "badge-win" : "badge-pass"}">${running ? "RUNNING" : "READY"}</span>
          </div>
          <div class="node-meta">1&times; ${n.gpu_product} (sm_121) &middot; ${n.alloc_cores}/${n.physical_cores}c &middot; ${n.alloc_memory_gib}/${n.physical_memory_gib} GiB</div>
          <div class="node-meta">${wfLine}</div>
        </div>
      `;
    })
    .join("");
}

function renderLiveTelemetry(snap) {
  latestLiveSnap = snap;
  const stEl = document.getElementById("live-run-state");
  stEl.textContent = `${String(snap.state || "idle").toUpperCase()} · ${snap.backend}`;
  stEl.className =
    snap.state === "running"
      ? "status-tag tag-running"
      : snap.state === "completed"
      ? "status-tag tag-ok"
      : snap.state === "failed"
      ? "status-tag tag-fault"
      : "status-tag";

  const activeWfs = snap.active_workflows || [];
  const wfMeta = document.getElementById("live-wf-meta");
  if (activeWfs.length > 0) {
    wfMeta.textContent = `WF ${String(activeWfs[0].workflow_id).slice(0, 12)}… · use: ${snap.definition_id}`;
  } else {
    wfMeta.textContent = `VAST use: ${snap.definition_id || "team3"} · GB10 sm_121`;
  }

  document.getElementById("live-stage-label").textContent = snap.stage_label || "Ready";
  updateStepper(snap.state, snap.current_stage);

  const errBanner = document.getElementById("live-error-banner");
  if (snap.error) {
    errBanner.textContent = snap.error;
    errBanner.classList.remove("hidden");
  } else {
    errBanner.classList.add("hidden");
  }

  renderNodeCards(snap);

  const feed = document.getElementById("live-event-feed");
  feed.innerHTML = (snap.events || [])
    .slice()
    .reverse()
    .map((ev) => `<div><span class="ts">${ev.time_utc}</span>${ev.message}</div>`)
    .join("");

  if (snap.state === "running" && snap.live_curves && Object.keys(snap.live_curves).length > 0) {
    renderLossSvg(snap.live_curves);
  }
  refreshYamlViewer();
}

function renderLossSvg(curves) {
  const wrap = document.getElementById("loss-chart-wrap");
  const legend = document.getElementById("loss-legend");
  const seriesDefs = [
    { key: "stageb-00-pytorch-baseline-1node", label: "PyTorch Baseline (1N)", color: "#e2e8e2", dash: "" },
    { key: "stageb-04-sm121-all-kernels-2node-ddp", label: "sm_121 Kernels + 2N DDP (Winner)", color: "#00ff66", dash: "6,3" },
    { key: "stageb-05-oversized-global-batch-2node", label: "Batch 96 Drift (Vetoed)", color: "#ff453a", dash: "2,3" },
  ];

  const active = seriesDefs.filter((s) => Array.isArray(curves[s.key]) && curves[s.key].length > 1);
  if (active.length === 0) {
    const keys = Object.keys(curves).slice(-3);
    const palette = ["#e2e8e2", "#00ff66", "#ffb000"];
    keys.forEach((k, i) => active.push({ key: k, label: k, color: palette[i % palette.length], dash: "" }));
  }
  if (active.length === 0) return;

  const width = 580;
  const height = 192;
  const pad = { top: 12, right: 14, bottom: 22, left: 42 };
  let vals = [];
  active.forEach((s) => (vals = vals.concat(curves[s.key])));
  const yMin = Math.min(...vals) - 0.03;
  const yMax = Math.max(...vals) + 0.03;

  const xScale = (i, len) => pad.left + (i / Math.max(len - 1, 1)) * (width - pad.left - pad.right);
  const yScale = (v) => pad.top + (1 - (v - yMin) / Math.max(yMax - yMin, 1e-5)) * (height - pad.top - pad.bottom);

  let grid = "";
  for (let g = 0; g <= 4; g++) {
    const val = yMin + (g / 4) * (yMax - yMin);
    const y = yScale(val).toFixed(1);
    grid += `<line x1="${pad.left}" y1="${y}" x2="${width - pad.right}" y2="${y}" stroke="#162119" stroke-width="1"/>`;
    grid += `<text x="${pad.left - 6}" y="${Number(y) + 3.5}" fill="#7d9181" font-family="monospace" font-size="10" text-anchor="end">${val.toFixed(2)}</text>`;
  }

  let lines = "";
  active.forEach((s) => {
    const arr = curves[s.key];
    const pts = arr.map((v, i) => `${xScale(i, arr.length).toFixed(1)},${yScale(v).toFixed(1)}`).join(" ");
    const dash = s.dash ? `stroke-dasharray="${s.dash}"` : "";
    lines += `<polyline fill="none" stroke="${s.color}" stroke-width="2" ${dash} points="${pts}"/>`;
  });

  wrap.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" width="100%" height="100%" preserveAspectRatio="none">
      ${grid}
      ${lines}
      <text x="${(width + pad.left) / 2}" y="${height - 4}" fill="#7d9181" font-family="monospace" font-size="10" text-anchor="middle">Training Step</text>
    </svg>
  `;
  legend.innerHTML = active
    .map((s) => `<span class="legend-item"><span class="legend-swatch" style="background:${s.color}"></span>${s.label}</span>`)
    .join("");
}

function refreshYamlViewer() {
  const viewer = document.getElementById("yaml-viewer");
  if (yamlMode === "trial" && latestLiveSnap && latestLiveSnap.latest_trial_fuzzfile) {
    viewer.textContent = latestLiveSnap.latest_trial_fuzzfile;
  } else if (latestLiveSnap && latestLiveSnap.turnkey_fuzzfile) {
    viewer.textContent = latestLiveSnap.turnkey_fuzzfile;
  } else if (latestSummary && latestSummary.turnkey_fuzzfile) {
    viewer.textContent = latestSummary.turnkey_fuzzfile;
  }
}

function renderDashboard(data) {
  latestSummary = data;
  const sa = data.stage_a;
  const sb = data.stage_b;
  const sc = data.stage_c;

  const tunedPpl = sa.winning_summary.held_out_perplexity.toFixed(2);
  document.getElementById("kpi-ppl").textContent = `${sa.base_held_out_perplexity.toFixed(2)} → ${tunedPpl}`;
  document.getElementById("meta-stage-a").textContent = `${sa.winning_label} (${sa.early_stopped_count} pruned)`;

  const winTrain = sb.outcomes.find((o) => o.label === sb.winning_label) || sb.outcomes[0];
  document.getElementById("kpi-train-speed").textContent =
    `${Math.round(winTrain.summary.mean_tokens_per_sec).toLocaleString()} tok/s`;
  document.getElementById("meta-stage-b").textContent =
    `${sb.overall_speedup_x}x total · ${sb.single_node_kernel_speedup_x}x 1N kernels`;

  const winServe = sc.outcomes.find((o) => o.label === sc.winning_label) || sc.outcomes[0];
  document.getElementById("kpi-serve-speed").textContent =
    `${Math.round(winServe.throughput_tokens_per_sec).toLocaleString()} tok/s`;
  document.getElementById("meta-stage-c").textContent =
    `${sc.serving_speedup_x}x (${winServe.precision} · P99 ${winServe.p99_latency_ms.toFixed(0)}ms)`;

  document.getElementById("curve-badge").textContent =
    `Max Δ = ${winTrain.max_loss_curve_delta.toFixed(4)} ≤ 0.025`;
  document.getElementById("util-badge").textContent =
    `${winTrain.summary.hardware_utilization_pct.toFixed(1)}% Peak · use: ${data.cluster.vast.provisioner}`;

  renderLossSvg(data.trial_loss_curves || {});

  // Unified concise Trials & Winners table across Stage A winner + Stage B + Stage C
  const rows = [];
  rows.push({
    stage: "A · Recipe",
    cfg: sa.winning_label,
    qual: `PPL ${sa.winning_summary.held_out_perplexity.toFixed(3)}`,
    tok: Math.round(sa.winning_summary.mean_tokens_per_sec).toLocaleString(),
    win: true,
    ok: true,
  });
  sb.outcomes.forEach((o) => {
    rows.push({
      stage: `B · Train (${o.summary.world_size}N)`,
      cfg: o.label,
      qual: `Δ ${o.max_loss_curve_delta.toFixed(4)}`,
      tok: Math.round(o.summary.mean_tokens_per_sec).toLocaleString(),
      win: o.label === sb.winning_label,
      ok: o.accepted,
    });
  });
  sc.outcomes.forEach((s) => {
    rows.push({
      stage: `C · ${s.precision}`,
      cfg: `${s.label} (${s.replicas}xTP${s.tensor_parallel})`,
      qual: `PPL +${s.perplexity_regression_pct.toFixed(2)}% · ${s.p99_latency_ms.toFixed(0)}ms`,
      tok: Math.round(s.throughput_tokens_per_sec).toLocaleString(),
      win: s.label === sc.winning_label,
      ok: s.accepted,
    });
  });

  document.querySelector("#trials-table tbody").innerHTML = rows
    .map((r) => {
      const badge = !r.ok
        ? `<span class="badge-veto">VETO</span>`
        : r.win
        ? `<span class="badge-win">WINNER</span>`
        : `<span class="badge-pass">PASS</span>`;
      return `
        <tr class="${r.win ? "row-win" : ""}">
          <td>${r.stage}</td>
          <td>${r.cfg}</td>
          <td>${r.qual}</td>
          <td><strong>${r.tok}</strong></td>
          <td>${badge}</td>
        </tr>
      `;
    })
    .join("");

  document.querySelector("#kernel-table tbody").innerHTML = data.kernel_benchmarks
    .map(
      (k) => `
      <tr>
        <td>${k.kernel_name}</td>
        <td>${k.max_abs_error.toExponential(2)}</td>
        <td>${k.memory_reduction_ratio.toFixed(2)}&times;</td>
        <td><span class="badge-win">${k.correctness_passed ? "PASS" : "FAIL"}</span></td>
      </tr>
    `
    )
    .join("");

  document.getElementById("advisor-rationales").innerHTML = data.advisor_explanations
    .map((e) => `<div><strong>[${e.stage}]</strong> ${e.rationale}</div>`)
    .join("");

  const audits = data.privacy_audit || [];
  const lastDigest = audits.length ? audits[audits.length - 1].payload_sha256.slice(0, 24) : "00";
  const fwBox = document.getElementById("firewall-audit-box");
  fwBox.classList.remove("tripped");
  fwBox.innerHTML =
    `<strong>SCHEMA FIREWALL VERIFIED:</strong> ${audits.length} outbound frames audited &middot; ` +
    `raw_data=false &middot; sha256=<code>0x${lastDigest}…</code>`;

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
      document.getElementById("btn-run").textContent = "[ LAUNCH ]";
    } else if (snap.state === "failed") {
      document.getElementById("btn-run").disabled = false;
      document.getElementById("btn-run").textContent = "[ LAUNCH ]";
    }
    prevState = snap.state;
  };
}

// Drawer tabs
document.querySelectorAll(".tab-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    const target = btn.dataset.drawer;
    ["yaml", "cluster", "audit"].forEach((id) => {
      document.getElementById(`pane-${id}`).classList.toggle("hidden", id !== target);
    });
    document.getElementById("yaml-actions").classList.toggle("hidden", target !== "yaml");
  });
});

document.getElementById("btn-yaml-mode").addEventListener("click", (ev) => {
  yamlMode = yamlMode === "turnkey" ? "trial" : "turnkey";
  ev.currentTarget.textContent = yamlMode === "turnkey" ? "VIEW: TURNKEY" : "VIEW: LATEST TRIAL";
  refreshYamlViewer();
});

document.getElementById("btn-copy-yaml").addEventListener("click", async (ev) => {
  const btn = ev.currentTarget;
  await navigator.clipboard.writeText(document.getElementById("yaml-viewer").textContent);
  btn.textContent = "[ COPIED ]";
  setTimeout(() => (btn.textContent = "[ COPY YAML ]"), 1200);
});

document.getElementById("btn-run").addEventListener("click", async () => {
  const btn = document.getElementById("btn-run");
  const profile = document.getElementById("profile-select").value;
  const backend = document.getElementById("backend-select").value;
  const teamId = document.getElementById("team-input").value.trim() || "team3";
  btn.disabled = true;
  btn.textContent = "[ RUNNING… ]";
  const resp = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      profile,
      backend,
      fuzzfile_version: "v4",
      definition_id: teamId,
      async_mode: true,
    }),
  });
  if (!resp.ok) {
    btn.disabled = false;
    btn.textContent = "[ LAUNCH ]";
    const err = await resp.json();
    const errBanner = document.getElementById("live-error-banner");
    errBanner.textContent = err.detail || "Dispatch failed";
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
    fwBox.classList.add("tripped");
    fwBox.innerHTML =
      `<strong>INTERCEPTED (0 B EGRESS):</strong> Blocked raw C++ snippet — ` +
      `<code>${res.violation}</code> [sha256=<code>0x${res.record.payload_sha256.slice(0, 18)}…</code>]`;
  }
});

(async () => {
  await initClusterDefaults();
  const [summary, liveSnap] = await Promise.all([fetchSummary(), fetchLiveStatus()]);
  renderDashboard(summary);
  renderLiveTelemetry(liveSnap);
  setTimeout(connectLiveStream, 600);
})();

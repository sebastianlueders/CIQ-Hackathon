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

function asciiBar(ratio, slots = 16) {
  const filled = Math.max(0, Math.min(slots, Math.round(ratio * slots)));
  return "[" + "█".repeat(filled) + "░".repeat(slots - filled) + "]";
}

function renderNodeCards(liveSnap) {
  const wrap = document.getElementById("node-cards");
  const nodes = (liveSnap && liveSnap.nodes) || [];
  wrap.innerHTML = nodes
    .map((n, idx) => {
      const running = n.state === "RUNNING";
      const coreRatio = n.alloc_cores / Math.max(n.physical_cores, 1);
      const wfText = running
        ? `EXEC :: ${n.active_trial_id} // WF=${String(n.active_workflow_id || "").slice(0, 12)}`
        : n.last_trial_id
        ? `IDLE :: LAST=${n.last_trial_id} // COMPLETED=${n.trials_completed}`
        : "IDLE :: AWAITING_SUBSTRATE_ALLOC";
      return `
        <div class="node-blade ${running ? "blade-active" : ""}">
          <div class="blade-head">
            <div>
              <div class="blade-host">[NODE_0${idx} :: ${n.hostname}]</div>
              <div class="blade-addr">SUBSTRATE // ${n.node_id}</div>
            </div>
            <span class="${running ? "verdict-win" : "verdict-pass"}">${n.state}</span>
          </div>
          <div class="blade-kv"><span>GPU_DEV</span><span>1&times; ${n.gpu_product} [sm_121]</span></div>
          <div class="blade-kv"><span>CPU_SVE</span><span>${n.alloc_cores}/${n.physical_cores}c [${n.arch_device}] ${asciiBar(coreRatio, 10)}</span></div>
          <div class="blade-kv"><span>UNIFIED_MEM</span><span>${n.alloc_memory_gib}/${n.physical_memory_gib} GiB</span></div>
          <div class="blade-alloc">${wfText}</div>
        </div>
      `;
    })
    .join("");
}

function renderLiveTelemetry(snap) {
  latestLiveSnap = snap;
  document.getElementById("pill-def-id").textContent = snap.definition_id || "team20";
  document.getElementById("live-stage-label").textContent =
    `> ${String(snap.stage_label || "READY ON TEAM20").toUpperCase()}`;

  const stBadge = document.getElementById("live-run-state");
  const stUpper = String(snap.state || "idle").toUpperCase();
  stBadge.textContent = `STATE::${stUpper} [${String(snap.backend || "").toUpperCase()}]`;
  stBadge.className =
    snap.state === "running"
      ? "state-flag flag-amber"
      : snap.state === "failed"
      ? "state-flag flag-fault"
      : "state-flag";

  const errBanner = document.getElementById("live-error-banner");
  if (snap.error) {
    errBanner.textContent = `FAULT // ${snap.error}`;
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
        `<div class="tty-line"><span class="ts">${ev.time_utc}Z</span><span class="prompt">&gt;</span>${ev.message}</div>`
    )
    .join("");

  if (snap.state === "running" && snap.live_curves && Object.keys(snap.live_curves).length > 0) {
    renderLossSvg(snap.live_curves);
  }
  if (snap.state === "running" && snap.live_firewall && snap.live_firewall.length > 0) {
    const lastRec = snap.live_firewall[snap.live_firewall.length - 1].record || {};
    const fwBox = document.getElementById("firewall-audit-box");
    fwBox.innerHTML =
      `<strong>SEAL_OK // ZERO_RAW_EGRESS:</strong> FRAMES=${snap.live_firewall.length} &middot; ` +
      `RAW_DATA=FALSE &middot; SHA256=<code>0x${String(lastRec.payload_sha256 || "").slice(0, 24)}…</code>`;
  }
  refreshYamlViewer();
}

function renderLossSvg(curves) {
  const wrap = document.getElementById("loss-chart-wrap");
  const legend = document.getElementById("loss-legend");
  const seriesDefs = [
    { key: "stageb-00-pytorch-baseline-1node", label: "REF :: PYTORCH_FP32_1N [SPARK39]", color: "#d8dfd8", dash: "" },
    { key: "stageb-04-sm121-all-kernels-2node-ddp", label: "OPT :: SM121_KERNELS_2N_DDP [LOCKED]", color: "#00ff66", dash: "6,3" },
    { key: "stageb-05-oversized-global-batch-2node", label: "REJ :: BATCH_96_DRIFT [VETOED]", color: "#ff3b30", dash: "2,3" },
  ];

  const active = seriesDefs.filter((s) => Array.isArray(curves[s.key]) && curves[s.key].length > 1);
  if (active.length === 0) {
    const fallbackKeys = Object.keys(curves).slice(-3);
    const palette = ["#d8dfd8", "#00ff66", "#ffb000"];
    fallbackKeys.forEach((k, idx) => {
      active.push({ key: k, label: k.toUpperCase(), color: palette[idx % palette.length], dash: "" });
    });
  }
  if (active.length === 0) return;

  const width = 620;
  const height = 194;
  const pad = { top: 14, right: 16, bottom: 24, left: 46 };
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
    gridLines += `<line x1="${pad.left}" y1="${y}" x2="${width - pad.right}" y2="${y}" stroke="#172419" stroke-width="1"/>`;
    gridLines += `<text x="${pad.left - 6}" y="${Number(y) + 3.5}" fill="#748877" font-family="monospace" font-size="10" text-anchor="end">${val.toFixed(2)}</text>`;
  }

  let polylines = "";
  active.forEach((s) => {
    const arr = curves[s.key];
    const pts = arr.map((v, i) => `${xScale(i, arr.length).toFixed(1)},${yScale(v).toFixed(1)}`).join(" ");
    const dashAttr = s.dash ? `stroke-dasharray="${s.dash}"` : "";
    polylines += `<polyline fill="none" stroke="${s.color}" stroke-width="1.9" ${dashAttr} points="${pts}"/>`;
  });

  wrap.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" width="100%" height="100%" preserveAspectRatio="none">
      ${gridLines}
      <line x1="${pad.left}" y1="${pad.top}" x2="${pad.left}" y2="${height - pad.bottom}" stroke="#2e4733" stroke-width="1"/>
      <line x1="${pad.left}" y1="${height - pad.bottom}" x2="${width - pad.right}" y2="${height - pad.bottom}" stroke="#2e4733" stroke-width="1"/>
      ${polylines}
      <text x="${(width + pad.left) / 2}" y="${height - 5}" fill="#748877" font-family="monospace" font-size="10" text-anchor="middle">STEP_INDEX // STAGE_B EQUIVALENCE VERIFICATION WINDOW</text>
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
  document.getElementById("kpi-ppl").textContent = `${sa.base_held_out_perplexity.toFixed(3)} -> ${tunedPpl}`;
  document.getElementById("meta-stage-a").textContent =
    `WIN=${sa.winning_label} :: TRIALS=${sa.total_trials_launched} [PRUNED=${sa.early_stopped_count}] :: 2N=${sa.two_node_wall_speedup}x`;

  const baseTrainTok = sb.outcomes[0].summary.mean_tokens_per_sec;
  const winTrainOut = sb.outcomes.find((o) => o.label === sb.winning_label) || sb.outcomes[0];
  const bestTrainTok = winTrainOut.summary.mean_tokens_per_sec;
  document.getElementById("kpi-train-speed").textContent =
    `${Math.round(bestTrainTok).toLocaleString()} [${sb.overall_speedup_x}x]`;
  document.getElementById("meta-stage-b").textContent =
    `${Math.round(baseTrainTok).toLocaleString()}->${Math.round(bestTrainTok).toLocaleString()} tok/s :: 1N_KERNELS=${sb.single_node_kernel_speedup_x}x`;

  const winServeOut = sc.outcomes.find((o) => o.label === sc.winning_label) || sc.outcomes[0];
  document.getElementById("kpi-serve-speed").textContent =
    `${Math.round(winServeOut.throughput_tokens_per_sec).toLocaleString()} [${sc.serving_speedup_x}x]`;
  document.getElementById("meta-stage-c").textContent =
    `WIN=${sc.winning_label} :: PPL_+${winServeOut.perplexity_regression_pct.toFixed(2)}% :: P99=${winServeOut.p99_latency_ms.toFixed(0)}ms`;

  document.getElementById("meta-stage-d").textContent =
    `DEF_ID=${data.cluster.definition_id} :: VOL=${data.cluster.vast.persistent_volume_name} :: SEALED_FILES=${data.dataset_manifest.held_out_files.length}`;

  document.getElementById("curve-badge").textContent =
    `Δ_MAX = ${winTrainOut.max_loss_curve_delta.toFixed(4)} <= 0.0250`;
  document.getElementById("util-badge").textContent =
    `${winTrainOut.summary.hardware_utilization_pct.toFixed(1)}% HW_PEAK`;

  renderLossSvg(data.trial_loss_curves || {});

  const maxTok = Math.max(...sb.outcomes.map((o) => o.summary.mean_tokens_per_sec), 1);
  document.getElementById("train-bars").innerHTML = sb.outcomes
    .map((o) => {
      const tok = o.summary.mean_tokens_per_sec;
      const pct = Math.max(6, (tok / maxTok) * 100).toFixed(1);
      const isWin = o.label === sb.winning_label;
      const cls = !o.accepted
        ? "meter-fill meter-fill-fault"
        : isWin
        ? "meter-fill meter-fill-win"
        : "meter-fill";
      const verdict = o.accepted
        ? isWin
          ? `<span class="verdict-win">LOCKED_WIN</span>`
          : `<span class="verdict-pass">PASS [Δ=${o.max_loss_curve_delta.toFixed(4)}]</span>`
        : `<span class="verdict-fault">VETO [Δ=${o.max_loss_curve_delta.toFixed(3)}]</span>`;
      return `
        <div class="meter-row">
          <div class="meter-head">
            <span>${o.label} [${o.summary.world_size}N // ${o.summary.peak_mem_gb.toFixed(0)}GiB]</span>
            <span><strong>${Math.round(tok).toLocaleString()} tok/s</strong> ${verdict}</span>
          </div>
          <div class="meter-track"><div class="${cls}" style="width:${pct}%"></div></div>
        </div>
      `;
    })
    .join("");

  document.querySelector("#serving-table tbody").innerHTML = sc.outcomes
    .map((s) => {
      const isWin = s.label === sc.winning_label;
      const badge = s.accepted
        ? `<span class="${isWin ? "verdict-win" : "verdict-pass"}">${isWin ? "SELECTED" : "PASS"}</span>`
        : `<span class="verdict-fault" title="${s.rejection_reasons.join("; ")}">VETOED</span>`;
      return `
        <tr class="${isWin ? "row-armed" : ""}">
          <td>${s.label}</td>
          <td>${s.precision}</td>
          <td>${s.replicas}&times;TP${s.tensor_parallel} [B=${s.max_num_seqs}]</td>
          <td><strong>${Math.round(s.throughput_tokens_per_sec).toLocaleString()}</strong></td>
          <td>${s.p99_latency_ms.toFixed(0)}</td>
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
        <td>${k.kernel_name}</td>
        <td>${k.max_abs_error.toExponential(2)}</td>
        <td>${k.memory_reduction_ratio.toFixed(2)}&times;</td>
        <td><span class="verdict-win">${k.correctness_passed ? "EXACT" : "FAULT"}</span></td>
      </tr>
    `
    )
    .join("");

  document.getElementById("advisor-rationales").innerHTML = data.advisor_explanations
    .map((e) => `<div class="ledger-item"><strong>[${e.stage.toUpperCase()}]</strong> ${e.rationale}</div>`)
    .join("");

  const audits = data.privacy_audit || [];
  const lastDigest = audits.length ? audits[audits.length - 1].payload_sha256.slice(0, 28) : "00";
  const fwBox = document.getElementById("firewall-audit-box");
  fwBox.classList.remove("seal-tripped");
  fwBox.innerHTML =
    `<strong>CRYPTO_SEAL :: VERIFIED</strong> &middot; FRAMES_AUDITED=${audits.length} &middot; ` +
    `RAW_CODE_EGRESS=<code>FALSE</code> &middot; MERKLE_LEAF=<code>0x${lastDigest}…</code>`;

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
      document.getElementById("btn-run").textContent = "[ EXEC // TEAM20 ]";
    } else if (snap.state === "failed") {
      document.getElementById("btn-run").disabled = false;
      document.getElementById("btn-run").textContent = "[ EXEC // TEAM20 ]";
    }
    prevState = snap.state;
  };
}

document.getElementById("btn-run").addEventListener("click", async () => {
  const btn = document.getElementById("btn-run");
  const profile = document.getElementById("profile-select").value;
  const backend = document.getElementById("backend-select").value;
  btn.disabled = true;
  btn.textContent = `[ RUNNING :: ${backend.toUpperCase()} ]`;
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
    btn.textContent = "[ EXEC // TEAM20 ]";
    const err = await resp.json();
    const errBanner = document.getElementById("live-error-banner");
    errBanner.textContent = `FAULT // ${err.detail || "DISPATCH_REJECTED"}`;
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
    fwBox.classList.add("seal-tripped");
    fwBox.innerHTML =
      `<strong>FIREWALL_INTERCEPT // ZERO_BYTES_TRANSMITTED:</strong> VETOED C++ PAYLOAD PRE-SERIALIZATION &mdash; ` +
      `<code>${res.violation}</code> [SHA256=<code>0x${res.record.payload_sha256.slice(0, 20)}…</code>]`;
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
  btn.textContent = "[ YANKED_OK ]";
  setTimeout(() => (btn.textContent = "[ YANK // CLIPBOARD ]"), 1400);
});

(async () => {
  await initBackendDefault();
  const [summary, liveSnap] = await Promise.all([fetchSummary(), fetchLiveStatus()]);
  renderDashboard(summary);
  renderLiveTelemetry(liveSnap);
  setTimeout(connectLiveStream, 650);
})();

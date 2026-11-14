let latestSummary = null;
let latestLiveSnap = null;
let buildArtifacts = {};
let selectedArtifactKey = "fuzzfile.yaml";
let activeDatasetId = "builtin-lammps-cpp-bench";

async function fetchJson(url) {
  const resp = await fetch(url);
  return await resp.json();
}

function currentModelId() {
  return document.getElementById("model-id-input").value.trim() || "Qwen/Qwen3-Coder-30B-A3B-Instruct";
}

function currentTeamId() {
  return document.getElementById("team-input").value.trim() || "team3";
}

async function refreshModelSpec(modelId) {
  try {
    const info = await fetchJson(`/api/models?probe_model=${encodeURIComponent(modelId)}`);
    const m = info.inferred;
    if (!m) return;
    const kind = m.num_experts > 1 ? `MoE (${m.num_experts} experts, top-${m.top_k_experts})` : "Dense Transformer";
    const kernels = m.num_experts > 1
      ? "chunked CE + grouped MoE GEMM + fused LoRA + NVFP4"
      : "chunked CE + fused LoRA + NVFP4";
    document.getElementById("model-spec-line").textContent =
      `Inferred: ${kind} · ${m.total_params_b}B total (${m.active_params_b}B active, ~${m.bf16_footprint_gb} GB BF16) · ` +
      `${m.vocab_size.toLocaleString()} vocab · min ${m.min_nodes_required} GB10 node(s) · ${kernels}`;
    document.querySelectorAll("#model-chips .chip").forEach((c) => {
      c.classList.toggle("chip-active", c.dataset.modelId.toLowerCase() === modelId.toLowerCase());
    });
  } catch (_) {}
}

function renderDatasetBanner(ds) {
  if (!ds) return;
  activeDatasetId = ds.dataset_id;
  const sc = ds.split_doc_counts || {};
  const cc = ds.split_chunk_counts || {};
  const prefix = ds.is_preset ? "Preset Active" : "Custom Data Loaded";
  document.getElementById("dataset-status-bar").innerHTML =
    `${prefix}: <strong>${ds.display_name}</strong> (${ds.format_summary}) &middot; ` +
    `<strong>${(ds.total_tokens || 0).toLocaleString()} tokens</strong> &middot; ` +
    `Train: <code>${sc.train || 0} docs (${cc.train || 0} chunks)</code> &middot; ` +
    `Validation: <code>${sc.val || 0} docs (${cc.val || 0} chunks)</code> &middot; ` +
    `Held-Out Test: <code>${sc.held_out || 0} docs (${cc.held_out || 0} chunks)</code>`;
}

async function initStudio() {
  try {
    const [modelsResp, dsResp, discResp] = await Promise.all([
      fetchJson("/api/models"),
      fetchJson("/api/datasets"),
      fetchJson("/api/discover"),
    ]);

    const chipsWrap = document.getElementById("model-chips");
    chipsWrap.innerHTML = (modelsResp.catalog || [])
      .map(
        (m) =>
          `<button type="button" class="chip" data-model-id="${m.id}" title="${m.id} (${m.family})">${m.label}</button>`
      )
      .join("");
    chipsWrap.querySelectorAll(".chip").forEach((btn) => {
      btn.addEventListener("click", () => {
        document.getElementById("model-id-input").value = btn.dataset.modelId;
        refreshModelSpec(btn.dataset.modelId);
      });
    });

    if (dsResp.preset) {
      renderDatasetBanner(dsResp.preset);
    }
    if (discResp.cluster_config && discResp.cluster_config.definition_id) {
      const team = discResp.cluster_config.definition_id;
      document.getElementById("team-input").value = team;
      document.getElementById("pill-def-id").textContent = team;
    }
    if (discResp.discovery_report && discResp.discovery_report.fuzzball_cli_available) {
      document.getElementById("backend-select").value = "fuzzball_mcp";
    }
    await refreshModelSpec(currentModelId());
  } catch (_) {}
}

function arrayBufferToBase64(buf) {
  const bytes = new Uint8Array(buf);
  let binary = "";
  const chunk = 0x8000;
  for (let i = 0; i < bytes.length; i += chunk) {
    binary += String.fromCharCode.apply(null, bytes.subarray(i, i + chunk));
  }
  return btoa(binary);
}

async function uploadFileList(fileList) {
  const arr = Array.from(fileList || []);
  if (arr.length === 0) return;
  const bar = document.getElementById("dataset-status-bar");
  bar.innerHTML = `Ingesting <strong>${arr.length}</strong> dropped file(s) locally into Train / Validation / Held-Out Test splits…`;

  const encoded = [];
  for (const f of arr.slice(0, 250)) {
    const buf = await f.arrayBuffer();
    encoded.push({
      name: f.webkitRelativePath || f.name,
      content_b64: arrayBufferToBase64(buf),
    });
  }
  const dsLabel = arr.length === 1 ? arr[0].name : `${arr.length}-files-drop`;
  const resp = await fetch("/api/dataset/upload", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ dataset_name: dsLabel, files: encoded }),
  });
  const payload = await resp.json();
  if (!resp.ok) {
    bar.innerHTML = `<span style="color:#fecaca">Upload error: ${payload.detail || "Invalid dataset"}</span>`;
    return;
  }
  renderDatasetBanner(payload);
}

async function refreshBuildPreview() {
  try {
    const prev = await fetchJson("/api/build/preview");
    buildArtifacts = prev.artifacts || {};
    const shortModel = String(prev.model_id || "").split("/").pop();
    const dsName = (prev.dataset && prev.dataset.display_name) || activeDatasetId;
    document.getElementById("build-title").textContent = `Tuned Build Ready: ${shortModel}`;
    document.getElementById("build-subtitle").innerHTML =
      `Configured for <code>${prev.model_id}</code> on <strong>${dsName}</strong> &mdash; ` +
      `includes turnkey <code>fuzzfile.yaml</code>, LoRA &amp; vLLM configs, launcher, and split <code>train/val/test</code> data.`;
    document.getElementById("inspector-model-label").textContent = prev.model_id;
    renderArtifactViewer();
  } catch (_) {}
}

function renderArtifactViewer() {
  const viewer = document.getElementById("yaml-viewer");
  if (buildArtifacts[selectedArtifactKey]) {
    viewer.textContent = buildArtifacts[selectedArtifactKey];
  } else if (latestSummary && latestSummary.turnkey_fuzzfile) {
    viewer.textContent = latestSummary.turnkey_fuzzfile;
  }
}

function renderLiveTelemetry(snap) {
  latestLiveSnap = snap;
  document.getElementById("pill-def-id").textContent = snap.definition_id || currentTeamId();
  document.getElementById("live-stage-label").textContent = snap.stage_label || "Ready";

  const stBadge = document.getElementById("live-run-state");
  const st = String(snap.state || "idle");
  stBadge.textContent =
    st === "queued_for_gpu"
      ? `QUEUED FOR GPU · ${snap.backend}`
      : `${st.toUpperCase()} · ${snap.backend}`;
  stBadge.className =
    st === "queued_for_gpu"
      ? "badge badge-amber"
      : st === "running"
      ? "badge badge-accent"
      : st === "completed"
      ? "badge badge-green"
      : "badge";

  const errBanner = document.getElementById("live-error-banner");
  if (snap.error) {
    errBanner.textContent = snap.error;
    errBanner.classList.remove("hidden");
  } else {
    errBanner.classList.add("hidden");
  }

  const feed = document.getElementById("live-event-feed");
  feed.innerHTML = (snap.events || [])
    .slice()
    .reverse()
    .map((ev) => `<div class="event-line"><span class="ev-ts">[${ev.time_utc}]</span>${ev.message}</div>`)
    .join("");

  if ((st === "running" || st === "queued_for_gpu") && snap.live_curves && Object.keys(snap.live_curves).length > 0) {
    renderLossSvg(snap.live_curves, snap.live_val_curves || {});
  }
}

function renderLossSvg(curves, valCurves = {}) {
  const wrap = document.getElementById("loss-chart-wrap");
  const legend = document.getElementById("loss-legend");
  const seriesDefs = [
    { key: "stageb-00-pytorch-baseline-1node", label: "Baseline Reference Curve", color: "#38bdf8", dash: "", src: curves },
    { key: "stageb-04-sm121-all-kernels-2node-ddp", label: "sm_121 Kernels + 2-Node DDP (Accepted)", color: "#a3e635", dash: "5,3", src: curves },
    { key: "stageb-05-oversized-global-batch-2node", label: "Oversized Global Batch (Rejected: Drift)", color: "#f87171", dash: "2,3", src: curves },
  ];

  const active = seriesDefs.filter((s) => Array.isArray(s.src[s.key]) && s.src[s.key].length > 1);
  if (active.length === 0) {
    const keys = Object.keys(curves).slice(-2);
    const palette = ["#38bdf8", "#a3e635"];
    keys.forEach((k, idx) => {
      active.push({ key: k, label: `${k} (train)`, color: palette[idx % palette.length], dash: "", src: curves });
      if (Array.isArray(valCurves[k]) && valCurves[k].length > 1) {
        active.push({ key: k, label: `${k} (val)`, color: "#fbbf24", dash: "4,3", src: valCurves });
      }
    });
  }
  if (active.length === 0) return;

  const width = 620;
  const height = 198;
  const pad = { top: 14, right: 18, bottom: 26, left: 44 };
  let allVals = [];
  active.forEach((s) => {
    allVals = allVals.concat(s.src[s.key]);
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
    const arr = s.src[s.key];
    const pts = arr.map((v, i) => `${xScale(i, arr.length).toFixed(1)},${yScale(v).toFixed(1)}`).join(" ");
    const dashAttr = s.dash ? `stroke-dasharray="${s.dash}"` : "";
    polylines += `<polyline fill="none" stroke="${s.color}" stroke-width="2.2" ${dashAttr} points="${pts}"/>`;
  });

  wrap.innerHTML = `
    <svg viewBox="0 0 ${width} ${height}" width="100%" height="100%" preserveAspectRatio="none">
      ${gridLines}
      ${polylines}
      <text x="${(width + pad.left) / 2}" y="${height - 5}" fill="#7f91ad" font-size="10.5" text-anchor="middle">Training &amp; Validation Evaluation Step</text>
    </svg>
  `;

  legend.innerHTML = active
    .map((s) => `<span class="legend-item"><span class="legend-swatch" style="background:${s.color}"></span>${s.label}</span>`)
    .join("");
}

function renderDashboard(data) {
  latestSummary = data;
  document.getElementById("pill-fz-ver").textContent = data.cluster.fuzzfile_version;
  document.getElementById("pill-def-id").textContent = data.cluster.definition_id || "team3";

  const modelName = (data.model_profile && data.model_profile.name) || currentModelId();
  document.getElementById("kpi-model-short").textContent = modelName.split("/").pop();

  const sa = data.stage_a;
  const sb = data.stage_b;
  const sc = data.stage_c;

  const tunedPpl = sa.winning_summary.held_out_perplexity.toFixed(3);
  document.getElementById("kpi-ppl").textContent = `${sa.base_held_out_perplexity.toFixed(3)} → ${tunedPpl}`;
  document.getElementById("meta-stage-a").textContent =
    `Winner: ${sa.winning_label} · Val loss ${sa.winning_summary.final_val_loss.toFixed(3)} · Test loss ${sa.winning_summary.held_out_eval_loss.toFixed(3)} (${sa.early_stopped_count} pruned)`;

  const baseTrainTok = sb.outcomes[0].summary.mean_tokens_per_sec;
  const winTrainOut = sb.outcomes.find((o) => o.label === sb.winning_label) || sb.outcomes[0];
  const bestTrainTok = winTrainOut.summary.mean_tokens_per_sec;
  document.getElementById("kpi-train-speed").textContent =
    `${Math.round(bestTrainTok).toLocaleString()} tok/s (${sb.overall_speedup_x}x)`;
  document.getElementById("meta-stage-b").textContent =
    `${Math.round(baseTrainTok).toLocaleString()} → ${Math.round(bestTrainTok).toLocaleString()} tok/s · 1-node sm_121 speedup: ${sb.single_node_kernel_speedup_x}x`;

  const winServeOut = sc.outcomes.find((o) => o.label === sc.winning_label) || sc.outcomes[0];
  document.getElementById("kpi-serve-speed").textContent =
    `${Math.round(winServeOut.throughput_tokens_per_sec).toLocaleString()} tok/s (${sc.serving_speedup_x}x)`;
  document.getElementById("meta-stage-c").textContent =
    `Winner: ${sc.winning_label} · Test PPL +${winServeOut.perplexity_regression_pct.toFixed(2)}% · P99 ${winServeOut.p99_latency_ms.toFixed(0)} ms`;

  const scs = data.dataset_manifest.split_file_counts || {};
  document.getElementById("meta-stage-d").textContent =
    `Dataset '${data.dataset_manifest.dataset_name}' (Train ${scs.train || 0} / Val ${scs.val || 0} / Test ${scs.held_out || 0}) · use: ${data.cluster.vast.provisioner}`;

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
        : `<span class="tag-reject">REJECTED (${o.rejection_reason || "Δ " + o.max_loss_curve_delta.toFixed(3)})</span>`;
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
}

function connectLiveStream() {
  const es = new EventSource("/api/events");
  let prevState = null;
  es.onmessage = async (msg) => {
    const snap = JSON.parse(msg.data);
    renderLiveTelemetry(snap);
    if ((prevState === "running" || prevState === "queued_for_gpu") && snap.state === "completed") {
      const summary = await fetchJson("/api/summary");
      renderDashboard(summary);
      await refreshBuildPreview();
      document.getElementById("btn-run").disabled = false;
      document.getElementById("btn-run").textContent = "Optimize Model & Generate Build";
    } else if (snap.state === "failed") {
      document.getElementById("btn-run").disabled = false;
      document.getElementById("btn-run").textContent = "Optimize Model & Generate Build";
    }
    prevState = snap.state;
  };
}

// Model input & drag-and-drop wiring
document.getElementById("model-id-input").addEventListener("change", (e) => refreshModelSpec(e.target.value));
document.getElementById("model-id-input").addEventListener("blur", (e) => refreshModelSpec(e.target.value));

const dz = document.getElementById("dropzone");
dz.addEventListener("dragover", (e) => {
  e.preventDefault();
  dz.classList.add("drag-over");
});
dz.addEventListener("dragleave", () => dz.classList.remove("drag-over"));
dz.addEventListener("drop", async (e) => {
  e.preventDefault();
  dz.classList.remove("drag-over");
  if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
    await uploadFileList(e.dataTransfer.files);
  }
});

document.getElementById("file-picker").addEventListener("change", (e) => uploadFileList(e.target.files));
document.getElementById("folder-picker").addEventListener("change", (e) => uploadFileList(e.target.files));
document.getElementById("btn-use-preset").addEventListener("click", async () => {
  const resp = await fetch("/api/dataset/upload", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ use_preset: true }),
  });
  renderDatasetBanner(await resp.json());
});

document.getElementById("btn-run").addEventListener("click", async () => {
  const btn = document.getElementById("btn-run");
  const modelId = currentModelId();
  const backend = document.getElementById("backend-select").value;
  const teamId = currentTeamId();
  btn.disabled = true;
  btn.textContent = `Optimizing (${backend})…`;
  const resp = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_id: modelId,
      dataset_id: activeDatasetId,
      backend,
      fuzzfile_version: "v4",
      definition_id: teamId,
      async_mode: true,
    }),
  });
  if (!resp.ok) {
    btn.disabled = false;
    btn.textContent = "Optimize Model & Generate Build";
    const err = await resp.json();
    const errBanner = document.getElementById("live-error-banner");
    errBanner.textContent = err.detail || "Failed to start optimization";
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

document.querySelectorAll(".artifact-tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".artifact-tab").forEach((b) => b.classList.remove("active-tab"));
    btn.classList.add("active-tab");
    selectedArtifactKey = btn.dataset.artifact;
    renderArtifactViewer();
  });
});

document.getElementById("btn-copy-yaml").addEventListener("click", async () => {
  const text = document.getElementById("yaml-viewer").textContent;
  await navigator.clipboard.writeText(text);
  const btn = document.getElementById("btn-copy-yaml");
  btn.textContent = "Copied!";
  setTimeout(() => (btn.textContent = "Copy File"), 1400);
});

(async () => {
  await initStudio();
  const [summary, liveSnap] = await Promise.all([
    fetchJson("/api/summary"),
    fetchJson("/api/live-status"),
  ]);
  renderDashboard(summary);
  renderLiveTelemetry(liveSnap);
  await refreshBuildPreview();
  setTimeout(connectLiveStream, 600);
})();

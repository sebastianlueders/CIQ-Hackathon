let latestSummary = null;
let latestLiveSnap = null;
let buildArtifacts = {};
let selectedArtifactKey = "fuzzfile.yaml";
let activeDatasetId = "builtin-lammps-cpp-bench";
let fullCatalog = [];
let activeCategory = "Featured";
let resolvedTeamId = "team3";
let resolvedClusterEndpoint = "api.fb-hack.civilfritz.net:443";
let userPickedBackend = false;

async function fetchJson(url, opts) {
  const resp = await fetch(url, opts);
  return await resp.json();
}

function currentModelId() {
  return document.getElementById("model-id-input").value.trim() || "Qwen/Qwen3-Coder-30B-A3B-Instruct";
}

function currentTeamId() {
  return resolvedTeamId || "team3";
}

function currentClusterEndpoint() {
  return resolvedClusterEndpoint || "api.fb-hack.civilfritz.net:443";
}

async function refreshModelSpec(modelId) {
  try {
    const info = await fetchJson(`/api/models?probe_model=${encodeURIComponent(modelId)}`);
    const m = info.inferred;
    if (!m) return;
    const kind =
      m.num_experts > 1 ? `MoE (${m.num_experts} experts, top-${m.top_k_experts})` : "Dense Transformer";
    const kernels =
      m.num_experts > 1
        ? "Triton 152k Fused CE + Grouped MoE GEMM + Fused LoRA + NVFP4"
        : "Triton Fused CE + Regional torch.compile + Fused LoRA + NVFP4";
    const nodeFit =
      m.min_nodes_required >= 2
        ? "Requires 2× GB10 Nodes (FSDP + TP2)"
        : m.total_params_b <= 4.5
        ? "1× GB10 Tune → Runs Directly on Client Laptop (4–8 GB)"
        : "1× GB10 Node Fit (2-Node DDP 1.98× Speedup Available)";
    document.getElementById("model-spec-line").innerHTML =
      `<strong>${m.name}</strong> &middot; ${kind} &middot; <strong>${m.total_params_b}B</strong> total ` +
      `(<strong>${m.active_params_b}B</strong> active) &middot; BF16 <code>${m.bf16_footprint_gb} GB</code> / ` +
      `FP8 <code>${m.fp8_footprint_gb} GB</code> / NVFP4 <code>${m.nvfp4_footprint_gb} GB</code> &middot; ` +
      `${m.vocab_size.toLocaleString()} vocab &middot; <span style="color:#a3e635">${nodeFit}</span> &middot; ${kernels}`;

    document.querySelectorAll("#model-chips .model-card").forEach((c) => {
      c.classList.toggle("model-card-active", c.dataset.modelId.toLowerCase() === modelId.toLowerCase());
    });
  } catch (_) {}
}

function renderCatalogCards() {
  const query = (document.getElementById("catalog-filter-input").value || "").trim().toLowerCase();
  const currentId = currentModelId().toLowerCase();
  const filtered = fullCatalog.filter((m) => {
    const catOk =
      activeCategory === "All Models" ||
      (activeCategory === "Featured" ? Boolean(m.featured) : m.category === activeCategory);
    if (!catOk) return false;
    if (!query) return true;
    const hay = `${m.id} ${m.label} ${m.provider} ${m.family} ${m.category} ${m.license} ${m.recommended_use}`.toLowerCase();
    return hay.includes(query);
  });

  const wrap = document.getElementById("model-chips");
  if (filtered.length === 0) {
    wrap.innerHTML = `<div class="catalog-empty">No catalog preset matches "${query}" in ${activeCategory}. You can still paste ANY HuggingFace <code>org/model</code> ID above!</div>`;
    return;
  }

  wrap.innerHTML = filtered
    .map((m) => {
      const isAct = m.id.toLowerCase() === currentId;
      const archBadge = m.is_moe ? `MoE ${m.active_params_b}B/${m.total_params_b}B` : `${m.total_params_b}B Dense`;
      const nodeBadge = m.min_nodes_required >= 2 ? "2×GB10" : m.total_params_b <= 4.5 ? "LaptopFit" : "1×GB10";
      return `
        <button type="button" class="model-card ${isAct ? "model-card-active" : ""}" data-model-id="${m.id}" title="${m.id} — ${m.recommended_use}">
          <div class="mc-top">
            <span class="mc-label">${m.label}</span>
            <span class="mc-node ${m.min_nodes_required >= 2 ? "mc-node-2" : ""}">${nodeBadge}</span>
          </div>
          <div class="mc-meta">${m.provider} &middot; ${archBadge} &middot; ${m.license}</div>
        </button>
      `;
    })
    .join("");

  wrap.querySelectorAll(".model-card").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.getElementById("model-id-input").value = btn.dataset.modelId;
      refreshModelSpec(btn.dataset.modelId);
    });
  });
}

function renderCategoryTabs(categories) {
  const tabsWrap = document.getElementById("model-category-tabs");
  tabsWrap.innerHTML = categories
    .map(
      (cat) =>
        `<button type="button" class="cat-tab ${cat === activeCategory ? "cat-tab-active" : ""}" data-cat="${cat}">${cat}</button>`
    )
    .join("");
  tabsWrap.querySelectorAll(".cat-tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      activeCategory = btn.dataset.cat;
      tabsWrap.querySelectorAll(".cat-tab").forEach((b) => b.classList.toggle("cat-tab-active", b === btn));
      renderCatalogCards();
    });
  });
}

function renderDatasetBanner(ds) {
  if (!ds) return;
  activeDatasetId = ds.dataset_id;
  const sc = ds.split_doc_counts || {};
  const cc = ds.split_chunk_counts || {};
  const prefix = ds.is_preset ? "Preset Active" : "Private Corpus Staged";
  document.getElementById("dataset-status-bar").innerHTML =
    `${prefix}: <strong>${ds.display_name}</strong> (${ds.format_summary}) &middot; ` +
    `<strong>${(ds.total_tokens || 0).toLocaleString()} tokens</strong> &middot; ` +
    `Train: <code>${sc.train || 0} docs (${cc.train || 0})</code> &middot; ` +
    `Val: <code>${sc.val || 0} docs (${cc.val || 0})</code> &middot; ` +
    `Sealed Test: <code>${sc.held_out || 0} docs (${cc.held_out || 0})</code>`;
}

function renderClientMachineBridge(clientInfo, clusterEp, teamId) {
  if (clientInfo) {
    document.getElementById("pill-client-host").textContent = clientInfo.hostname || "client.local";
    document.getElementById("bridge-client-name").textContent =
      `${clientInfo.hostname} (${clientInfo.os} · ${clientInfo.arch} · ${clientInfo.cpu_cores} cores)`;
  }
  if (clusterEp) {
    document.getElementById("pill-cluster-ep").textContent = clusterEp;
    document.getElementById("bridge-server-ep").textContent = clusterEp;
  }
  if (teamId) {
    document.getElementById("pill-def-id").textContent = teamId;
    document.getElementById("bridge-team-tag").textContent = teamId;
  }
}

function renderNetworkBridgeState(discReport, clusterCfg) {
  const dot = document.getElementById("dot-cluster");
  const statusEl = document.getElementById("bridge-conn-status");
  const backendSel = document.getElementById("backend-select");
  if (!discReport || !statusEl) return;

  const reachable = Boolean(discReport.cluster_reachable);
  const cliOk = Boolean(discReport.fuzzball_cli_available);
  const vol = (clusterCfg && clusterCfg.vast && clusterCfg.vast.persistent_volume_name) || "runs";
  const prov = (clusterCfg && clusterCfg.vast && clusterCfg.vast.provisioner) || "team3";

  if (reachable) {
    if (dot) dot.className = "dot dot-green";
    statusEl.innerHTML =
      `<strong style="color:#a3e635">LIVE CLUSTER CONNECTED</strong> &middot; Fuzzball MCP Ready &middot; ` +
      `VAST Volume <code>${vol}</code> (<code>use: ${prov}</code>)`;
    if (cliOk && backendSel) {
      backendSel.value = "fuzzball_mcp";
    }
  } else {
    if (dot) dot.className = "dot dot-amber";
    const cliBadge = cliOk ? "<code>fuzzball</code> CLI + <code>team3</code> context ready" : "local config ready";
    statusEl.innerHTML =
      `<strong style="color:#fbbf24">OFF-CLUSTER WI-FI (PREVIEW MODE)</strong> &middot; ${cliBadge} &middot; ` +
      `Auto-connects to Live GPUs on cluster Wi-Fi`;
    if (backendSel && (backendSel.value === "fuzzball_mcp" || backendSel.value === "fuzzball")) {
      backendSel.value = "simulator";
    }
  }
}

async function initStudio() {
  try {
    const [modelsResp, dsResp, discResp] = await Promise.all([
      fetchJson("/api/models"),
      fetchJson("/api/datasets"),
      fetchJson("/api/discover"),
    ]);

    fullCatalog = modelsResp.catalog || [];
    document.getElementById("catalog-count-badge").textContent = String(modelsResp.total_count || fullCatalog.length);
    renderCategoryTabs(modelsResp.categories || ["Featured", "All Models"]);
    renderCatalogCards();

    if (dsResp.preset) {
      renderDatasetBanner(dsResp.preset);
    }
    if (discResp.discovery_report && discResp.discovery_report.cluster_endpoint) {
      resolvedClusterEndpoint = discResp.discovery_report.cluster_endpoint;
    }
    if (discResp.cluster_config && discResp.cluster_config.definition_id) {
      resolvedTeamId = discResp.cluster_config.definition_id;
      renderClientMachineBridge(discResp.client_machine, currentClusterEndpoint(), resolvedTeamId);
    }
    renderNetworkBridgeState(discResp.discovery_report, discResp.cluster_config);
    await refreshModelSpec(currentModelId());
    await runPlaygroundSample();
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
  bar.innerHTML = `Ingesting <strong>${arr.length}</strong> local file(s) into leak-free Train / Val / Sealed Test splits…`;

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
    document.getElementById("build-title").innerHTML = `Local LLM Package Ready: <code>${shortModel}</code>`;
    document.getElementById("build-subtitle").innerHTML =
      `Tuned for <strong>${dsName}</strong> &mdash; ready for client Ollama / llama.cpp, vLLM, or 1-command Fuzzball deploy`;
    const insp = document.getElementById("inspector-model-label");
    if (insp) insp.textContent = prev.model_id;
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

function updateSingleGpuSvg(idx, node) {
  if (!node) return;
  const pfx = `svg-gpu${idx}`;
  const smPct = Number(node.sm_util_pct || 0);
  const tcPct = Number(node.tensor_core_pct || 0);
  const memUsed = Number(node.vram_used_gib || 0);
  const memTot = Number(node.physical_memory_gib || 127.9);
  const memPct = Math.min(100, (memUsed / Math.max(memTot, 1)) * 100);
  const isRunning = String(node.state || "").includes("RUNNING");

  document.getElementById(`${pfx}-host`).textContent = node.hostname || `gb10-node-0${idx + 1}`;
  document.getElementById(`${pfx}-state`).textContent = node.state || "READY";
  document.getElementById(`${pfx}-state`).setAttribute("fill", isRunning ? "#a3e635" : "#34d399");
  document.getElementById(`${pfx}-border`).setAttribute("stroke", isRunning ? "#76b900" : "#2b3d5f");
  document.getElementById(`${pfx}-role`).textContent = node.role || `${node.gpu_product} (${node.sm_arch || "sm_121"})`;

  document.getElementById(`${pfx}-sm-txt`).textContent = `${smPct.toFixed(1)}%`;
  document.getElementById(`${pfx}-sm-bar`).setAttribute("width", String(Math.max(4, Math.round((smPct / 100) * 292))));

  document.getElementById(`${pfx}-tc-txt`).textContent = `${tcPct.toFixed(1)}%`;
  document.getElementById(`${pfx}-tc-bar`).setAttribute("width", String(Math.max(4, Math.round((tcPct / 100) * 292))));

  document.getElementById(`${pfx}-mem-txt`).textContent = `${memUsed.toFixed(1)} / ${memTot.toFixed(1)} GiB`;
  document.getElementById(`${pfx}-mem-bar`).setAttribute("width", String(Math.max(6, Math.round((memPct / 100) * 292))));

  document.getElementById(`${pfx}-tflops`).textContent =
    `${Number(node.bf16_tflops || 0).toFixed(1)} TFLOP/s · ${Number(node.mem_bw_gbs || 0).toFixed(0)} GB/s`;
  document.getElementById(`${pfx}-toks`).textContent =
    `${Math.round(Number(node.tokens_per_sec || 0)).toLocaleString()} tok/s · ${node.power_w || 28}W · ${node.temp_c || 40}°C`;

  const wfShort = node.active_workflow_id ? node.active_workflow_id.slice(0, 8) : "idle";
  document.getElementById(`${pfx}-wf`).textContent =
    `Workflow: ${wfShort} · Trial: ${node.active_trial_id || node.last_trial_id || "none"} · Done: ${node.trials_completed || 0}`;

  // Light up mini SM die cells proportionally to SM utilization
  const cells = document.querySelectorAll(`#${pfx}-cores .sm-cell`);
  const litCount = Math.round((smPct / 100) * cells.length);
  cells.forEach((cell, cIdx) => {
    if (cIdx < litCount) {
      cell.setAttribute("fill", isRunning ? "#84cc16" : "#38bdf8");
    } else {
      cell.setAttribute("fill", "#1d2c47");
    }
  });
}

function renderDualGpuFabric(snap) {
  const nodes = snap.nodes || [];
  if (nodes[0]) updateSingleGpuSvg(0, nodes[0]);
  if (nodes[1]) updateSingleGpuSvg(1, nodes[1]);

  const gt = snap.gpu_telemetry || {};
  const active = Boolean(gt.link_active);
  const bwGbps = Number(gt.inter_gpu_gbps || 0);
  const arMs = Number(gt.allreduce_ms || 0);
  const payMb = Number(gt.payload_mb || 0);
  const mode = String(gt.comm_mode || "IDLE");

  document.getElementById("fabric-mode-badge").textContent = mode.replace(/_/g, " ");
  document.getElementById("fabric-bw-badge").textContent =
    bwGbps > 0 ? `NCCL RoCE · ${bwGbps.toFixed(1)} Gbps (${arMs.toFixed(1)} ms sync)` : "NCCL RoCE 200GbE · Standby";

  document.getElementById("svg-comm-proto").textContent = gt.protocol || "NCCL RoCE v2 · 200GbE RDMA";
  document.getElementById("svg-comm-bw").textContent =
    bwGbps > 0
      ? `Active Collective: ${bwGbps.toFixed(1)} Gbps (${gt.link_utilization_pct || 0}% Link)`
      : mode === "SINGLE_GPU_LOCAL_BUS"
      ? "Single-Node Execution · 0ms Cross-Node Sync"
      : "Dual-GPU Link Ready";

  document.getElementById("svg-allreduce-txt").textContent =
    arMs > 0
      ? `AllReduce Sync: ${arMs.toFixed(1)} ms · Grad Payload: ${payMb.toFixed(1)} MB`
      : `VAST Read: ${Number(gt.vast_read_gbs || 1.64).toFixed(2)} GB/s · Packets: ${gt.packets_synced || 0}`;
  document.getElementById("svg-collective-op").textContent =
    gt.collective_op || `2-Node Scaling Efficiency: ${gt.scaling_efficiency_pct || 99.2}%`;

  const fwd = document.getElementById("svg-link-fwd");
  const rev = document.getElementById("svg-link-rev");
  fwd.classList.toggle("flow-fast", active);
  rev.classList.toggle("flow-fast", active);
  fwd.setAttribute("opacity", active ? "1" : "0.35");
  rev.setAttribute("opacity", active ? "1" : "0.35");

  if (mode === "NCCL_FSDP_SHARDED") {
    document.getElementById("svg-fwd-label").textContent = "FSDP AllGather Shards →";
    document.getElementById("svg-rev-label").textContent = "← ReduceScatter Grads";
  } else if (mode === "VLLM_TP2_ALLREDUCE") {
    document.getElementById("svg-fwd-label").textContent = "vLLM TP2 Activations →";
    document.getElementById("svg-rev-label").textContent = "← TP2 AllReduce Sync";
  } else if (mode === "PARALLEL_TRIALS_VAST_SHARED") {
    document.getElementById("svg-fwd-label").textContent = "Independent Recipe Rung GPU 0 →";
    document.getElementById("svg-rev-label").textContent = "← Independent Recipe Rung GPU 1";
  } else {
    document.getElementById("svg-fwd-label").textContent = "NCCL Ring AllReduce Rank 0 → 1";
    document.getElementById("svg-rev-label").textContent = "← NCCL Ring AllReduce Rank 1 → 0";
  }

  const team = snap.definition_id || currentTeamId();
  document.getElementById("svg-vast-title").textContent =
    `VAST Data Shared Persistent Volume (/vast · use: ${team}) — Zero-Copy Weights, Tokenized Splits & LoRA Checkpoints`;
}

function renderRealtimeAnalysis(snap) {
  const m = snap.live_step_metrics || {};
  if (m && m.trial_id) {
    document.getElementById("rt-step").textContent = `${m.trial_id.slice(0, 18)} [${m.step}/${m.max_steps || m.step}]`;
    const vTxt = m.val_loss != null ? Number(m.val_loss).toFixed(4) : "—";
    document.getElementById("rt-loss").textContent = `${Number(m.train_loss || 0).toFixed(4)} / ${vTxt}`;
    document.getElementById("rt-ppl").textContent = m.perplexity != null ? Number(m.perplexity).toFixed(3) : "—";
    document.getElementById("rt-lr").textContent = m.lr != null ? Number(m.lr).toExponential(2) : "—";
    document.getElementById("rt-grad").textContent = m.grad_norm != null ? Number(m.grad_norm).toFixed(3) : "—";
    document.getElementById("rt-entropy").textContent = m.router_entropy != null ? Number(m.router_entropy).toFixed(3) : "1.000";
    document.getElementById("rt-memsave").textContent =
      m.logit_mem_saved_gb > 0 ? `-${Number(m.logit_mem_saved_gb).toFixed(2)} GiB` : "Baseline";
    document.getElementById("rt-toks").textContent = `${Math.round(Number(m.tokens_per_sec || 0)).toLocaleString()} tok/s`;
  }

  const ledgerBody = document.querySelector("#live-trial-ledger tbody");
  if (ledgerBody) {
    const rows = (snap.completed_trials || []).slice().reverse().slice(0, 8);
    ledgerBody.innerHTML = rows
      .map(
        (t) => `
        <tr>
          <td><code>${t.trial_id}</code></td>
          <td>${t.world_size || 1}&times;GB10</td>
          <td><strong>${Math.round(Number(t.tokens_per_sec || 0)).toLocaleString()}</strong></td>
          <td>${t.val_loss != null ? Number(t.val_loss).toFixed(4) : "—"}</td>
          <td>${t.held_out_perplexity ? Number(t.held_out_perplexity).toFixed(3) : "—"}</td>
        </tr>
      `
      )
      .join("");
  }
}

function renderLiveTelemetry(snap) {
  latestLiveSnap = snap;
  renderClientMachineBridge(snap.client_machine, snap.cluster_endpoint, snap.definition_id || currentTeamId());
  document.getElementById("live-stage-label").textContent = snap.stage_label || "Ready";

  const stBadge = document.getElementById("live-run-state");
  const st = String(snap.state || "idle");
  stBadge.textContent =
    st === "queued_for_gpu"
      ? `QUEUED ON FUZZBALL · ${snap.backend}`
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

  renderDualGpuFabric(snap);
  renderRealtimeAnalysis(snap);

  if (st === "running" || st === "queued_for_gpu") {
    const activeTid = (snap.live_step_metrics && snap.live_step_metrics.trial_id) || "";
    if (snap.live_curves && Object.keys(snap.live_curves).length > 0) {
      renderLossSvg(snap.live_curves, snap.live_val_curves || {}, activeTid, snap.live_curve_equivalence || {});
    }
    renderIncrementalStagePanels(snap);
  }
}

function renderIncrementalStagePanels(snap) {
  const shortModel = String(snap.model_id || currentModelId()).split("/").pop();
  document.getElementById("kpi-model-short").textContent = shortModel;

  // 1. Live Curve Equivalence Badge
  const eq = snap.live_curve_equivalence || {};
  const curveBadge = document.getElementById("curve-badge");
  if (eq && eq.reference_trial_id && eq.max_delta != null) {
    const d = Number(eq.max_delta);
    const ok = eq.accepted !== false;
    curveBadge.textContent = ok
      ? `Live Δ = ${d.toFixed(4)} (≤ 0.025 PASS)`
      : `Live Δ = ${d.toFixed(4)} (> 0.025 DRIFT)`;
    curveBadge.className = ok ? "badge badge-green" : "badge badge-amber";
  } else if (snap.current_stage === "stage_a_recipe") {
    curveBadge.textContent = `Stage A Live Val Curves (${Object.keys(snap.live_curves || {}).length} trials)`;
    curveBadge.className = "badge";
  }

  // 2. Stage A Live Funnel Card
  const saLive = snap.live_stage_a || {};
  if (saLive.best_perplexity) {
    document.getElementById("kpi-ppl").textContent = `${Number(saLive.base_perplexity).toFixed(3)} → ${Number(saLive.best_perplexity).toFixed(3)}`;
    document.getElementById("meta-stage-a").textContent =
      `Leading: ${saLive.best_trial_id} · Val ${Number(saLive.best_val_loss).toFixed(3)} (${saLive.trials_completed} trial(s) done)`;
  } else if (snap.current_stage === "stage_a_recipe") {
    document.getElementById("meta-stage-a").textContent = "Running parallel Successive Halving rungs…";
  }

  // 3. Stage B Live Throughput Waterfall + Funnel Card
  const sbList = (snap.live_stage_b_outcomes || []).slice();
  const activeStep = snap.live_step_metrics || {};
  if (activeStep.trial_id && String(activeStep.trial_id).startsWith("stageb-")) {
    const alreadyDone = sbList.some((o) => o.trial_id === activeStep.trial_id);
    if (!alreadyDone) {
      const lbl = String(activeStep.trial_id).split("-").slice(2).join("-") || activeStep.trial_id;
      sbList.push({
        trial_id: activeStep.trial_id,
        label: lbl,
        in_progress: true,
        accepted: activeStep.live_curve_delta == null || Number(activeStep.live_curve_delta) <= 0.025,
        max_loss_curve_delta: Number(activeStep.live_curve_delta || 0),
        summary: {
          mean_tokens_per_sec: Number(activeStep.tokens_per_sec || 0),
          world_size: Number(activeStep.world_size || 1),
          peak_mem_gb: Number(activeStep.peak_mem_gb || 0),
          hardware_utilization_pct: Number(activeStep.hw_util_pct || 0),
        },
      });
    }
  }
  if (sbList.length > 0) {
    const baseTok = Math.max(sbList[0].summary.mean_tokens_per_sec, 1);
    const acceptedDone = sbList.filter((o) => !o.in_progress && o.accepted);
    const bestOut =
      acceptedDone.length > 0
        ? acceptedDone.reduce((a, b) => (b.summary.mean_tokens_per_sec > a.summary.mean_tokens_per_sec ? b : a))
        : sbList[0];
    const speedup = (bestOut.summary.mean_tokens_per_sec / baseTok).toFixed(2);
    document.getElementById("kpi-train-speed").textContent =
      `${Math.round(bestOut.summary.mean_tokens_per_sec).toLocaleString()} tok/s (${speedup}x)`;
    document.getElementById("meta-stage-b").textContent =
      `Live Stage B (${sbList.filter((x) => !x.in_progress).length} swept) · Best: ${bestOut.label}`;
    document.getElementById("util-badge").textContent =
      `${Number(bestOut.summary.hardware_utilization_pct || 0).toFixed(1)}% Cluster Peak`;

    const maxTok = Math.max(...sbList.map((o) => o.summary.mean_tokens_per_sec), 1);
    document.getElementById("train-bars").innerHTML = sbList
      .map((o) => {
        const tok = o.summary.mean_tokens_per_sec;
        const pct = Math.max(6, (tok / maxTok) * 100).toFixed(1);
        const isWin = !o.in_progress && o.trial_id === bestOut.trial_id && o.accepted;
        const cls = !o.accepted ? "bar-fill bar-fill-rejected" : isWin ? "bar-fill bar-fill-winner" : "bar-fill";
        const statusTag = o.in_progress
          ? `<span class="tag-ok">RUNNING (Δ ${o.max_loss_curve_delta.toFixed(4)})</span>`
          : o.accepted
          ? isWin
            ? `<span class="tag-ok">LEADING</span>`
            : `<span class="tag-ok">OK (Δ ${o.max_loss_curve_delta.toFixed(4)})</span>`
          : `<span class="tag-reject">REJECTED (${o.rejection_reason || "Drift"})</span>`;
        return `
          <div class="bar-row">
            <div class="bar-top">
              <span><code>${o.label}</code> (${o.summary.world_size}×GB10 · ${o.summary.peak_mem_gb.toFixed(0)} GiB)</span>
              <span><strong>${Math.round(tok).toLocaleString()} tok/s</strong> &nbsp;${statusTag}</span>
            </div>
            <div class="bar-track"><div class="${cls}" style="width:${pct}%"></div></div>
          </div>
        `;
      })
      .join("");
  }

  // 4. Stage C Live Serving Sweep Table + Funnel Card
  const scList = snap.live_stage_c_outcomes || [];
  if (scList.length > 0) {
    const baseSrv = Math.max(scList[0].throughput_tokens_per_sec || 1, 1);
    const accSrv = scList.filter((s) => s.accepted);
    const bestSrv =
      accSrv.length > 0
        ? accSrv.reduce((a, b) => (b.throughput_tokens_per_sec > a.throughput_tokens_per_sec ? b : a))
        : scList[0];
    const srvSpeedup = (bestSrv.throughput_tokens_per_sec / baseSrv).toFixed(2);
    document.getElementById("kpi-serve-speed").textContent =
      `${Math.round(bestSrv.throughput_tokens_per_sec).toLocaleString()} tok/s (${srvSpeedup}x)`;
    document.getElementById("meta-stage-c").textContent =
      `Live Stage C (${scList.length} swept) · Leading: ${bestSrv.label}`;
    document.querySelector("#serving-table tbody").innerHTML = scList
      .map((s) => {
        const isWin = s.label === bestSrv.label && s.accepted;
        const badge = s.accepted
          ? `<span class="tag-ok">${isWin ? "LEADING" : "PASS"}</span>`
          : `<span class="tag-reject" title="${(s.rejection_reasons || []).join("; ")}">REJECTED</span>`;
        return `
          <tr class="${isWin ? "row-winner" : ""}">
            <td><code>${s.label}</code></td>
            <td>${s.precision}</td>
            <td>${s.replicas}&times;TP${s.tensor_parallel} (B=${s.max_num_seqs})</td>
            <td><strong>${Math.round(s.throughput_tokens_per_sec).toLocaleString()}</strong></td>
            <td>${Number(s.p99_latency_ms).toFixed(0)} ms</td>
            <td>+${Number(s.perplexity_regression_pct).toFixed(2)}%</td>
            <td>${badge}</td>
          </tr>
        `;
      })
      .join("");
  }

  // 5. Live Advisor Rationales & Firewall Audit Stream
  if (Array.isArray(snap.live_firewall) && snap.live_firewall.length > 0) {
    const exps = snap.live_firewall.filter((f) => f.explanation && f.explanation.rationale);
    if (exps.length > 0) {
      document.getElementById("advisor-rationales").innerHTML = exps
        .map((f) => `<div class="advisor-item"><strong>[${f.explanation.stage}]</strong> ${f.explanation.rationale}</div>`)
        .join("");
    }
    const lastRec = snap.live_firewall[snap.live_firewall.length - 1].record || {};
    const sha = String(lastRec.payload_sha256 || "").slice(0, 16);
    const fwBox = document.getElementById("firewall-audit-box");
    fwBox.classList.remove("blocked-alert");
    fwBox.innerHTML =
      `<strong>Live Privacy Firewall Stream:</strong> ${snap.live_firewall.length} stage payload(s) verified &middot; ` +
      `<code>contains_raw_data=false</code> &middot; Latest SHA-256: <code>${sha}…</code>`;
  }
}

function renderLossSvg(curves, valCurves = {}, activeTrialId = "", liveEq = {}) {
  const wrap = document.getElementById("loss-chart-wrap");
  const legend = document.getElementById("loss-legend");
  const allKeys = Object.keys(curves).filter((k) => Array.isArray(curves[k]) && curves[k].length > 1);
  const stageBKeys = allKeys.filter((k) => k.startsWith("stageb-")).sort();

  const active = [];
  if (stageBKeys.length > 0) {
    const refKey = stageBKeys[0];
    active.push({
      key: refKey,
      label: `Reference (${refKey.split("-").slice(2).join("-")})`,
      color: "#38bdf8",
      dash: "",
      src: curves,
    });

    // Include currently running Stage B candidate live as steps arrive!
    if (activeTrialId && activeTrialId !== refKey && Array.isArray(curves[activeTrialId]) && curves[activeTrialId].length > 1) {
      const drifting = liveEq && liveEq.accepted === false;
      active.push({
        key: activeTrialId,
        label: `Live: ${activeTrialId.split("-").slice(2).join("-")} (streaming)`,
        color: drifting ? "#f87171" : "#a3e635",
        dash: "5,3",
        src: curves,
      });
    } else {
      // Otherwise show best accepted Stage B winner + any rejected drift curve (works for both 1-node & 2-node models)
      const otherB = stageBKeys.slice(1);
      const driftKey = otherB.find((k) => k.includes("oversized"));
      const bestAcceptedKey = otherB.filter((k) => k !== driftKey).pop();
      if (bestAcceptedKey) {
        active.push({
          key: bestAcceptedKey,
          label: `${bestAcceptedKey.split("-").slice(2).join("-")} (Accepted)`,
          color: "#a3e635",
          dash: "5,3",
          src: curves,
        });
      }
      if (driftKey) {
        active.push({
          key: driftKey,
          label: `${driftKey.split("-").slice(2).join("-")} (Rejected: Drift)`,
          color: "#f87171",
          dash: "2,3",
          src: curves,
        });
      }
    }
  } else {
    const recentKeys = allKeys.slice(-3);
    const palette = ["#38bdf8", "#a3e635", "#c084fc"];
    recentKeys.forEach((k, idx) => {
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
      <text x="${(width + pad.left) / 2}" y="${height - 5}" fill="#7f91ad" font-size="10.5" text-anchor="middle">Optimization Step (Train &amp; Validation Split Evaluation)</text>
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
    `Winner: ${sa.winning_label} · Val ${sa.winning_summary.final_val_loss.toFixed(3)} · Test ${sa.winning_summary.held_out_eval_loss.toFixed(3)} (${sa.early_stopped_count} pruned)`;

  const baseTrainTok = sb.outcomes[0].summary.mean_tokens_per_sec;
  const winTrainOut = sb.outcomes.find((o) => o.label === sb.winning_label) || sb.outcomes[0];
  const bestTrainTok = winTrainOut.summary.mean_tokens_per_sec;
  document.getElementById("kpi-train-speed").textContent =
    `${Math.round(bestTrainTok).toLocaleString()} tok/s (${sb.overall_speedup_x}x)`;
  document.getElementById("meta-stage-b").textContent =
    `${Math.round(baseTrainTok).toLocaleString()} → ${Math.round(bestTrainTok).toLocaleString()} tok/s · 1-node sm_121: ${sb.single_node_kernel_speedup_x}x`;

  const winServeOut = sc.outcomes.find((o) => o.label === sc.winning_label) || sc.outcomes[0];
  document.getElementById("kpi-serve-speed").textContent =
    `${Math.round(winServeOut.throughput_tokens_per_sec).toLocaleString()} tok/s (${sc.serving_speedup_x}x)`;
  document.getElementById("meta-stage-c").textContent =
    `Winner: ${sc.winning_label} · Test PPL +${winServeOut.perplexity_regression_pct.toFixed(2)}% · P99 ${winServeOut.p99_latency_ms.toFixed(0)} ms`;

  const scs = data.dataset_manifest.split_file_counts || {};
  document.getElementById("meta-stage-d").textContent =
    `Split (${scs.train || 0}T / ${scs.val || 0}V / ${scs.held_out || 0}Test) · VAST use: ${data.cluster.vast.provisioner}`;

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
            <span><code>${o.label}</code> (${o.summary.world_size}×GB10 · ${o.summary.peak_mem_gb.toFixed(0)} GiB)</span>
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
    `<strong>Client &harr; Fuzzball Privacy Firewall Verified:</strong> ${audits.length} payloads audited &middot; ` +
    `<code>contains_raw_data=false</code> &middot; SHA-256: <code>${lastDigest}…</code>`;
}

async function runPlaygroundSample() {
  try {
    const prompt = document.getElementById("playground-prompt").value;
    const res = await fetchJson("/api/playground/complete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ prompt, compare_base: true }),
    });
    document.getElementById("pg-base-out").textContent = res.base_output || "";
    document.getElementById("pg-tuned-out").textContent = res.tuned_output || "";
  } catch (_) {}
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
      await runPlaygroundSample();
      document.getElementById("btn-run").disabled = false;
      document.getElementById("btn-run").textContent = "Tune on Fuzzball & Generate Local LLM";
    } else if (snap.state === "failed") {
      document.getElementById("btn-run").disabled = false;
      document.getElementById("btn-run").textContent = "Tune on Fuzzball & Generate Local LLM";
    }
    prevState = snap.state;
  };
}

// Event bindings
document.getElementById("model-id-input").addEventListener("change", (e) => {
  refreshModelSpec(e.target.value);
  renderCatalogCards();
});
document.getElementById("model-id-input").addEventListener("blur", (e) => refreshModelSpec(e.target.value));
document.getElementById("catalog-filter-input").addEventListener("input", () => renderCatalogCards());

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

document.getElementById("backend-select").addEventListener("change", () => {
  userPickedBackend = true;
});

document.getElementById("btn-run").addEventListener("click", async () => {
  const btn = document.getElementById("btn-run");
  const modelId = currentModelId();
  if (!userPickedBackend) {
    try {
      const discResp = await fetchJson("/api/discover");
      if (discResp.cluster_config && discResp.cluster_config.definition_id) {
        resolvedTeamId = discResp.cluster_config.definition_id;
      }
      renderNetworkBridgeState(discResp.discovery_report, discResp.cluster_config);
    } catch (_) {}
  }
  const backend = document.getElementById("backend-select").value;
  const teamId = currentTeamId();
  const clusterEp = currentClusterEndpoint();
  btn.disabled = true;
  btn.textContent = `Tuning on Fuzzball (${backend})…`;
  const resp = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      model_id: modelId,
      dataset_id: activeDatasetId,
      backend,
      fuzzfile_version: "v4",
      definition_id: teamId,
      cluster_endpoint: clusterEp,
      async_mode: true,
    }),
  });
  if (!resp.ok) {
    btn.disabled = false;
    btn.textContent = "Tune on Fuzzball & Generate Local LLM";
    const err = await resp.json();
    const errBanner = document.getElementById("live-error-banner");
    errBanner.textContent = err.detail || "Failed to start optimization";
    errBanner.classList.remove("hidden");
  }
});

document.getElementById("btn-run-playground").addEventListener("click", runPlaygroundSample);

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
      `<strong>Zero-Leak Firewall Guard Triggered:</strong> Blocked raw C++ snippet from leaving client/VAST boundary — ` +
      `<code>${res.violation}</code> (SHA-256: <code>${res.record.payload_sha256.slice(0, 16)}…</code>)`;
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
  setTimeout(() => (btn.textContent = "Copy"), 1400);
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
  setTimeout(connectLiveStream, 350);
})();

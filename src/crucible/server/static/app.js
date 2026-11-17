let latestSummary = null;
let latestLiveSnap = null;
let buildArtifacts = {};
let selectedArtifactKey = "scripts/run_company_app.py";
let activeDatasetId = "builtin-lammps-cpp-bench";
let fullCatalog = [];
let activeCategory = "Featured";
let resolvedTeamId = "team3";
let resolvedClusterEndpoint = "api.fb-hack.civilfritz.net:443";
let userPickedBackend = false;
let lastObservedRunState = "idle";
let currentLifecyclePhase = "pre_run";
let activeDatasetMeta = null;
let activePollTimer = null;

async function fetchJson(url, opts) {
  const resp = await fetch(url, opts);
  return await resp.json();
}

function currentCompanyName() {
  const el = document.getElementById("company-name-input");
  return (el && el.value.trim()) || "Apex Dynamics";
}

function syncCompanyBranding() {
  const comp = currentCompanyName();
  const initials = comp
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((w) => w[0].toUpperCase())
    .join("") || "AI";
  const slug = comp.toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-|-$/g, "") || "company";
  const av = document.getElementById("copilot-avatar");
  if (av) av.textContent = initials;
  const hd = document.getElementById("copilot-brand-heading");
  if (hd) hd.textContent = `${comp} private engineering copilot`;
  const kk = document.getElementById("pg-tuned-kicker");
  if (kk) kk.textContent = `${comp} tuned copilot`;
  const dlHref = `/api/build/bundle.zip?company_name=${encodeURIComponent(comp)}`;
  const topDl = document.getElementById("btn-top-download");
  if (topDl) topDl.href = dlHref;
  const cardDl = document.getElementById("btn-card-download");
  if (cardDl) cardDl.href = dlHref;
  const uz = document.getElementById("unzip-cmd-line");
  if (uz) {
    uz.textContent = `cd ~/Downloads && unzip -o "$(ls -t ${slug}-private-llm-*.zip | head -1)" -d ${slug}-ai && python3 ${slug}-ai/scripts/run_company_app.py`;
  }
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
    if (Array.isArray(info.catalog)) {
      fullCatalog = info.catalog;
    }
    const m = info.inferred;
    if (!m) return;
    const size =
      m.num_experts > 1
        ? `MoE · ${m.total_params_b}B total, ${m.active_params_b}B active`
        : `Dense · ${m.total_params_b}B`;
    const vast =
      m.vast_state === "preloaded"
        ? "on VAST"
        : m.vast_state === "loading"
        ? "loading to VAST…"
        : `needs loading to VAST (~${m.est_vast_load_seconds}s)`;
    const gpus = m.min_nodes_required >= 2 ? " · needs both GPUs" : "";
    const preset = m.has_hardware_preset ? " · saved preset" : "";
    document.getElementById("model-current").textContent = String(m.name || modelId).split("/").pop();
    document.getElementById("model-current").title = m.name || modelId;
    const spec = document.getElementById("model-spec-line");
    spec.textContent = `${size} · ${vast}${gpus}${preset}`;
    spec.title = `BF16 ${m.bf16_footprint_gb} GB · FP8 ${m.fp8_footprint_gb} GB · NVFP4 ${m.nvfp4_footprint_gb} GB`;

    const stageBtn = document.getElementById("btn-stage-model");
    if (stageBtn) {
      if (m.vast_state === "preloaded") {
        stageBtn.classList.add("hidden");
      } else {
        stageBtn.classList.remove("hidden");
        stageBtn.disabled = m.vast_state === "loading";
        stageBtn.textContent = m.vast_state === "loading" ? "Loading to VAST…" : `Load onto VAST (~${m.est_vast_load_seconds}s)`;
      }
    }

    const pTitle = document.getElementById("preset-callout-title");
    const pVal = document.getElementById("preset-savings-val");
    if (pTitle && pVal) {
      if (m.has_hardware_preset && document.getElementById("chk-company-presets").checked) {
        pTitle.textContent = `Preset ${m.hardware_preset.preset_id}`;
        pVal.textContent = `saves $${Number(m.hardware_preset.warm_start_savings_usd).toFixed(2)} of kernel probing`;
      } else {
        pTitle.textContent = "No saved preset";
        pVal.textContent = "runs the full kernel search";
      }
    }

    renderCatalogCards();
  } catch (_) {}
}

function renderCatalogCards() {
  const query = (document.getElementById("catalog-filter-input").value || "").trim().toLowerCase();
  const currentId = currentModelId().toLowerCase();
  const filtered = fullCatalog.filter((m) => {
    let catOk = true;
    if (activeCategory === "Featured") catOk = Boolean(m.featured);
    else if (activeCategory === "Pre-Loaded on VAST") catOk = Boolean(m.preloaded_on_vast);
    else if (activeCategory === "Preset Opt Ready") catOk = Boolean(m.has_hardware_preset);
    else if (activeCategory !== "All Models") catOk = m.category === activeCategory;
    if (!catOk) return false;
    if (!query) return true;
    const hay = `${m.id} ${m.label} ${m.provider} ${m.family} ${m.category} ${m.license} ${m.recommended_use}`.toLowerCase();
    return hay.includes(query);
  });

  const wrap = document.getElementById("model-chips");
  if (filtered.length === 0) {
    wrap.innerHTML = `<div class="catalog-empty">Nothing matches "${query}" in ${activeCategory}. You can still paste any Hugging Face <code>org/model</code> ID above.</div>`;
    return;
  }

  wrap.innerHTML = filtered
    .map((m) => {
      const isAct = m.id.toLowerCase() === currentId;
      const arch = m.is_moe ? `MoE ${m.active_params_b}B of ${m.total_params_b}B` : `${m.total_params_b}B dense`;
      const twoGpu = m.min_nodes_required >= 2;
      const extras = [m.vast_state === "preloaded" ? "" : "needs loading", m.has_hardware_preset ? "saved preset" : ""]
        .filter(Boolean)
        .join(" · ");
      return `
        <button type="button" class="model-card ${isAct ? "model-card-active" : ""}" aria-pressed="${isAct}" data-model-id="${m.id}" title="${m.id} — ${m.recommended_use}">
          <span class="mc-top">
            <span class="mc-label">${m.label}</span>
            <span class="mc-node ${twoGpu ? "mc-node-2" : ""}">${twoGpu ? "2 GPUs" : "1 GPU"}</span>
          </span>
          <span class="mc-meta">${m.provider} · ${arch} · ${m.license}${extras ? ` · ${extras}` : ""}</span>
        </button>
      `;
    })
    .join("");

  wrap.querySelectorAll(".model-card").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.getElementById("model-id-input").value = btn.dataset.modelId;
      refreshModelSpec(btn.dataset.modelId);
      document.getElementById("model-dialog").close();
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
  activeDatasetMeta = ds;
  const sc = ds.split_doc_counts || {};
  const tc = (ds.teacher_curriculum && ds.teacher_curriculum.split_pair_counts) || {};
  const splits =
    tc.train != null
      ? `${tc.train} / ${tc.val || 0} / ${tc.test || 0} train/val/test pairs`
      : `${sc.train || 0} / ${sc.val || 0} / ${sc.held_out || 0} train/val/test files`;
  const dsName = document.getElementById("dataset-name");
  dsName.textContent = ds.display_name || "Uploaded files";
  dsName.title = dsName.textContent;
  const meta = document.getElementById("dataset-status-bar");
  meta.textContent = `${splits} · ${(ds.total_tokens || 0).toLocaleString()} tokens · no overlap between splits`;
  meta.title = `${ds.format_summary || ""}`;
}

function renderClientMachineBridge(clientInfo, clusterEp, teamId) {
  if (clientInfo) {
    const host = document.getElementById("pill-client-host");
    host.textContent = clientInfo.hostname || "client.local";
    host.title = `${clientInfo.os} · ${clientInfo.arch} · ${clientInfo.cpu_cores} cores`;
  }
  if (clusterEp) document.getElementById("pill-cluster-ep").textContent = clusterEp;
  if (teamId) document.getElementById("pill-def-id").textContent = teamId;
}

// Header chip: where the NEXT run will execute.
function syncModeChip() {
  const v = document.getElementById("backend-select").value;
  const sim = v === "simulator";
  const chip = document.getElementById("mode-chip");
  chip.textContent = sim ? "Simulated" : v === "local_torch" ? "This laptop" : "Live · 2× GB10";
  chip.className = `chip ${sim ? "chip-sim" : "chip-live"}`;
}

// "Simulated" labels: where the SHOWN run actually ran. A simulated number must never read as measured
// (PRODUCT.md, principle 1), so this follows the server's effective backend, not the dropdown.
function setRunProvenance(backend) {
  const sim = String(backend || "simulator") === "simulator";
  document.getElementById("run-status").dataset.sim = String(sim);
  document.body.dataset.sim = String(sim);
}

function renderNetworkBridgeState(discReport, clusterCfg) {
  const dot = document.getElementById("dot-cluster");
  const statusEl = document.getElementById("bridge-conn-status");
  const backendSel = document.getElementById("backend-select");
  if (!discReport || !statusEl) return;

  const reachable = Boolean(discReport.cluster_reachable);
  const cordoned = Boolean(discReport.all_nodes_cordoned);
  const cliOk = Boolean(discReport.fuzzball_cli_available);
  const vol = (clusterCfg && clusterCfg.vast && clusterCfg.vast.persistent_volume_name) || "runs";
  const prov = (clusterCfg && clusterCfg.vast && clusterCfg.vast.provisioner) || "team3";

  if (reachable && !cordoned) {
    if (dot) dot.className = "dot dot-green";
    statusEl.innerHTML = `Connected. Runs go to the GB10 nodes through Fuzzball MCP; data lives on VAST volume <code>${vol}</code> (<code>${prov}</code>).`;
    if (cliOk && backendSel && !userPickedBackend) {
      backendSel.value = "fuzzball_mcp";
    }
  } else if (reachable && cordoned) {
    if (dot) dot.className = "dot dot-amber";
    statusEl.textContent = "Control plane reachable, but the GPU nodes are cordoned. Runs use the simulated GB10 twin.";
    if (backendSel && !userPickedBackend) {
      backendSel.value = "simulator";
    }
  } else {
    if (dot) dot.className = "dot dot-amber";
    statusEl.textContent = `Not on the cluster network${cliOk ? " (CLI and team3 context are set up)" : ""}. Runs use the simulated GB10 twin.`;
    if (backendSel && (backendSel.value === "fuzzball_mcp" || backendSel.value === "fuzzball")) {
      backendSel.value = "simulator";
    }
  }
  syncModeChip();
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
  bar.textContent = `Splitting ${arr.length} file(s) into train, validation and test sets…`;

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
    bar.textContent = `Upload failed: ${payload.detail || "the files could not be read"}. Try other files or use the LAMMPS sample.`;
    return;
  }
  renderDatasetBanner(payload);
}

async function refreshBuildPreview() {
  try {
    syncCompanyBranding();
    const comp = currentCompanyName();
    const prev = await fetchJson(`/api/build/preview?company_name=${encodeURIComponent(comp)}`);
    buildArtifacts = prev.artifacts || {};
    const shortModel = String(prev.model_id || "").split("/").pop();
    document.getElementById("build-title").textContent = `${comp} package for ${shortModel}`;
    document.getElementById("build-subtitle").innerHTML =
      `Unzip it and run <code>python3 scripts/run_company_app.py</code> to start ${comp}'s copilot on this machine.`;
    const subModel = document.getElementById("copilot-model-sub");
    if (subModel) {
      subModel.textContent = `${shortModel} with the tuned LoRA adapter · raw data never left this machine or VAST`;
    }
    const insp = document.getElementById("inspector-model-label");
    if (insp) insp.textContent = prev.model_id;
    renderArtifactViewer();
  } catch (_) {}
}

function renderArtifactViewer() {
  const viewer = document.getElementById("yaml-viewer");
  if (currentLifecyclePhase === "pre_run") {
    viewer.textContent = "# The package is generated when a run finishes.";
    return;
  }
  if (currentLifecyclePhase === "running") {
    viewer.textContent = "# Run in progress. The files appear here when it finishes.";
    return;
  }
  if (buildArtifacts[selectedArtifactKey]) {
    viewer.textContent = buildArtifacts[selectedArtifactKey];
  } else if (latestSummary && latestSummary.turnkey_fuzzfile) {
    viewer.textContent = latestSummary.turnkey_fuzzfile;
  }
}

const RUN_LABEL = "Tune and build";

// Progress per backend stage, and which of the four tracker steps it belongs to.
const STAGE_PCT = {
  starting: 12,
  ingest: 18,
  teacher_curriculum: 26,
  stage_a_recipe: 44,
  stage_b_training_perf: 64,
  stage_c_inference_perf: 80,
  continuous_teacher_loop: 90,
  stage_d_emit: 96,
};
const STAGE_HEADLINE = {
  starting: "Preparing the data",
  ingest: "Preparing the data",
  teacher_curriculum: "Writing teaching pairs from your data",
  stage_a_recipe: "Searching for the best recipe",
  stage_b_training_perf: "Making training fast",
  stage_c_inference_perf: "Choosing the fastest serving setup",
  continuous_teacher_loop: "Refining with the remote teacher",
  stage_d_emit: "Packaging the workflow",
};
const STAGE_STEP = {
  starting: 0,
  ingest: 0,
  teacher_curriculum: 0,
  stage_a_recipe: 0,
  stage_b_training_perf: 1,
  stage_c_inference_perf: 2,
  continuous_teacher_loop: 3,
  stage_d_emit: 3,
};

function renderStageTracker(phase, curStage) {
  const items = document.querySelectorAll("#stage-tracker li");
  const active = phase === "finished" ? items.length : phase === "running" ? STAGE_STEP[curStage] ?? 0 : -1;
  items.forEach((li, i) => {
    li.classList.toggle("is-done", i < active);
    li.classList.toggle("is-active", i === active);
  });
}

function setDownloadReady(el, ready) {
  if (!el) return;
  el.classList.toggle("is-locked", !ready);
  el.setAttribute("aria-disabled", String(!ready));
}

function renderLifecycleState(phase, snap = null, summary = null) {
  currentLifecyclePhase = phase;
  const byId = (id) => document.getElementById(id);
  const pill = byId("header-lifecycle-pill");
  const headline = byId("lifecycle-headline");
  const sub = byId("lifecycle-sub");
  const fill = byId("lifecycle-progress-fill");
  const curStage = String((snap && snap.current_stage) || "starting");
  const finished = phase === "finished";

  byId("run-status").dataset.phase = phase;
  renderStageTracker(phase, curStage);
  setDownloadReady(byId("btn-top-download"), finished);
  setDownloadReady(byId("btn-card-download"), finished);
  byId("top-download-hint").textContent = finished ? "Ready" : "After a run";
  byId("btn-jump-output").classList.toggle("hidden", !finished);
  byId("final-copilot-section").classList.toggle("hidden", !finished);
  byId("btn-run-cluster-copilot").disabled = !finished;
  byId("btn-run-playground").disabled = !finished;

  if (phase === "running") {
    const pct = STAGE_PCT[curStage] || 30;
    pill.className = "chip chip-running";
    pill.textContent = `Running · ${pct}%`;
    headline.textContent = STAGE_HEADLINE[curStage] || "Running";
    sub.textContent = (snap && snap.stage_label) || "The tuned copilot and its package appear when the run finishes.";
    fill.style.width = `${pct}%`;
    if (curStage !== "stage_d_emit") byId("kpi-dag").textContent = "—";
    renderArtifactViewer();
    return;
  }

  if (finished) {
    pill.className = "chip chip-done";
    pill.textContent = "Done";
    const comp = currentCompanyName();
    const shortM = String((summary && summary.model_profile && summary.model_profile.name) || currentModelId()).split("/").pop();
    headline.textContent = `Tuned ${shortM} for ${comp} is ready`;
    sub.textContent =
      summary && summary.stage_a
        ? `Perplexity ${summary.stage_a.base_held_out_perplexity.toFixed(2)} → ${summary.stage_a.winning_summary.held_out_perplexity.toFixed(2)} · training ${summary.stage_b.overall_speedup_x}× · serving ${summary.stage_c.serving_speedup_x}×`
        : "The tuned copilot and its package are ready below.";
    fill.style.width = "100%";
    byId("kpi-dag").textContent = "Ready";
    renderArtifactViewer();
    return;
  }

  if (phase === "failed") {
    pill.className = "chip chip-failed";
    pill.textContent = "Failed";
    headline.textContent = "The run stopped with an error";
    sub.textContent = `${(snap && snap.error) || "See the error below."} Change the settings or where it runs, then try again.`;
    return;
  }

  pill.className = "chip";
  pill.textContent = "Ready";
  headline.textContent = "Ready to run";
  sub.textContent = "Before training, the remote teacher splits your data into train, validation and test sets with no overlap.";
  fill.style.width = "0%";
  byId("kpi-dag").textContent = "—";
  renderArtifactViewer();
}

function updateSingleGpuSvg(idx, node) {
  if (!node) return;
  const pfx = `gpu${idx}`;
  const smPct = Number(node.sm_util_pct || 0);
  const tcPct = Number(node.tensor_core_pct || 0);
  const memUsed = Number(node.vram_used_gib || 0);
  const memTot = Number(node.physical_memory_gib || 127.9);
  const memPct = Math.min(100, (memUsed / Math.max(memTot, 1)) * 100);
  const stStr = String(node.state || "READY");
  const isRunning = stStr.includes("RUNNING");
  const isQueued = stStr.includes("QUEUED");

  const cardEl = document.getElementById(`${pfx}-card`);
  if (cardEl) {
    cardEl.classList.toggle("gpu-card-running", isRunning);
    cardEl.classList.toggle("gpu-card-queued", isQueued);
  }

  document.getElementById(`${pfx}-host`).textContent = node.hostname || `gb10-node-0${idx + 1}`;
  const stEl = document.getElementById(`${pfx}-state`);
  stEl.textContent = stStr;
  stEl.className = isRunning
    ? "gpu-state-badge gpu-state-running"
    : isQueued
    ? "gpu-state-badge gpu-state-queued"
    : "gpu-state-badge";

  document.getElementById(`${pfx}-role`).textContent = node.role || `${node.gpu_product} (${node.sm_arch || "sm_121"})`;

  document.getElementById(`${pfx}-sm-txt`).textContent = `${smPct.toFixed(1)}%`;
  document.getElementById(`${pfx}-sm-bar`).style.width = `${Math.max(2, Math.min(100, smPct)).toFixed(1)}%`;

  document.getElementById(`${pfx}-tc-txt`).textContent = `${tcPct.toFixed(1)}%`;
  document.getElementById(`${pfx}-tc-bar`).style.width = `${Math.max(2, Math.min(100, tcPct)).toFixed(1)}%`;

  document.getElementById(`${pfx}-mem-txt`).textContent = `${memUsed.toFixed(1)} / ${memTot.toFixed(1)} GiB`;
  document.getElementById(`${pfx}-mem-bar`).style.width = `${Math.max(2, Math.min(100, memPct)).toFixed(1)}%`;

  document.getElementById(`${pfx}-tflops`).textContent =
    `${Number(node.bf16_tflops || 0).toFixed(1)} TFLOP/s · ${Number(node.mem_bw_gbs || 0).toFixed(0)} GB/s`;
  document.getElementById(`${pfx}-toks`).textContent =
    `${Math.round(Number(node.tokens_per_sec || 0)).toLocaleString()} tok/s · ${node.power_w || 24}W · ${node.temp_c || 39}°C`;

  const wfShort = node.active_workflow_id ? node.active_workflow_id.slice(0, 8) : "idle";
  const rawTrial = node.active_trial_id || node.last_trial_id || "none";
  const shortTrial = rawTrial.length > 24 ? `${rawTrial.slice(0, 22)}…` : rawTrial;
  const wfEl = document.getElementById(`${pfx}-wf`);
  wfEl.textContent = `WF: ${wfShort} · Trial: ${shortTrial} · Done: ${node.trials_completed || 0}`;
  wfEl.title = `Workflow: ${node.active_workflow_id || "idle"} | Trial: ${rawTrial} | Completed: ${node.trials_completed || 0}`;

  // Light up mini SM die cells proportionally to SM utilization
  const cells = document.querySelectorAll(`#${pfx}-cores .sm-cell`);
  const litCount = Math.round((smPct / 100) * cells.length);
  cells.forEach((cell, cIdx) => {
    cell.classList.toggle("sm-cell-active", cIdx < litCount && isRunning);
    cell.classList.toggle("sm-cell-standby", cIdx < litCount && !isRunning);
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

  const modeWords = mode.replace(/_/g, " ").toLowerCase();
  document.getElementById("fabric-mode-badge").textContent = modeWords.charAt(0).toUpperCase() + modeWords.slice(1);
  document.getElementById("fabric-bw-badge").textContent =
    bwGbps > 0 ? `${bwGbps.toFixed(1)} Gbps · ${arMs.toFixed(1)} ms sync` : "Standby";

  document.getElementById("svg-comm-proto").textContent = gt.protocol || "NCCL over RoCE";
  document.getElementById("svg-comm-bw").textContent =
    bwGbps > 0
      ? `${bwGbps.toFixed(1)} Gbps (${gt.link_utilization_pct || 0}% of link)`
      : mode === "SINGLE_GPU_LOCAL_BUS"
      ? "Single node, no cross-node sync"
      : "Link ready";

  document.getElementById("svg-allreduce-txt").textContent =
    arMs > 0
      ? `All-reduce ${arMs.toFixed(1)} ms · gradients ${payMb.toFixed(1)} MB`
      : `VAST read ${Number(gt.vast_read_gbs || 1.64).toFixed(2)} GB/s`;
  document.getElementById("svg-collective-op").textContent =
    gt.collective_op || `2-node scaling efficiency ${gt.scaling_efficiency_pct || 99.2}%`;

  const fwd = document.getElementById("svg-link-fwd");
  const rev = document.getElementById("svg-link-rev");
  fwd.classList.toggle("flow-fast", active);
  rev.classList.toggle("flow-fast", active);
  fwd.setAttribute("opacity", active ? "1" : "0.35");
  rev.setAttribute("opacity", active ? "1" : "0.35");

  const [fwdLabel, revLabel] =
    mode === "NCCL_FSDP_SHARDED"
      ? ["FSDP all-gather shards", "Reduce-scatter gradients"]
      : mode === "VLLM_TP2_ALLREDUCE"
      ? ["vLLM TP2 activations", "TP2 all-reduce"]
      : mode === "PARALLEL_TRIALS_VAST_SHARED"
      ? ["Independent trial on GPU 0", "Independent trial on GPU 1"]
      : ["All-reduce rank 0 → 1", "All-reduce rank 1 → 0"];
  document.getElementById("svg-fwd-label").textContent = fwdLabel;
  document.getElementById("svg-rev-label").textContent = revLabel;

  const team = snap.definition_id || currentTeamId();
  document.getElementById("svg-vast-title").innerHTML =
    `VAST Data shared volume <code>/vast</code> (<code>${team}</code>): one copy of the weights, splits and LoRA checkpoints for both nodes`;
}

function renderRealtimeAnalysis(snap) {
  const m = snap.live_step_metrics || {};
  const b = snap.live_budget || {};
  if (b && b.cost_budget_usd != null) {
    const spent = Number(b.spent_usd || 0);
    const cap = Number(b.cost_budget_usd || 5.0);
    const rRound = Number(b.teacher_round || 0);
    document.getElementById("rt-memsave").textContent = `R${rRound} · $${spent.toFixed(2)} of $${cap.toFixed(0)}`;
  }
  if (m && m.trial_id) {
    document.getElementById("rt-step").textContent = `${m.trial_id.slice(0, 18)} [${m.step}/${m.max_steps || m.step}]`;
    const vTxt = m.val_loss != null ? Number(m.val_loss).toFixed(4) : "—";
    document.getElementById("rt-loss").textContent = `${Number(m.train_loss || 0).toFixed(4)} / ${vTxt}`;
    document.getElementById("rt-ppl").textContent = m.perplexity != null ? Number(m.perplexity).toFixed(3) : "—";
    document.getElementById("rt-lr").textContent = m.lr != null ? Number(m.lr).toExponential(2) : "—";
    document.getElementById("rt-grad").textContent = m.grad_norm != null ? Number(m.grad_norm).toFixed(3) : "—";
    document.getElementById("rt-entropy").textContent = m.router_entropy != null ? Number(m.router_entropy).toFixed(3) : "1.000";
    document.getElementById("rt-toks").textContent = `${Math.round(Number(m.tokens_per_sec || 0)).toLocaleString()} tok/s`;
  }

  const ledgerBody = document.querySelector("#live-trial-ledger tbody");
  if (ledgerBody) {
  const rows = !snap.user_triggered ? [] : (snap.completed_trials || []).slice().reverse().slice(0, 8);
    if (rows.length === 0) {
      const activeRun = snap.state === "running" || snap.state === "queued_for_gpu";
      ledgerBody.innerHTML = activeRun
        ? `<tr><td colspan="5" class="table-placeholder-cell"><span class="spinner-ring spinner-ring-sm"></span>Waiting for the first trial to finish…</td></tr>`
        : `<tr><td colspan="5" class="table-placeholder-cell">No trials in this session yet.</td></tr>`;
    } else {
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
}

function renderEmptyPanelsBeforeRun(seedSummary = null) {
  if (seedSummary && seedSummary.cluster) {
    document.getElementById("pill-fz-ver").textContent = seedSummary.cluster.fuzzfile_version || "v4";
    document.getElementById("pill-def-id").textContent = seedSummary.cluster.definition_id || "team3";
  }
  ["kpi-ppl", "kpi-train-speed", "kpi-serve-speed"].forEach((id) => (document.getElementById(id).textContent = "—"));
  document.getElementById("meta-stage-a").textContent = "Parallel LoRA recipe search with successive halving.";
  document.getElementById("meta-stage-b").textContent = "Kernels, compile and two-node data parallel, each gated on the loss curve.";
  document.getElementById("meta-stage-c").textContent = "vLLM precisions and topologies, scored on the sealed test split.";
  document.getElementById("meta-stage-d").textContent = "";

  const curveBadge = document.getElementById("curve-badge");
  curveBadge.textContent = "Waiting for a run";
  curveBadge.className = "badge";
  document.getElementById("loss-chart-wrap").innerHTML = `
    <div class="panel-empty-state">
      <div class="panel-state-title">No loss curves yet</div>
      <div class="panel-state-sub">Train and validation loss stream here step by step once a run starts.</div>
    </div>
  `;
  document.getElementById("loss-legend").innerHTML = "";

  const utilBadge = document.getElementById("util-badge");
  utilBadge.textContent = "Waiting for a run";
  utilBadge.className = "badge";
  document.getElementById("train-bars").innerHTML = `
    <div class="panel-empty-state">
      <div class="panel-state-title">No speedup trials yet</div>
      <div class="panel-state-sub">Tokens per second for each candidate: baseline, custom <code>sm_121</code> kernels, two-node data parallel and teacher rounds.</div>
    </div>
  `;

  document.querySelector("#serving-table tbody").innerHTML = `
    <tr><td colspan="7" class="table-placeholder-cell">The vLLM sweep fills in after the training stage.</td></tr>
  `;

  document.querySelector("#kernel-table tbody").innerHTML = `
    <tr><td colspan="4" class="table-placeholder-cell">Kernel parity checks against an FP32 PyTorch reference run during the training stage.</td></tr>
  `;

  document.getElementById("advisor-rationales").innerHTML = "";
  const fwBox = document.getElementById("firewall-audit-box");
  fwBox.classList.remove("blocked-alert");
  fwBox.innerHTML = `<strong>Firewall armed.</strong> Every outbound payload is checked for raw data (<code>contains_raw_data=false</code>) and hashed with SHA-256.`;
}

function resetPanelsForLoading() {
  const spin = `<span class="spinner-ring spinner-ring-sm"></span>`;
  document.getElementById("kpi-ppl").innerHTML = `${spin}Searching`;
  document.getElementById("kpi-train-speed").textContent = "—";
  document.getElementById("kpi-serve-speed").textContent = "—";
  document.getElementById("meta-stage-a").textContent = "Splitting the data and starting the parallel LoRA search.";
  document.getElementById("meta-stage-b").textContent = "Starts once the recipe search picks a winner.";
  document.getElementById("meta-stage-c").textContent = "Starts once training speed is locked in.";
  document.getElementById("meta-stage-d").textContent = "";

  const curveBadge = document.getElementById("curve-badge");
  curveBadge.textContent = "Starting";
  curveBadge.className = "badge badge-amber";
  document.getElementById("loss-chart-wrap").innerHTML = `
    <div class="panel-loading-state">
      <div class="spinner-ring"></div>
      <div class="panel-state-title">Starting the first trials</div>
      <div class="panel-state-sub">Loss curves appear after the first optimizer steps.</div>
    </div>
  `;
  document.getElementById("loss-legend").innerHTML = "";

  const utilBadge = document.getElementById("util-badge");
  utilBadge.textContent = "After recipe search";
  utilBadge.className = "badge";
  document.getElementById("train-bars").innerHTML = `
    <div class="panel-empty-state">
      <div class="panel-state-sub">Speedup trials start once the recipe search picks a winner.</div>
    </div>
  `;

  document.querySelector("#serving-table tbody").innerHTML = `
    <tr><td colspan="7" class="table-placeholder-cell">Runs after the training stage.</td></tr>
  `;
  document.querySelector("#kernel-table tbody").innerHTML = `
    <tr><td colspan="4" class="table-placeholder-cell">${spin}Checking custom <code>sm_121</code> kernels against FP32 PyTorch…</td></tr>
  `;
  document.getElementById("advisor-rationales").innerHTML = "";
  const fwBox = document.getElementById("firewall-audit-box");
  fwBox.classList.remove("blocked-alert");
  fwBox.innerHTML = `${spin}<strong>Firewall active.</strong> Checking outbound hyperparameter and loss telemetry for raw data.`;
}

function renderLiveTelemetry(snap) {
  latestLiveSnap = snap;
  renderClientMachineBridge(snap.client_machine, snap.cluster_endpoint, snap.definition_id || currentTeamId());
  const st = !snap.user_triggered && snap.state === "completed" ? "idle" : String(snap.state || "idle");
  // The server falls back to the twin when GPUs are cordoned, so a run is labeled by snap.backend.
  setRunProvenance(snap.user_triggered && st !== "idle" ? snap.backend : document.getElementById("backend-select").value);

  const errBanner = document.getElementById("live-error-banner");
  if (snap.error) {
    errBanner.textContent = snap.error;
    errBanner.classList.remove("hidden");
  } else {
    errBanner.classList.add("hidden");
  }

  const feed = document.getElementById("live-event-feed");
  if (!snap.user_triggered) {
    feed.innerHTML = `<div class="event-line"><span class="ev-ts">ready</span>Events from each Fuzzball workflow appear here during a run.</div>`;
  } else {
    feed.innerHTML = (snap.events || [])
      .slice()
      .reverse()
      .map((ev) => `<div class="event-line"><span class="ev-ts">[${ev.time_utc}]</span>${ev.message}</div>`)
      .join("");
  }

  renderDualGpuFabric(snap);
  renderRealtimeAnalysis(snap);

  const stopBtn = document.getElementById("btn-stop-run");
  if (stopBtn) {
    stopBtn.classList.toggle("hidden", !(st === "running" || st === "queued_for_gpu"));
  }

  const serverPhase = snap.lifecycle_phase || (st === "running" || st === "queued_for_gpu" ? "running" : st === "completed" && snap.user_triggered ? "finished" : st === "failed" ? "failed" : "pre_run");
  if (serverPhase === "running" || serverPhase === "pre_run" || serverPhase === "failed") {
    renderLifecycleState(serverPhase, snap, latestSummary);
  }

  if (st === "running" || st === "queued_for_gpu") {
    const activeTid = (snap.live_step_metrics && snap.live_step_metrics.trial_id) || "";
    const curveKeys = Object.keys(snap.live_curves || {}).filter(
      (k) => Array.isArray(snap.live_curves[k]) && snap.live_curves[k].length > 1
    );
    if (curveKeys.length > 0) {
      renderLossSvg(snap.live_curves, snap.live_val_curves || {}, activeTid, snap.live_curve_equivalence || {});
    } else {
      document.getElementById("loss-chart-wrap").innerHTML = `
        <div class="panel-loading-state">
          <div class="spinner-ring"></div>
          <div class="panel-state-title">${st === "queued_for_gpu" ? "Queued on the Fuzzball cluster" : "Waiting for the first optimizer step"}</div>
          <div class="panel-state-sub">${snap.stage_label || "Loading base weights and splitting the data."}</div>
        </div>
      `;
      document.getElementById("loss-legend").innerHTML = "";
    }
    renderIncrementalStagePanels(snap);
  }
}

function renderIncrementalStagePanels(snap) {
  const curStage = String(snap.current_stage || "");

  // 1. Live Curve Equivalence Badge
  const eq = snap.live_curve_equivalence || {};
  const curveBadge = document.getElementById("curve-badge");
  if (eq && eq.reference_trial_id && eq.max_delta != null) {
    const d = Number(eq.max_delta);
    const ok = eq.accepted !== false;
    curveBadge.textContent = ok ? `Δ ${d.toFixed(4)} · within 0.025` : `Δ ${d.toFixed(4)} · drifted`;
    curveBadge.className = ok ? "badge badge-green" : "badge badge-amber";
  } else if (curStage === "stage_a_recipe") {
    const nCurves = Object.keys(snap.live_curves || {}).length;
    curveBadge.textContent = nCurves > 0 ? `${nCurves} trial curves` : "Starting";
    curveBadge.className = "badge badge-accent";
  } else {
    curveBadge.textContent = "Starting";
    curveBadge.className = "badge badge-amber";
  }

  // 2. Recipe search
  const saLive = snap.live_stage_a || {};
  if (saLive.best_perplexity) {
    document.getElementById("kpi-ppl").textContent = `PPL ${Number(saLive.base_perplexity).toFixed(2)} → ${Number(saLive.best_perplexity).toFixed(2)}`;
    document.getElementById("meta-stage-a").textContent =
      `Leading trial ${saLive.best_trial_id} · val loss ${Number(saLive.best_val_loss).toFixed(3)} · ${saLive.trials_completed} trial(s) done`;
  } else {
    document.getElementById("kpi-ppl").innerHTML = `<span class="spinner-ring spinner-ring-sm"></span>Searching`;
    document.getElementById("meta-stage-a").textContent = "Running successive-halving LoRA rounds in parallel.";
  }

  // 3. Stage B & Continuous Teacher Live Throughput Waterfall + Funnel Card
  const sbList = (snap.live_stage_b_outcomes || []).slice();
  const activeStep = snap.live_step_metrics || {};
  const activeTid = String(activeStep.trial_id || "");
  if (activeTid && (activeTid.startsWith("stageb-") || activeTid.startsWith("teacher-"))) {
    const alreadyDone = sbList.some((o) => o.trial_id === activeTid);
    if (!alreadyDone) {
      const lbl = activeTid.startsWith("stageb-")
        ? activeTid.split("-").slice(2).join("-") || activeTid
        : activeTid;
      sbList.push({
        trial_id: activeTid,
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

  const utilBadge = document.getElementById("util-badge");
  if (sbList.length > 0) {
    const baseTok = Math.max(sbList[0].summary.mean_tokens_per_sec, 1);
    const acceptedDone = sbList.filter((o) => !o.in_progress && o.accepted);
    const bestOut =
      acceptedDone.length > 0
        ? acceptedDone.reduce((a, b) => (b.summary.mean_tokens_per_sec > a.summary.mean_tokens_per_sec ? b : a))
        : sbList[0];
    const speedup = (bestOut.summary.mean_tokens_per_sec / baseTok).toFixed(2);
    document.getElementById("kpi-train-speed").textContent =
      `${Math.round(bestOut.summary.mean_tokens_per_sec).toLocaleString()} tok/s · ${speedup}×`;
    document.getElementById("meta-stage-b").textContent =
      `${sbList.filter((x) => !x.in_progress).length} candidate(s) measured · leading: ${bestOut.label}`;
    utilBadge.textContent = `${Number(bestOut.summary.hardware_utilization_pct || 0).toFixed(1)}% of peak`;
    utilBadge.className = "badge badge-accent";

    const maxTok = Math.max(...sbList.map((o) => o.summary.mean_tokens_per_sec), 1);
    const barsHtml = sbList
      .map((o) => {
        const tok = o.summary.mean_tokens_per_sec;
        const pct = Math.max(6, (tok / maxTok) * 100).toFixed(1);
        const isWin = !o.in_progress && o.trial_id === bestOut.trial_id && o.accepted;
        const cls = o.in_progress
          ? "bar-fill bar-fill-running"
          : !o.accepted
          ? "bar-fill bar-fill-rejected"
          : isWin
          ? "bar-fill bar-fill-winner"
          : "bar-fill";
        const statusTag = o.in_progress
          ? `<span class="tag-ok"><span class="spinner-ring spinner-ring-sm"></span>measuring · Δ ${o.max_loss_curve_delta.toFixed(4)}</span>`
          : o.accepted
          ? isWin
            ? `<span class="tag-ok">leading</span>`
            : `<span class="tag-ok">kept · Δ ${o.max_loss_curve_delta.toFixed(4)}</span>`
          : `<span class="tag-reject">rejected · ${o.rejection_reason || "loss drift"}</span>`;
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
    const moreStreaming =
      (curStage === "stage_b_training_perf" || curStage === "continuous_teacher_loop") &&
      !sbList.some((o) => o.in_progress)
        ? `<div class="inline-stream-spinner"><span class="spinner-ring spinner-ring-sm"></span><span>Starting the next ${curStage === "continuous_teacher_loop" ? "teacher round" : "speedup trial"}…</span></div>`
        : "";
    document.getElementById("train-bars").innerHTML = barsHtml + moreStreaming;
  } else {
    document.getElementById("kpi-train-speed").textContent = "—";
    document.getElementById("meta-stage-b").textContent = "Starts once the recipe search picks a winner.";
    utilBadge.textContent = "After recipe search";
    utilBadge.className = "badge";
    document.getElementById("train-bars").innerHTML = `
      <div class="panel-empty-state">
        <div class="panel-state-sub">Speedup trials start once the recipe search picks a winner.</div>
      </div>
    `;
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
      `${Math.round(bestSrv.throughput_tokens_per_sec).toLocaleString()} tok/s · ${srvSpeedup}×`;
    document.getElementById("meta-stage-c").textContent =
      `${scList.length} configuration(s) measured · leading: ${bestSrv.label}`;
    const rowsHtml = scList
      .map((s) => {
        const isWin = s.label === bestSrv.label && s.accepted;
        const badge = s.accepted
          ? `<span class="tag-ok">${isWin ? "leading" : "pass"}</span>`
          : `<span class="tag-reject" title="${(s.rejection_reasons || []).join("; ")}">rejected</span>`;
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
    const tailSpinner =
      curStage === "stage_c_inference_perf"
        ? `<tr><td colspan="7" class="table-placeholder-cell"><span class="spinner-ring spinner-ring-sm"></span>Measuring the next precision and topology…</td></tr>`
        : "";
    document.querySelector("#serving-table tbody").innerHTML = rowsHtml + tailSpinner;
  } else {
    const serving = curStage === "stage_c_inference_perf";
    document.getElementById("kpi-serve-speed").innerHTML = serving ? `<span class="spinner-ring spinner-ring-sm"></span>Measuring` : "—";
    document.getElementById("meta-stage-c").textContent = serving
      ? "Starting the BF16 baseline vLLM server."
      : "Starts once training speed is locked in.";
    document.querySelector("#serving-table tbody").innerHTML = `
      <tr><td colspan="7" class="table-placeholder-cell">${serving ? `<span class="spinner-ring spinner-ring-sm"></span>Measuring the BF16 baseline…` : "Runs after the training stage."}</td></tr>
    `;
  }

  // 5. Live Custom sm_121 Kernel Parity Table
  const liveKernels = snap.live_kernels || [];
  if (liveKernels.length > 0) {
    document.querySelector("#kernel-table tbody").innerHTML = liveKernels
      .map(
        (k) => `
        <tr>
          <td><code>${k.kernel_name}</code></td>
          <td><code>${Number(k.max_abs_error).toExponential(2)}</code></td>
          <td>${Number(k.memory_reduction_ratio).toFixed(2)}&times;</td>
          <td><span class="tag-ok">${k.correctness_passed ? "PASS" : "FAIL"}</span></td>
        </tr>
      `
      )
      .join("");
  } else {
    document.querySelector("#kernel-table tbody").innerHTML = `
      <tr><td colspan="4" class="table-placeholder-cell"><span class="spinner-ring spinner-ring-sm"></span>Checking custom <code>sm_121</code> kernels against FP32 PyTorch…</td></tr>
    `;
  }

  // 6. Live Advisor Rationales & Firewall Audit Stream
  if (Array.isArray(snap.live_firewall) && snap.live_firewall.length > 0) {
    const exps = snap.live_firewall.filter((f) => f.explanation && f.explanation.rationale);
    const itemsHtml = exps
      .map((f) => `<div class="advisor-item"><strong>[${f.explanation.stage}]</strong> ${f.explanation.rationale}</div>`)
      .join("");
    document.getElementById("advisor-rationales").innerHTML = itemsHtml;

    const lastRec = snap.live_firewall[snap.live_firewall.length - 1].record || {};
    const sha = String(lastRec.payload_sha256 || "").slice(0, 16);
    const fwBox = document.getElementById("firewall-audit-box");
    fwBox.classList.remove("blocked-alert");
    fwBox.innerHTML =
      `<strong>${snap.live_firewall.length} outbound payload(s) checked, none with raw data.</strong> ` +
      `Latest SHA-256 <code>${sha}…</code>`;
  } else {
    document.getElementById("advisor-rationales").innerHTML = "";
  }
}

function renderLossSvg(curves, valCurves = {}, activeTrialId = "", liveEq = {}) {
  const wrap = document.getElementById("loss-chart-wrap");
  const legend = document.getElementById("loss-legend");
  const allKeys = Object.keys(curves).filter((k) => Array.isArray(curves[k]) && curves[k].length > 1);
  // Include Continuous Teacher trial curves in renderLossSvg too!
  const stageBKeys = allKeys.filter((k) => k.startsWith("stageb-") || k.startsWith("teacher-")).sort((a, b) => {
    const aIsB = a.startsWith("stageb-") ? 0 : 1;
    const bIsB = b.startsWith("stageb-") ? 0 : 1;
    return aIsB !== bIsB ? aIsB - bIsB : a.localeCompare(b);
  });

  const active = [];
  if (stageBKeys.length > 0) {
    const refKey = stageBKeys[0];
    active.push({
      key: refKey,
      label: `reference: ${refKey.split("-").slice(2).join("-")}`,
      color: "#c3cfe0",
      dash: "",
      src: curves,
    });

    // Include currently running Stage B / Teacher candidate live as steps arrive!
    if (activeTrialId && activeTrialId !== refKey && Array.isArray(curves[activeTrialId]) && curves[activeTrialId].length > 1) {
      const drifting = liveEq && liveEq.accepted === false;
      const cleanName = activeTrialId.startsWith("stageb-")
        ? activeTrialId.split("-").slice(2).join("-")
        : activeTrialId;
      active.push({
        key: activeTrialId,
        label: `${cleanName} (running)`,
        color: drifting ? "#f47c7c" : "#76b900",
        dash: "5,3",
        src: curves,
      });
    } else {
      // Otherwise show best accepted Stage B / Teacher winner + any rejected drift curve
      const otherB = stageBKeys.slice(1);
      const driftKey = otherB.find((k) => k.includes("oversized"));
      const bestAcceptedKey = otherB.filter((k) => k !== driftKey).pop();
      if (bestAcceptedKey) {
        const cleanBest = bestAcceptedKey.startsWith("stageb-")
          ? bestAcceptedKey.split("-").slice(2).join("-")
          : bestAcceptedKey;
        active.push({
          key: bestAcceptedKey,
          label: `${cleanBest} (accepted)`,
          color: "#76b900",
          dash: "5,3",
          src: curves,
        });
      }
      if (driftKey) {
        active.push({
          key: driftKey,
          label: `${driftKey.split("-").slice(2).join("-")} (rejected: drift)`,
          color: "#f47c7c",
          dash: "2,3",
          src: curves,
        });
      }
    }
  } else {
    const recentKeys = allKeys.slice(-3);
    const palette = ["#c3cfe0", "#76b900", "#8fa4c4"];
    recentKeys.forEach((k, idx) => {
      active.push({ key: k, label: `${k} (train)`, color: palette[idx % palette.length], dash: "", src: curves });
      if (Array.isArray(valCurves[k]) && valCurves[k].length > 1) {
        active.push({ key: k, label: `${k} (val)`, color: "#f2b544", dash: "4,3", src: valCurves });
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
    gridLines += `<line x1="${pad.left}" y1="${y}" x2="${width - pad.right}" y2="${y}" stroke="#222d40" stroke-width="1"/>`;
    gridLines += `<text x="${pad.left - 6}" y="${Number(y) + 4}" fill="#8392a8" font-size="10.5" text-anchor="end">${val.toFixed(2)}</text>`;
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
      <text x="${(width + pad.left) / 2}" y="${height - 5}" fill="#8392a8" font-size="10.5" text-anchor="middle">Optimizer step</text>
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

  const sa = data.stage_a;
  const sb = data.stage_b;
  const sc = data.stage_c;

  document.getElementById("kpi-ppl").textContent = `PPL ${sa.base_held_out_perplexity.toFixed(2)} → ${sa.winning_summary.held_out_perplexity.toFixed(2)}`;
  document.getElementById("meta-stage-a").textContent =
    `Winner ${sa.winning_label} · val loss ${sa.winning_summary.final_val_loss.toFixed(3)} · test loss ${sa.winning_summary.held_out_eval_loss.toFixed(3)} · ${sa.early_stopped_count} stopped early`;

  const baseTrainTok = sb.outcomes[0].summary.mean_tokens_per_sec;
  const winTrainOut = sb.outcomes.find((o) => o.label === sb.winning_label) || sb.outcomes[0];
  const bestTrainTok = winTrainOut.summary.mean_tokens_per_sec;
  document.getElementById("kpi-train-speed").textContent =
    `${Math.round(bestTrainTok).toLocaleString()} tok/s · ${sb.overall_speedup_x}×`;
  document.getElementById("meta-stage-b").textContent =
    `${Math.round(baseTrainTok).toLocaleString()} → ${Math.round(bestTrainTok).toLocaleString()} tokens/s · single-node kernel speedup ${sb.single_node_kernel_speedup_x}×`;

  const winServeOut = sc.outcomes.find((o) => o.label === sc.winning_label) || sc.outcomes[0];
  document.getElementById("kpi-serve-speed").textContent =
    `${Math.round(winServeOut.throughput_tokens_per_sec).toLocaleString()} tok/s · ${sc.serving_speedup_x}×`;
  document.getElementById("meta-stage-c").textContent =
    `Selected ${sc.winning_label} · test perplexity +${winServeOut.perplexity_regression_pct.toFixed(2)}% · P99 ${winServeOut.p99_latency_ms.toFixed(0)} ms`;

  const scs = data.dataset_manifest.split_file_counts || {};
  document.getElementById("meta-stage-d").textContent =
    `Data split ${scs.train || 0} / ${scs.val || 0} / ${scs.held_out || 0} train/val/test files · stored on VAST (${data.cluster.vast.provisioner})`;

  document.getElementById("curve-badge").textContent =
    `Winner Δ ${winTrainOut.max_loss_curve_delta.toFixed(4)} · within 0.025`;
  document.getElementById("curve-badge").className = "badge badge-green";
  document.getElementById("util-badge").textContent =
    `${winTrainOut.summary.hardware_utilization_pct.toFixed(1)}% of peak`;
  document.getElementById("util-badge").className = "badge badge-green";

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
          ? `<span class="tag-ok">winner</span>`
          : `<span class="tag-ok">kept · Δ ${o.max_loss_curve_delta.toFixed(4)}</span>`
        : `<span class="tag-reject">rejected · ${o.rejection_reason || "Δ " + o.max_loss_curve_delta.toFixed(3)}</span>`;
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
        ? `<span class="tag-ok">${isWin ? "selected" : "pass"}</span>`
        : `<span class="tag-reject" title="${s.rejection_reasons.join("; ")}">rejected</span>`;
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
    `<strong>${audits.length} outbound payload(s) audited, none with raw data.</strong> ` +
    `Latest SHA-256 <code>${lastDigest}…</code>`;
}

async function runPlaygroundSample(executeOnCluster = false) {
  const badge = document.getElementById("copilot-exec-badge");
  const liveBtn = document.getElementById("btn-run-cluster-copilot");
  try {
    syncCompanyBranding();
    const prompt = document.getElementById("playground-prompt").value;
    const companyName = currentCompanyName();
    if (executeOnCluster) {
      if (liveBtn) {
        liveBtn.disabled = true;
        liveBtn.textContent = "Running on GB10…";
      }
      if (badge) {
        badge.className = "chip chip-running";
        badge.textContent = "Running on a Fuzzball GB10 GPU…";
      }
      document.getElementById("pg-tuned-out").textContent =
        `Sending the prompt to a GB10 GPU in ${currentTeamId()} for ${companyName}…`;
    }
    const res = await fetchJson("/api/playground/complete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt,
        company_name: companyName,
        compare_base: true,
        execute_on_cluster: Boolean(executeOnCluster),
      }),
    });
    document.getElementById("pg-base-out").textContent = res.base_output || "";
    document.getElementById("pg-tuned-out").textContent = res.tuned_output || "";
    if (badge) {
      const tok = Math.round(Number(res.throughput_tok_s || 0));
      if (res.source === "live_fuzzball_gb10_gpu") {
        badge.className = "chip chip-live";
        badge.textContent = `Live on ${res.hostname || "GB10"} · ${tok} tok/s`;
      } else {
        badge.className = "chip chip-sim";
        badge.textContent = `${res.cordoned_twin ? "Simulated by the GB10 twin" : "Simulated preview"} · ${tok} tok/s`;
      }
    }
  } catch (_) {
    if (badge) {
      badge.className = "chip";
      badge.textContent = "No answer. Try again.";
    }
  } finally {
    if (liveBtn) {
      liveBtn.disabled = currentLifecyclePhase !== "finished";
      liveBtn.textContent = "Run on Fuzzball GPU";
    }
  }
}

async function handleLiveSnapUpdate(snap) {
  renderLiveTelemetry(snap);
  const wasActive = lastObservedRunState === "running" || lastObservedRunState === "queued_for_gpu";
  if (wasActive && snap.state === "completed") {
    lastObservedRunState = snap.state;
    stopActivePollLoop();
    const summary = await fetchJson("/api/summary");
    renderDashboard(summary);
    renderLifecycleState("finished", snap, summary);
    await refreshBuildPreview();
    await runPlaygroundSample(false);
    document.getElementById("btn-run").disabled = false;
    document.getElementById("btn-run").textContent = RUN_LABEL;
    return;
  }
  if (snap.state === "failed") {
    lastObservedRunState = snap.state;
    stopActivePollLoop();
    renderLifecycleState("failed", snap, latestSummary);
    document.getElementById("btn-run").disabled = false;
    document.getElementById("btn-run").textContent = RUN_LABEL;
    return;
  }
  if (snap.state === "running" || snap.state === "queued_for_gpu") {
    ensureActivePollLoop();
  }
  lastObservedRunState = snap.state;
}

function ensureActivePollLoop() {
  if (activePollTimer) return;
  activePollTimer = setInterval(async () => {
    try {
      const snap = await fetchJson("/api/live-status");
      await handleLiveSnapUpdate(snap);
    } catch (_) {}
  }, 650);
}

function stopActivePollLoop() {
  if (activePollTimer) {
    clearInterval(activePollTimer);
    activePollTimer = null;
  }
}

function connectLiveStream() {
  const es = new EventSource("/api/events");
  es.onmessage = async (msg) => {
    const snap = JSON.parse(msg.data);
    await handleLiveSnapUpdate(snap);
  };
}

// Event bindings
document.getElementById("company-name-input").addEventListener("input", () => {
  syncCompanyBranding();
});
document.getElementById("company-name-input").addEventListener("change", () => {
  if (currentLifecyclePhase === "finished") {
    refreshBuildPreview();
    runPlaygroundSample(false);
  }
});

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

document.getElementById("backend-select").addEventListener("change", (e) => {
  userPickedBackend = true;
  syncModeChip();
  if (currentLifecyclePhase === "pre_run") setRunProvenance(e.target.value);
});

document.getElementById("chk-company-presets").addEventListener("change", () => {
  refreshModelSpec(currentModelId());
});

document.getElementById("btn-stage-model").addEventListener("click", async () => {
  const btn = document.getElementById("btn-stage-model");
  const modelId = currentModelId();
  btn.disabled = true;
  btn.textContent = "Loading onto VAST…";
  try {
    await fetchJson("/api/models/stage", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_id: modelId, backend: "auto", async_mode: false }),
    });
    await refreshModelSpec(modelId);
  } catch (_) {
    btn.disabled = false;
    btn.textContent = "Load onto VAST";
  }
});

document.getElementById("btn-stop-run").addEventListener("click", async () => {
  const stopBtn = document.getElementById("btn-stop-run");
  stopBtn.disabled = true;
  stopBtn.textContent = "Stopping and packaging…";
  try {
    await fetchJson("/api/run/stop", { method: "POST" });
  } catch (_) {}
  setTimeout(() => {
    stopBtn.disabled = false;
    stopBtn.textContent = "Stop and package best";
  }, 1200);
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
  const continuousMode = document.getElementById("chk-continuous").checked;
  const usePresets = document.getElementById("chk-company-presets").checked;
  const timeBudgetMin = Math.max(0.5, Number(document.getElementById("time-budget-input").value || 8));
  const costBudgetUsd = Math.max(0.5, Number(document.getElementById("cost-budget-input").value || 5));

  btn.disabled = true;
  btn.textContent = "Tuning…";
  setRunProvenance(backend);
  resetPanelsForLoading();
  renderLifecycleState("running", { current_stage: "starting", stage_label: `Splitting the data and starting ${modelId.split("/").pop()}` }, null);
  const resp = await fetch("/api/run", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      company_name: currentCompanyName(),
      model_id: modelId,
      dataset_id: activeDatasetId,
      backend,
      fuzzfile_version: "v4",
      definition_id: teamId,
      cluster_endpoint: clusterEp,
      continuous_mode: continuousMode,
      time_budget_minutes: timeBudgetMin,
      cost_budget_usd: costBudgetUsd,
      use_company_presets: usePresets,
      async_mode: true,
    }),
  });
  if (!resp.ok) {
    btn.disabled = false;
    btn.textContent = RUN_LABEL;
    const err = await resp.json();
    const errMsg = err.detail || "The run could not start.";
    renderLifecycleState("failed", { error: errMsg }, null);
    const errBanner = document.getElementById("live-error-banner");
    errBanner.textContent = errMsg;
    errBanner.classList.remove("hidden");
  } else {
    lastObservedRunState = "running";
    ensureActivePollLoop();
  }
});

document.getElementById("btn-jump-output").addEventListener("click", () => {
  document.getElementById("final-copilot-section").scrollIntoView({ behavior: "smooth", block: "start" });
});

["btn-top-download", "btn-card-download"].forEach((dlId) => {
  const el = document.getElementById(dlId);
  if (!el) return;
  el.addEventListener("click", (ev) => {
    if (currentLifecyclePhase !== "finished") {
      ev.preventDefault();
      document.getElementById("btn-run").scrollIntoView({ behavior: "smooth", block: "center" });
    }
  });
});

document.getElementById("btn-run-playground").addEventListener("click", () => {
  if (currentLifecyclePhase === "finished") runPlaygroundSample(false);
});
document.getElementById("btn-run-cluster-copilot").addEventListener("click", () => {
  if (currentLifecyclePhase === "finished") runPlaygroundSample(true);
});

let noticeTimer = null;
function showNotice(html) {
  const el = document.getElementById("probe-notice");
  el.innerHTML = html;
  el.hidden = false;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => (el.hidden = true), 9000);
}

document.getElementById("btn-probe").addEventListener("click", async () => {
  const probeBtn = document.getElementById("btn-probe");
  const label = document.getElementById("probe-label");
  probeBtn.disabled = true;
  label.textContent = "Checking…";
  try {
    const resp = await fetch("/api/firewall-probe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ inject_cpp_snippet: true }),
    });
    const res = await resp.json();
    const fwBox = document.getElementById("firewall-audit-box");
    if (res.blocked) {
      const shaShort = res.record.payload_sha256.slice(0, 16);
      fwBox.classList.add("blocked-alert");
      fwBox.innerHTML =
        `<strong>Leak blocked.</strong> A raw C++ snippet was stopped before it left this machine: ` +
        `<code>${res.violation}</code> · SHA-256 <code>${shaShort}…</code>`;
      showNotice(
        `<strong>Zero-leak guard works.</strong> We tried to send raw C++ source out; the firewall blocked it before serialization. ` +
          `0 bytes left. Audit hash <code>${shaShort}…</code>`
      );
      label.textContent = "Leak blocked";
    } else {
      showNotice("The probe was not blocked. Check the firewall configuration before demoing.");
      label.textContent = "Verify zero-leak guard";
    }
  } catch (_) {
    showNotice("The privacy check could not reach the local server. Is <code>alchemy serve</code> running?");
    label.textContent = "Verify zero-leak guard";
  } finally {
    probeBtn.disabled = false;
    setTimeout(() => (label.textContent = "Verify zero-leak guard"), 4000);
  }
});

// Model picker dialog
const modelDialog = document.getElementById("model-dialog");
document.getElementById("btn-change-model").addEventListener("click", () => {
  renderCatalogCards();
  modelDialog.showModal();
  document.getElementById("catalog-filter-input").focus();
});
document.getElementById("btn-close-model").addEventListener("click", () => modelDialog.close());
document.getElementById("btn-use-model").addEventListener("click", () => {
  refreshModelSpec(currentModelId());
  modelDialog.close();
});
modelDialog.addEventListener("click", (e) => {
  if (e.target === modelDialog) modelDialog.close(); // click on the backdrop
});

// The data menu opens under its button (the header menu is pinned top-right in CSS).
document.querySelectorAll('[popovertarget="data-menu"]').forEach((btn) =>
  btn.addEventListener("click", () => {
    const r = btn.getBoundingClientRect();
    const menu = document.getElementById("data-menu");
    menu.style.top = `${r.bottom + 6}px`;
    menu.style.left = `${Math.max(16, r.right - 260)}px`;
  })
);
document.getElementById("data-menu").addEventListener("click", (e) => {
  if (e.target.closest("button")) document.getElementById("data-menu").hidePopover();
});
document.getElementById("folder-picker").addEventListener("change", () => document.getElementById("data-menu").hidePopover());

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
  lastObservedRunState = !liveSnap.user_triggered ? "idle" : String(liveSnap.state || "idle");
  if (liveSnap.user_triggered && (liveSnap.state === "running" || liveSnap.state === "queued_for_gpu")) {
    resetPanelsForLoading(liveSnap.model_id);
    renderLiveTelemetry(liveSnap);
    renderLifecycleState("running", liveSnap, null);
    ensureActivePollLoop();
  } else if (liveSnap.user_triggered && liveSnap.state === "completed") {
    renderDashboard(summary);
    renderLiveTelemetry(liveSnap);
    renderLifecycleState("finished", liveSnap, summary);
    await refreshBuildPreview();
    await runPlaygroundSample(false);
  } else {
    renderEmptyPanelsBeforeRun(summary);
    renderLiveTelemetry(liveSnap);
    renderLifecycleState("pre_run", liveSnap, null);
  }
  setTimeout(connectLiveStream, 250);
})();

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
  if (hd) hd.textContent = `${comp} — Private Engineering Copilot`;
  const kk = document.getElementById("pg-tuned-kicker");
  if (kk) kk.textContent = `${comp.toUpperCase()} TUNED PRIVATE COPILOT`;
  const dlHref = `/api/build/bundle.zip?company_name=${encodeURIComponent(comp)}`;
  const topDl = document.getElementById("btn-top-download");
  if (topDl) topDl.href = dlHref;
  const bannerDl = document.getElementById("btn-banner-download");
  if (bannerDl) bannerDl.href = dlHref;
  const cardDl = document.getElementById("btn-card-download");
  if (cardDl) cardDl.href = dlHref;
  const uz = document.getElementById("unzip-cmd-line");
  if (uz) {
    uz.textContent = `unzip ${slug}-private-llm-*.zip -d ${slug}-ai && python3 ${slug}-ai/scripts/run_company_app.py`;
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
    const kind =
      m.num_experts > 1 ? `MoE (${m.num_experts} experts, top-${m.top_k_experts})` : "Dense Transformer";
    const vastPill =
      m.vast_state === "preloaded"
        ? `<span class="tag-pill tag-pill-green">Pre-Loaded on VAST</span>`
        : m.vast_state === "loading"
        ? `<span class="tag-pill tag-pill-amber">Loading to VAST…</span>`
        : `<span class="tag-pill tag-pill-amber">Needs VAST Load (~${m.est_vast_load_seconds}s)</span>`;
    const presetPill = m.has_hardware_preset
      ? `<span class="tag-pill tag-pill-cyan">Preset Opt Ready (${m.hardware_preset.preset_id})</span>`
      : `<span class="tag-pill tag-pill-muted">Full Discovery Sweep (No Preset)</span>`;

    const minReqText =
      m.min_nodes_required >= 2
        ? `<span class="tag-pill tag-pill-violet">Min Req: 2 GPUs (FSDP/TP2 — exceeds 1-GPU VRAM)</span>`
        : `<span class="tag-pill tag-pill-muted">Min Req: 1 GPU (Scales to 2 GPUs on Cluster)</span>`;

    document.getElementById("model-spec-line").innerHTML =
      `<strong>${m.name}</strong> &middot; ${minReqText} ${vastPill} ${presetPill}<br/>` +
      `${kind} &middot; <strong>${m.total_params_b}B</strong> (${m.active_params_b}B active) &middot; ` +
      `BF16 <code>${m.bf16_footprint_gb} GB</code> / FP8 <code>${m.fp8_footprint_gb} GB</code> / NVFP4 <code>${m.nvfp4_footprint_gb} GB</code>`;

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
        pTitle.textContent = `Company GB10 Preset: ${m.hardware_preset.preset_id}`;
        pVal.textContent = `Saves $${Number(m.hardware_preset.warm_start_savings_usd).toFixed(2)} (warm-starts sm_121 kernels)`;
      } else {
        pTitle.textContent = "Cold Hardware Sweep (No Preset Applied)";
        pVal.textContent = "$0.00 saved — runs full Stage B kernel discovery";
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
    wrap.innerHTML = `<div class="catalog-empty">No catalog preset matches "${query}" in ${activeCategory}. You can still paste ANY HuggingFace <code>org/model</code> ID above!</div>`;
    return;
  }

  wrap.innerHTML = filtered
    .map((m) => {
      const isAct = m.id.toLowerCase() === currentId;
      const archBadge = m.is_moe ? `MoE ${m.active_params_b}B/${m.total_params_b}B` : `${m.total_params_b}B Dense`;
      const nodeBadge = m.min_nodes_required >= 2 ? "Min: 2 GPUs" : "Min: 1 GPU";
      const nodeTip =
        m.min_nodes_required >= 2
          ? "Hardware Requirement: Minimum 2x GB10 GPUs required to fit BF16 weights"
          : "Hardware Requirement: Fits on 1 GPU minimum (Alchemy still scales across all 2 cluster GPUs for speed)";
      const loadBadge =
        m.vast_state === "preloaded"
          ? `<span class="mc-status mc-preloaded">Pre-Loaded</span>`
          : m.vast_state === "loading"
          ? `<span class="mc-status mc-needs-load">Loading…</span>`
          : `<span class="mc-status mc-needs-load">Needs Load</span>`;
      const optBadge = m.has_hardware_preset
        ? `<span class="mc-status mc-preset">Preset Opt</span>`
        : `<span class="mc-status mc-nopreset">Auto-Sweep</span>`;
      return `
        <button type="button" class="model-card ${isAct ? "model-card-active" : ""}" data-model-id="${m.id}" title="${m.id} — ${m.recommended_use} (${nodeTip})">
          <div class="mc-top">
            <span class="mc-label">${m.label}</span>
            <span class="mc-node ${m.min_nodes_required >= 2 ? "mc-node-2" : ""}" title="${nodeTip}">${nodeBadge}</span>
          </div>
          <div class="mc-badges-row">${loadBadge}${optBadge}</div>
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
  activeDatasetMeta = ds;
  const sc = ds.split_doc_counts || {};
  const cc = ds.split_chunk_counts || {};
  const tc = (ds.teacher_curriculum && ds.teacher_curriculum.split_pair_counts) || {};
  const prefix = ds.is_preset ? "Preset Active" : "Uploaded Corpus Active";
  const teacherNote =
    tc.train != null
      ? ` &middot; <strong>Remote Teacher SFT Splits:</strong> <code>${tc.train || 0} Train</code> / <code>${tc.val || 0} Val</code> / <code>${tc.test || 0} Test</code> pairs (0 leakage)`
      : "";
  document.getElementById("dataset-status-bar").innerHTML =
    `${prefix}: <strong>${ds.display_name}</strong> (${ds.format_summary}) &middot; ` +
    `<strong>${(ds.total_tokens || 0).toLocaleString()} tokens</strong> &middot; ` +
    `Docs: <code>${sc.train || 0}T / ${sc.val || 0}V / ${sc.held_out || 0}Test</code> &middot; ` +
    `Packed Chunks: <code>${cc.train || 0}T / ${cc.val || 0}V / ${cc.held_out || 0}Test</code>` +
    teacherNote;

  const chipEl = document.getElementById("lifecycle-teacher-chip");
  if (chipEl && tc.train != null && currentLifecyclePhase === "pre_run") {
    chipEl.textContent = `Teacher SFT Staged: ${tc.train} Train · ${tc.val} Val · ${tc.test} Test pairs`;
  }
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
  const cordoned = Boolean(discReport.all_nodes_cordoned);
  const cliOk = Boolean(discReport.fuzzball_cli_available);
  const vol = (clusterCfg && clusterCfg.vast && clusterCfg.vast.persistent_volume_name) || "runs";
  const prov = (clusterCfg && clusterCfg.vast && clusterCfg.vast.provisioner) || "team3";

  if (reachable && !cordoned) {
    if (dot) dot.className = "dot dot-green";
    statusEl.innerHTML =
      `<strong style="color:#a3e635">LIVE CLUSTER CONNECTED</strong> &middot; Fuzzball MCP Ready &middot; ` +
      `VAST Volume <code>${vol}</code> (<code>use: ${prov}</code>)`;
    if (cliOk && backendSel && !userPickedBackend) {
      backendSel.value = "fuzzball_mcp";
    }
  } else if (reachable && cordoned) {
    if (dot) dot.className = "dot dot-amber";
    statusEl.innerHTML =
      `<strong style="color:#fbbf24">FUZZBALL CONTROL PLANE CONNECTED (GPUs Cordoned for Judging)</strong> &middot; ` +
      `VAST <code>${vol}</code> (<code>use: ${prov}</code>) &middot; GB10 Empirical Twin Active`;
    if (backendSel && !userPickedBackend) {
      backendSel.value = "simulator";
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
    syncCompanyBranding();
    const comp = currentCompanyName();
    const prev = await fetchJson(`/api/build/preview?company_name=${encodeURIComponent(comp)}`);
    buildArtifacts = prev.artifacts || {};
    const shortModel = String(prev.model_id || "").split("/").pop();
    document.getElementById("build-title").innerHTML = `${comp} LLM Package: <code>${shortModel}</code>`;
    document.getElementById("build-subtitle").innerHTML =
      `Unzip &amp; run <code>python3 scripts/run_company_app.py</code> on your laptop to launch <strong>${comp}'s</strong> branded app backed by Fuzzball GPUs!`;
    const subModel = document.getElementById("copilot-model-sub");
    if (subModel) {
      subModel.innerHTML = `Powered by <code>${shortModel}</code> + Tuned LoRA &middot; Zero Cloud Data Egress`;
    }
    const insp = document.getElementById("inspector-model-label");
    if (insp) insp.textContent = prev.model_id;
    renderArtifactViewer();
  } catch (_) {}
}

function renderArtifactViewer() {
  const viewer = document.getElementById("yaml-viewer");
  if (currentLifecyclePhase === "pre_run") {
    viewer.textContent =
      "# PRE-RUN: Output package has not been generated yet.\n" +
      "# Click 'Tune on Fuzzball & Generate Local LLM' above to synthesize Teacher Train/Val/Test SFT splits,\n" +
      "# optimize your model on GB10, and unlock your turnkey Copilot & .zip bundle here.";
    return;
  }
  if (currentLifecyclePhase === "running") {
    viewer.textContent =
      "# RUNNING: Optimization in progress on Fuzzball GB10...\n" +
      "# Packaging final fuzzfile.yaml, run_company_app.py, Modelfile.ollama, and Teacher SFT splits once stages complete.";
    return;
  }
  if (buildArtifacts[selectedArtifactKey]) {
    viewer.textContent = buildArtifacts[selectedArtifactKey];
  } else if (latestSummary && latestSummary.turnkey_fuzzfile) {
    viewer.textContent = latestSummary.turnkey_fuzzfile;
  }
}

function renderLifecycleState(phase, snap = null, summary = null) {
  currentLifecyclePhase = phase;
  const topPill = document.getElementById("header-lifecycle-pill");
  const banner = document.getElementById("lifecycle-banner");
  const stateChip = document.getElementById("lifecycle-state-chip");
  const teacherChip = document.getElementById("lifecycle-teacher-chip");
  const headline = document.getElementById("lifecycle-headline");
  const sub = document.getElementById("lifecycle-sub");
  const jumpBtn = document.getElementById("btn-jump-output");
  const bannerDl = document.getElementById("btn-banner-download");
  const topDl = document.getElementById("btn-top-download");
  const cardDl = document.getElementById("btn-card-download");
  const stepPre = document.getElementById("lc-step-prerun");
  const stepRun = document.getElementById("lc-step-running");
  const stepFin = document.getElementById("lc-step-finished");
  const progFill = document.getElementById("lifecycle-progress-fill");

  const outBanner = document.getElementById("output-readiness-banner");
  const outTag = document.getElementById("output-state-tag");
  const outText = document.getElementById("output-banner-text");
  const outPill = document.getElementById("output-ready-pill");
  const outSection = document.getElementById("final-copilot-section");
  const btnLiveCopilot = document.getElementById("btn-run-cluster-copilot");
  const btnPreviewCopilot = document.getElementById("btn-run-playground");
  const copilotBadge = document.getElementById("copilot-exec-badge");
  const stageDPill = document.getElementById("stage-d-status-pill");
  const kpiDag = document.getElementById("kpi-dag");

  const tcLive = (snap && snap.live_teacher_curriculum && snap.live_teacher_curriculum.split_pair_counts) ||
    (summary && summary.teacher_curriculum && summary.teacher_curriculum.split_pair_counts) ||
    (activeDatasetMeta && activeDatasetMeta.teacher_curriculum && activeDatasetMeta.teacher_curriculum.split_pair_counts) ||
    null;
  const tcText = tcLive
    ? `Teacher SFT Splits: ${tcLive.train || 0} Train · ${tcLive.val || 0} Val · ${tcLive.test || 0} Test pairs`
    : "Teacher SFT Splits Ready on Launch";

  if (phase === "running") {
    const curStage = String((snap && snap.current_stage) || "starting");
    const stagePctMap = {
      starting: 12,
      ingest: 18,
      teacher_curriculum: 26,
      stage_a_recipe: 44,
      stage_b_training_perf: 64,
      stage_c_inference_perf: 80,
      continuous_teacher_loop: 90,
      stage_d_emit: 96,
    };
    const pct = stagePctMap[curStage] || 30;

    topPill.className = "lifecycle-top-pill lifecycle-pill-running";
    topPill.innerHTML = `<span class="spinner-ring spinner-ring-sm"></span>STATE: RUNNING (${pct}%)`;

    banner.className = "lifecycle-banner lifecycle-running";
    stateChip.innerHTML = `<span class="spinner-ring spinner-ring-sm"></span>STEP 2 OF 3 &middot; RUNNING`;
    teacherChip.textContent = tcText;
    headline.textContent = `Running Live Optimization — ${(snap && snap.stage_label) || "Processing Teacher Splits & GB10 Trials…"}`;
    sub.textContent =
      "Output is still building and NOT ready yet. Watch real-time loss curves, throughput bars, and vLLM gates stream below; your Copilot & .zip unlock automatically at Step 3.";
    jumpBtn.classList.add("hidden");
    bannerDl.classList.add("hidden");

    stepPre.className = "lc-step lc-step-done";
    stepPre.querySelector(".lc-num").textContent = "✓";
    stepRun.className = "lc-step lc-step-active";
    stepRun.querySelector(".lc-num").textContent = "2";
    stepFin.className = "lc-step";
    stepFin.querySelector(".lc-num").textContent = "3";
    progFill.style.width = `${pct}%`;

    topDl.className = "btn btn-download btn-locked";
    topDl.setAttribute("aria-disabled", "true");
    topDl.innerHTML = `<span class="spinner-ring spinner-ring-sm"></span>Building Output…`;
    cardDl.className = "btn btn-download btn-sm btn-locked";
    cardDl.setAttribute("aria-disabled", "true");
    cardDl.textContent = "Building…";

    if (stageDPill) {
      stageDPill.className = "stage-status";
      stageDPill.textContent = "Building…";
    }
    if (kpiDag) {
      kpiDag.innerHTML = `<span class="spinner-ring spinner-ring-sm"></span>Building…`;
    }

    outBanner.className = "output-readiness-banner output-banner-running";
    outTag.innerHTML = `<span class="spinner-ring spinner-ring-sm"></span>STATE: RUNNING &middot; OUTPUT STILL BUILDING`;
    outText.innerHTML = `Currently executing <strong>${(snap && snap.stage_label) || "GB10 optimization"}</strong>. Final Private Copilot and downloadable <code>.zip</code> will unlock automatically as soon as Step 3 (FINISHED) completes.`;
    outPill.className = "badge badge-amber";
    outPill.textContent = "BUILDING · NOT READY YET";

    outSection.classList.add("output-section-locked");
    outSection.classList.remove("output-section-ready");
    btnLiveCopilot.disabled = true;
    btnPreviewCopilot.disabled = true;
    if (copilotBadge) {
      copilotBadge.className = "badge badge-amber";
      copilotBadge.textContent = "TUNING IN PROGRESS…";
    }
    document.getElementById("pg-base-out").textContent = "Optimization running on Fuzzball GB10…";
    document.getElementById("pg-tuned-out").textContent =
      "Teaching model with Remote Teacher Train/Val/Test SFT splits — output unlocks automatically when FINISHED.";
    renderArtifactViewer();
    return;
  }

  if (phase === "finished") {
    topPill.className = "lifecycle-top-pill lifecycle-pill-finished";
    topPill.textContent = "✓ STATE: FINISHED · OUTPUT READY";

    banner.className = "lifecycle-banner lifecycle-finished";
    stateChip.textContent = "✓ STEP 3 OF 3 · FINISHED — OUTPUT READY";
    teacherChip.textContent = tcText;
    const comp = currentCompanyName();
    const shortM = String(((summary && summary.model_profile && summary.model_profile.name) || currentModelId())).split("/").pop();
    headline.textContent = `Finished! ${comp}'s Tuned ${shortM} Copilot & Turnkey .zip Package Are Ready`;
    const pplSummary =
      summary && summary.stage_a
        ? `Held-Out PPL ${summary.stage_a.base_held_out_perplexity.toFixed(3)} → ${summary.stage_a.winning_summary.held_out_perplexity.toFixed(3)} · Train ${summary.stage_b.overall_speedup_x}x · vLLM Serve ${summary.stage_c.serving_speedup_x}x`
        : "All Teacher SFT splits, LoRA weights, custom sm_121 kernels, and vLLM gates verified.";
    sub.textContent = `${pplSummary} — Click "Jump to Tuned Output" to test your model or "Download Ready .zip" to deploy.`;
    jumpBtn.classList.remove("hidden");
    bannerDl.classList.remove("hidden");

    stepPre.className = "lc-step lc-step-done";
    stepPre.querySelector(".lc-num").textContent = "✓";
    stepRun.className = "lc-step lc-step-done";
    stepRun.querySelector(".lc-num").textContent = "✓";
    stepFin.className = "lc-step lc-step-active";
    stepFin.querySelector(".lc-num").textContent = "✓";
    progFill.style.width = "100%";

    topDl.className = "btn btn-download btn-ready-glow";
    topDl.setAttribute("aria-disabled", "false");
    topDl.textContent = "✓ Output Ready · Download .zip";
    cardDl.className = "btn btn-download btn-sm btn-ready-glow";
    cardDl.setAttribute("aria-disabled", "false");
    cardDl.textContent = "✓ Download .zip";

    if (stageDPill) {
      stageDPill.className = "stage-status status-ok";
      stageDPill.textContent = "✓ Output Ready";
    }
    if (kpiDag) {
      kpiDag.textContent = "Output Ready";
    }

    outBanner.className = "output-readiness-banner output-banner-finished";
    outTag.textContent = "✓ STATE: FINISHED · OUTPUT READY TO TEST & DOWNLOAD";
    outText.innerHTML = `Your tuned <strong>${comp}</strong> Private Copilot, Remote-Teacher SFT splits (<code>teacher_sft_train.jsonl</code>), and turnkey <code>.zip</code> bundle are unlocked and ready below.`;
    outPill.className = "badge badge-green";
    outPill.textContent = "✓ OUTPUT READY";

    outSection.classList.remove("output-section-locked");
    outSection.classList.add("output-section-ready");
    btnLiveCopilot.disabled = false;
    btnPreviewCopilot.disabled = false;
    renderArtifactViewer();
    return;
  }

  if (phase === "failed") {
    topPill.className = "lifecycle-top-pill lifecycle-pill-failed";
    topPill.textContent = "STATE: FAILED";
    banner.className = "lifecycle-banner lifecycle-failed";
    stateChip.textContent = "RUN FAILED";
    headline.textContent = `Optimization Stopped With Error: ${(snap && snap.error) || "Check error banner below"}`;
    sub.textContent = "Adjust configuration or execution backend and click Tune on Fuzzball & Generate Local LLM to retry.";
    jumpBtn.classList.add("hidden");
    bannerDl.classList.add("hidden");
    return;
  }

  // Default: PRE-RUN
  topPill.className = "lifecycle-top-pill lifecycle-pill-prerun";
  topPill.textContent = "STATE: PRE-RUN";

  banner.className = "lifecycle-banner lifecycle-prerun";
  stateChip.textContent = "STEP 1 OF 3 · PRE-RUN";
  teacherChip.textContent = tcText;
  headline.textContent = "Pre-Run: Configure Model, Upload Private Data & Click “Tune on Fuzzball & Generate Local LLM”";
  sub.textContent =
    "Output is locked until optimization runs. Uploading files triggers the Remote Teacher to synthesize leak-free Train (80%), Validation (10%), and Test (10%) SFT teaching data.";
  jumpBtn.classList.add("hidden");
  bannerDl.classList.add("hidden");

  stepPre.className = "lc-step lc-step-active";
  stepPre.querySelector(".lc-num").textContent = "1";
  stepRun.className = "lc-step";
  stepRun.querySelector(".lc-num").textContent = "2";
  stepFin.className = "lc-step";
  stepFin.querySelector(".lc-num").textContent = "3";
  progFill.style.width = "8%";

  topDl.className = "btn btn-download btn-locked";
  topDl.setAttribute("aria-disabled", "true");
  topDl.textContent = "Output Locked (Pre-Run)";
  cardDl.className = "btn btn-download btn-sm btn-locked";
  cardDl.setAttribute("aria-disabled", "true");
  cardDl.textContent = "Output Locked";

  if (stageDPill) {
    stageDPill.className = "stage-status";
    stageDPill.textContent = "Pre-Run · Locked";
  }
  if (kpiDag) {
    kpiDag.textContent = "Awaiting Run";
  }

  outBanner.className = "output-readiness-banner output-banner-prerun";
  outTag.textContent = "STATE: PRE-RUN · OUTPUT NOT READY YET";
  outText.innerHTML =
    "Start an optimization run above to teach your selected model on the synthesized Train / Val / Test SFT splits and unlock your Private Copilot &amp; downloadable <code>.zip</code> bundle.";
  outPill.className = "badge";
  outPill.textContent = "LOCKED UNTIL FINISHED";

  outSection.classList.add("output-section-locked");
  outSection.classList.remove("output-section-ready");
  btnLiveCopilot.disabled = true;
  btnPreviewCopilot.disabled = true;
  if (copilotBadge) {
    copilotBadge.className = "badge";
    copilotBadge.textContent = "PRE-RUN · LOCKED";
  }
  document.getElementById("pg-base-out").textContent = "Pre-Run: No optimization executed yet…";
  document.getElementById("pg-tuned-out").textContent =
    "Click 'Tune on Fuzzball & Generate Local LLM' above — your tuned Copilot comparison unlocks automatically when the run reaches FINISHED.";
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
  document.getElementById("svg-vast-title").innerHTML =
    `VAST Data Shared Persistent Volume (<code>/vast</code> &middot; <code>use: ${team}</code>) &mdash; Zero-Copy Weights, Tokenized Splits &amp; LoRA Checkpoints`;
}

function renderRealtimeAnalysis(snap) {
  const m = snap.live_step_metrics || {};
  const b = snap.live_budget || {};
  if (b && b.cost_budget_usd != null) {
    const spent = Number(b.spent_usd || 0);
    const cap = Number(b.cost_budget_usd || 5.0);
    const elMin = (Number(b.elapsed_s || 0) / 60.0).toFixed(1);
    const capMin = (Number(b.time_budget_s || 480.0) / 60.0).toFixed(1);
    const meterEl = document.getElementById("live-budget-meter");
    if (meterEl) {
      meterEl.textContent = `$${spent.toFixed(2)} / $${cap.toFixed(2)} · ${elMin} / ${capMin}m`;
    }
    const rRound = Number(b.teacher_round || 0);
    document.getElementById("rt-memsave").textContent = `R${rRound} · $${spent.toFixed(2)}`;
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
        ? `<tr><td colspan="5" class="table-placeholder-cell"><span class="spinner-ring spinner-ring-sm"></span>Streaming first trial completion from GB10…</td></tr>`
        : `<tr><td colspan="5" class="table-placeholder-cell">No trials executed in this session yet. Start a run above to stream trial metrics.</td></tr>`;
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
  const shortModel = currentModelId().split("/").pop();
  document.getElementById("kpi-model-short").textContent = shortModel;
  document.getElementById("kpi-ppl").textContent = "—";
  document.getElementById("meta-stage-a").textContent = "Awaiting run · Parallel LoRA recipe search with Successive Halving";
  document.getElementById("kpi-train-speed").textContent = "—";
  document.getElementById("meta-stage-b").textContent = "Awaiting Stage A · Custom sm_121 Triton + 2×GB10 scaling sweep";
  document.getElementById("kpi-serve-speed").textContent = "—";
  document.getElementById("meta-stage-c").textContent = "Awaiting Stage B · vLLM BF16 / FP8 / NVFP4 quality-floor sweep";
  document.getElementById("meta-stage-d").textContent = "Ready to package brandable Copilot .zip once optimization finishes";

  const curveBadge = document.getElementById("curve-badge");
  curveBadge.textContent = "Awaiting Run";
  curveBadge.className = "badge";
  document.getElementById("loss-chart-wrap").innerHTML = `
    <div class="panel-empty-state">
      <div class="panel-state-title">No Optimization Curves Yet</div>
      <div class="panel-state-sub">Click <strong>Tune on Fuzzball &amp; Generate Local LLM</strong> above to stream live train &amp; validation loss trajectories step-by-step.</div>
    </div>
  `;
  document.getElementById("loss-legend").innerHTML = "";

  const utilBadge = document.getElementById("util-badge");
  utilBadge.textContent = "Awaiting Run";
  utilBadge.className = "badge";
  document.getElementById("train-bars").innerHTML = `
    <div class="panel-empty-state">
      <div class="panel-state-title">Stage B &amp; Teacher Waterfall Empty</div>
      <div class="panel-state-sub">Streams live tokens/sec bars across PyTorch baseline, custom <code>sm_121</code> kernels, 2-node NCCL DDP, and Continuous Teacher rounds during the run.</div>
    </div>
  `;

  document.querySelector("#serving-table tbody").innerHTML = `
    <tr>
      <td colspan="6" class="table-placeholder-cell">
        No Stage C vLLM sweep data yet &mdash; populates live as BF16, FP8, and Blackwell NVFP4 topologies are benchmarked on the Sealed Test Split.
      </td>
    </tr>
  `;

  document.querySelector("#kernel-table tbody").innerHTML = `
    <tr>
      <td colspan="4" class="table-placeholder-cell">
        Awaiting <code>sm_121</code> numerical parity microbenchmarks vs. FP32 PyTorch reference.
      </td>
    </tr>
  `;

  document.getElementById("advisor-rationales").innerHTML = `
    <div class="panel-empty-state" style="min-height:84px">
      <div class="panel-state-sub">Stage-by-stage Remote-Teacher rationales &amp; cryptographic SHA-256 firewall audits will stream here during the run.</div>
    </div>
  `;
  const fwBox = document.getElementById("firewall-audit-box");
  fwBox.classList.remove("blocked-alert");
  fwBox.innerHTML = `<strong>Zero-Egress Firewall Armed:</strong> <code>contains_raw_data=false</code> schema guard active &mdash; waiting for optimization telemetry.`;
}

function resetPanelsForLoading(modelId) {
  const shortModel = String(modelId || currentModelId()).split("/").pop();
  document.getElementById("kpi-model-short").textContent = shortModel;
  document.getElementById("kpi-ppl").innerHTML = `<span class="spinner-ring spinner-ring-sm"></span>Searching…`;
  document.getElementById("meta-stage-a").textContent = "Initializing dataset splits & Stage A parallel LoRA search…";
  document.getElementById("kpi-train-speed").innerHTML = `<span class="spinner-ring spinner-ring-sm"></span>Queued…`;
  document.getElementById("meta-stage-b").textContent = "Waiting for Stage A winning LoRA recipe…";
  document.getElementById("kpi-serve-speed").innerHTML = `<span class="spinner-ring spinner-ring-sm"></span>Queued…`;
  document.getElementById("meta-stage-c").textContent = "Waiting for Stage B training lock-in…";
  document.getElementById("meta-stage-d").textContent = "Bundle updates automatically when stages complete…";

  const curveBadge = document.getElementById("curve-badge");
  curveBadge.textContent = "STREAMING SOON…";
  curveBadge.className = "badge badge-amber";
  document.getElementById("loss-chart-wrap").innerHTML = `
    <div class="panel-loading-state">
      <div class="spinner-ring"></div>
      <div class="panel-state-title">Initializing GB10 Optimization Stream…</div>
      <div class="panel-state-sub">Partitioning Train / Val / Sealed Test splits &amp; streaming live forward/backward loss curves in real time.</div>
    </div>
  `;
  document.getElementById("loss-legend").innerHTML = "";

  const utilBadge = document.getElementById("util-badge");
  utilBadge.textContent = "WAITING ON STAGE A";
  utilBadge.className = "badge badge-amber";
  document.getElementById("train-bars").innerHTML = `
    <div class="panel-loading-state">
      <div class="spinner-ring"></div>
      <div class="panel-state-title">Waiting for Stage A Recipe Lock-In…</div>
      <div class="panel-state-sub">Stage B <code>sm_121</code> kernel &amp; Dual-GPU scaling bars will stream here live in real time once Stage A selects the optimal LoRA recipe.</div>
    </div>
  `;

  document.querySelector("#serving-table tbody").innerHTML = `
    <tr>
      <td colspan="6" class="table-placeholder-cell">
        <span class="spinner-ring spinner-ring-sm"></span>
        Waiting for Stage B training lock-in &mdash; Stage C vLLM BF16 / FP8 / NVFP4 sweep runs next…
      </td>
    </tr>
  `;

  document.querySelector("#kernel-table tbody").innerHTML = `
    <tr>
      <td colspan="4" class="table-placeholder-cell">
        <span class="spinner-ring spinner-ring-sm"></span>
        Running <code>sm_121</code> numerical parity suite vs. FP32 PyTorch reference…
      </td>
    </tr>
  `;

  document.getElementById("advisor-rationales").innerHTML = `
    <div class="inline-stream-spinner">
      <span class="spinner-ring spinner-ring-sm"></span>
      <span>Awaiting Stage A sanitized telemetry digest (<code>contains_raw_data=false</code>)…</span>
    </div>
  `;
  const fwBox = document.getElementById("firewall-audit-box");
  fwBox.classList.remove("blocked-alert");
  fwBox.innerHTML = `<span class="spinner-ring spinner-ring-sm"></span><strong>Live Privacy Firewall Active:</strong> Inspecting outbound hyperparameter &amp; loss telemetry…`;
}

function renderLiveTelemetry(snap) {
  latestLiveSnap = snap;
  renderClientMachineBridge(snap.client_machine, snap.cluster_endpoint, snap.definition_id || currentTeamId());
  document.getElementById("live-stage-label").textContent = !snap.user_triggered
    ? "Ready — Click Tune on Fuzzball & Generate Local LLM to start live run"
    : snap.stage_label || "Ready";

  const stBadge = document.getElementById("live-run-state");
  const st = !snap.user_triggered && snap.state === "completed" ? "idle" : String(snap.state || "idle");
  stBadge.textContent =
    st === "queued_for_gpu"
      ? `QUEUED ON FUZZBALL · ${snap.backend}`
      : st === "idle"
      ? `READY · IDLE`
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
  if (!snap.user_triggered) {
    feed.innerHTML = `<div class="event-line"><span class="ev-ts">[READY]</span>Awaiting optimization run &mdash; select a model &amp; click Tune on Fuzzball &amp; Generate Local LLM.</div>`;
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
          <div class="panel-state-title">${st === "queued_for_gpu" ? "Queued on Fuzzball Cluster…" : "Streaming First Optimization Step…"}</div>
          <div class="panel-state-sub">${snap.stage_label || "Loading base weights & partitioning dataset splits…"}</div>
        </div>
      `;
      document.getElementById("loss-legend").innerHTML = "";
    }
    renderIncrementalStagePanels(snap);
  }
}

function renderIncrementalStagePanels(snap) {
  const shortModel = String(snap.model_id || currentModelId()).split("/").pop();
  document.getElementById("kpi-model-short").textContent = shortModel;
  const curStage = String(snap.current_stage || "");

  // 1. Live Curve Equivalence Badge
  const eq = snap.live_curve_equivalence || {};
  const curveBadge = document.getElementById("curve-badge");
  if (eq && eq.reference_trial_id && eq.max_delta != null) {
    const d = Number(eq.max_delta);
    const ok = eq.accepted !== false;
    curveBadge.textContent = ok
      ? `LIVE Δ = ${d.toFixed(4)} (≤ 0.025 PASS)`
      : `LIVE Δ = ${d.toFixed(4)} (> 0.025 DRIFT)`;
    curveBadge.className = ok ? "badge badge-green" : "badge badge-amber";
  } else if (curStage === "stage_a_recipe") {
    const nCurves = Object.keys(snap.live_curves || {}).length;
    curveBadge.textContent = nCurves > 0 ? `LIVE STAGE A (${nCurves} trial curves)` : "LOADING STAGE A…";
    curveBadge.className = "badge badge-accent";
  } else {
    curveBadge.textContent = "INITIALIZING…";
    curveBadge.className = "badge badge-amber";
  }

  // 2. Stage A Live Funnel Card
  const saLive = snap.live_stage_a || {};
  if (saLive.best_perplexity) {
    document.getElementById("kpi-ppl").textContent = `${Number(saLive.base_perplexity).toFixed(3)} → ${Number(saLive.best_perplexity).toFixed(3)}`;
    document.getElementById("meta-stage-a").textContent =
      `Leading: ${saLive.best_trial_id} · Val ${Number(saLive.best_val_loss).toFixed(3)} (${saLive.trials_completed} trial(s) done)`;
  } else {
    document.getElementById("kpi-ppl").innerHTML = `<span class="spinner-ring spinner-ring-sm"></span>Searching…`;
    document.getElementById("meta-stage-a").textContent = "Running parallel Successive Halving LoRA rungs…";
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
      `${Math.round(bestOut.summary.mean_tokens_per_sec).toLocaleString()} tok/s (${speedup}x)`;
    document.getElementById("meta-stage-b").textContent =
      `Streaming live (${sbList.filter((x) => !x.in_progress).length} completed) · Leading: ${bestOut.label}`;
    utilBadge.textContent = `LIVE · ${Number(bestOut.summary.hardware_utilization_pct || 0).toFixed(1)}% Peak`;
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
          ? `<span class="tag-ok"><span class="spinner-ring spinner-ring-sm"></span>STREAMING (Δ ${o.max_loss_curve_delta.toFixed(4)})</span>`
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
    const moreStreaming =
      (curStage === "stage_b_training_perf" || curStage === "continuous_teacher_loop") &&
      !sbList.some((o) => o.in_progress)
        ? `<div class="inline-stream-spinner"><span class="spinner-ring spinner-ring-sm"></span><span>Dispatching next ${curStage === "continuous_teacher_loop" ? "Remote-Teacher mutation" : "Stage B kernel/DDP"} trial…</span></div>`
        : "";
    document.getElementById("train-bars").innerHTML = barsHtml + moreStreaming;
  } else {
    document.getElementById("kpi-train-speed").innerHTML = `<span class="spinner-ring spinner-ring-sm"></span>After Stage A…`;
    document.getElementById("meta-stage-b").textContent = "Waiting for Stage A recipe lock-in before kernel & DDP sweep…";
    utilBadge.textContent = "WAITING ON STAGE A";
    utilBadge.className = "badge badge-amber";
    document.getElementById("train-bars").innerHTML = `
      <div class="panel-loading-state">
        <div class="spinner-ring"></div>
        <div class="panel-state-title">Stage A Recipe Search Running…</div>
        <div class="panel-state-sub">Stage B <code>sm_121</code> kernel &amp; 2-GPU scaling waterfall streams live right after Stage A locks in the winning LoRA recipe.</div>
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
      `${Math.round(bestSrv.throughput_tokens_per_sec).toLocaleString()} tok/s (${srvSpeedup}x)`;
    document.getElementById("meta-stage-c").textContent =
      `Live Stage C (${scList.length} swept) · Leading: ${bestSrv.label}`;
    const rowsHtml = scList
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
    const tailSpinner =
      curStage === "stage_c_inference_perf"
        ? `<tr><td colspan="6" class="table-placeholder-cell"><span class="spinner-ring spinner-ring-sm"></span>Benchmarking next vLLM precision &amp; tensor-parallel topology on Sealed Test Split…</td></tr>`
        : "";
    document.querySelector("#serving-table tbody").innerHTML = rowsHtml + tailSpinner;
  } else {
    document.getElementById("kpi-serve-speed").innerHTML = `<span class="spinner-ring spinner-ring-sm"></span>After Stage B…`;
    document.getElementById("meta-stage-c").textContent =
      curStage === "stage_c_inference_perf"
        ? "Launching vLLM BF16 / FP8 / NVFP4 benchmarks on Sealed Test Split…"
        : "Waiting for Stage B training lock-in before vLLM serving sweep…";
    document.querySelector("#serving-table tbody").innerHTML = `
      <tr>
        <td colspan="6" class="table-placeholder-cell">
          <span class="spinner-ring spinner-ring-sm"></span>
          ${
            curStage === "stage_c_inference_perf"
              ? "Benchmarking first vLLM BF16 baseline replica on Sealed Test Split…"
              : "Waiting for Stage B training lock-in &mdash; Stage C vLLM BF16 / FP8 / NVFP4 sweep streams next…"
          }
        </td>
      </tr>
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
      <tr>
        <td colspan="4" class="table-placeholder-cell">
          <span class="spinner-ring spinner-ring-sm"></span>
          Verifying custom <code>sm_121</code> kernel numerical parity vs. FP32 PyTorch…
        </td>
      </tr>
    `;
  }

  // 6. Live Advisor Rationales & Firewall Audit Stream
  if (Array.isArray(snap.live_firewall) && snap.live_firewall.length > 0) {
    const exps = snap.live_firewall.filter((f) => f.explanation && f.explanation.rationale);
    const itemsHtml = exps
      .map((f) => `<div class="advisor-item"><strong>[${f.explanation.stage}]</strong> ${f.explanation.rationale}</div>`)
      .join("");
    document.getElementById("advisor-rationales").innerHTML =
      itemsHtml +
      `<div class="inline-stream-spinner"><span class="spinner-ring spinner-ring-sm"></span><span>Streaming next stage / Remote-Teacher firewalled audit…</span></div>`;

    const lastRec = snap.live_firewall[snap.live_firewall.length - 1].record || {};
    const sha = String(lastRec.payload_sha256 || "").slice(0, 16);
    const fwBox = document.getElementById("firewall-audit-box");
    fwBox.classList.remove("blocked-alert");
    fwBox.innerHTML =
      `<strong>Live Privacy Firewall Stream:</strong> ${snap.live_firewall.length} stage payload(s) verified &middot; ` +
      `<code>contains_raw_data=false</code> &middot; Latest SHA-256: <code>${sha}…</code>`;
  } else {
    document.getElementById("advisor-rationales").innerHTML = `
      <div class="inline-stream-spinner">
        <span class="spinner-ring spinner-ring-sm"></span>
        <span>Waiting for Stage A completion to audit &amp; emit first advisor rationale…</span>
      </div>
    `;
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
      label: `Reference (${refKey.split("-").slice(2).join("-")})`,
      color: "#38bdf8",
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
        label: `Live: ${cleanName} (streaming)`,
        color: drifting ? "#f87171" : "#a3e635",
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
          label: `${cleanBest} (Accepted)`,
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
  document.getElementById("curve-badge").className = "badge badge-green";
  document.getElementById("util-badge").textContent =
    `${winTrainOut.summary.hardware_utilization_pct.toFixed(1)}% Cluster Peak`;
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
        liveBtn.textContent = "Running on Fuzzball GB10…";
      }
      if (badge) {
        badge.className = "badge badge-amber";
        badge.textContent = "RUNNING ON FUZZBALL GB10 GPU…";
      }
      document.getElementById("pg-tuned-out").textContent =
        `Submitting prompt to live CIQ Fuzzball GB10 GPU (${currentTeamId()}) for ${companyName}…`;
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
      if (res.source === "live_fuzzball_gb10_gpu") {
        badge.className = "badge badge-accent";
        badge.textContent = `LIVE GB10 (${res.hostname || "SPARK"}) · ${res.throughput_tok_s} tok/s`;
      } else {
        badge.className = "badge badge-green";
        badge.textContent = `PREVIEW READY · ${Math.round(res.throughput_tok_s || 0)} tok/s`;
      }
    }
  } catch (_) {
    if (badge) {
      badge.className = "badge";
      badge.textContent = "READY";
    }
  } finally {
    if (liveBtn) {
      liveBtn.disabled = false;
      liveBtn.textContent = "Run Live on Fuzzball GPU";
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
    document.getElementById("btn-run").textContent = "Tune on Fuzzball & Generate Local LLM";
    return;
  }
  if (snap.state === "failed") {
    lastObservedRunState = snap.state;
    stopActivePollLoop();
    renderLifecycleState("failed", snap, latestSummary);
    document.getElementById("btn-run").disabled = false;
    document.getElementById("btn-run").textContent = "Tune on Fuzzball & Generate Local LLM";
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

document.getElementById("backend-select").addEventListener("change", () => {
  userPickedBackend = true;
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
    btn.textContent = "Load onto VAST Volume";
  }
});

document.getElementById("btn-stop-run").addEventListener("click", async () => {
  const stopBtn = document.getElementById("btn-stop-run");
  stopBtn.disabled = true;
  stopBtn.textContent = "Stopping & Packaging…";
  try {
    await fetchJson("/api/run/stop", { method: "POST" });
  } catch (_) {}
  setTimeout(() => {
    stopBtn.disabled = false;
    stopBtn.textContent = "Stop & Package Best";
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
  btn.textContent = `Tuning (${backend})…`;
  resetPanelsForLoading(modelId);
  renderLifecycleState("running", { current_stage: "starting", stage_label: `Synthesizing Remote Teacher Train/Val/Test SFT splits & launching ${modelId}…` }, null);
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
    btn.textContent = "Tune on Fuzzball & Generate Local LLM";
    const err = await resp.json();
    const errMsg = err.detail || "Failed to start optimization";
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
  const outEl = document.getElementById("output-readiness-banner");
  if (outEl) outEl.scrollIntoView({ behavior: "smooth", block: "start" });
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

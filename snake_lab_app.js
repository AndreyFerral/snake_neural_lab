class SnakeLabApp {
  constructor() {
    this.gameCanvas = document.getElementById("gameCanvas");
    this.networkCanvas = document.getElementById("networkCanvas");
    this.chartCanvas = document.getElementById("chartCanvas");
    this.deathSnapshotCanvas = document.getElementById("deathSnapshotCanvas");

    this.gameCtx = this.getCanvasContext(this.gameCanvas);
    this.networkCtx = this.getCanvasContext(this.networkCanvas);
    this.chartCtx = this.getCanvasContext(this.chartCanvas);
    this.deathSnapshotCtx = this.getCanvasContext(this.deathSnapshotCanvas);

    this.agentModeSelect = document.getElementById("agentMode");
    this.smartModelControl = document.getElementById("smartModelControl");
    this.smartModelSelect = document.getElementById("smartModel");
    this.boardSizeSelect = document.getElementById("boardSize");
    this.speedRange = document.getElementById("speedRange");
    this.runToggle = document.getElementById("runToggle");
    this.aiModeButton = document.getElementById("aiMode");
    this.trainToggle = document.getElementById("trainToggle");
    this.evalGreedyButton = document.getElementById("evalGreedy");
    this.evalResultBar = document.getElementById("evalResultBar");
    this.resetAgentButton = document.getElementById("resetAgent");
    this.restartBackendButton = document.getElementById("restartBackend");
    this.shutdownBackendButton = document.getElementById("shutdownBackend");
    this.recordReplaysCheckbox = document.getElementById("recordReplays");
    this.ppoMultiboardControl = document.getElementById("ppoMultiboardControl");
    this.ppoMultiboardCheckbox = document.getElementById("ppoMultiboard");

    this.networkPanel = document.getElementById("networkPanel");
    this.networkGraphCard = document.getElementById("networkGraphCard");

    this.deviceBadge = document.getElementById("deviceBadge");
    this.boardSizeBadge = document.getElementById("boardSizeBadge");
    this.inputGridBadge = document.getElementById("inputGridBadge");
    this.networkSourceBadge = document.getElementById("networkSourceBadge");

    this.agentPillBadge = document.getElementById("agentPillBadge");
    this.modeStat = document.getElementById("modeStat");
    this.epochStat = document.getElementById("epochStat");
    this.scoreStat = document.getElementById("scoreStat");
    this.gamePanelKicker = document.getElementById("gamePanelKicker");
    this.bestScoreStat = document.getElementById("bestScoreStat");
    this.avgScoreStat = document.getElementById("avgScoreStat");
    this.epsilonStat = document.getElementById("epsilonStat");
    this.updatesLabel = document.getElementById("updatesLabel");
    this.leaderStat = document.getElementById("leaderStat");
    this.auxMetricLabel = document.getElementById("auxMetricLabel");
    this.auxMetricStat = document.getElementById("auxMetricStat");
    this.decisionMeta = document.getElementById("decisionMeta");
    this.chartMetricLabel = document.getElementById("chartMetricLabel");
    this.chartMetricDescription = document.getElementById("chartMetricDescription");
    this.rewardRulesBody = document.getElementById("rewardRulesBody");

    this.decisionText = document.getElementById("decisionText");
    this.decisionStatsTable = document.getElementById("decisionStatsTable");
    this.deathSnapshotMeta = document.getElementById("deathSnapshotMeta");
    this.deathChoiceMeta = document.getElementById("deathChoiceMeta");
    this.deathStepTimeline = document.getElementById("deathStepTimeline");
    this.deathRiskTable = document.getElementById("deathRiskTable");

    this.state = null;
    this.stateDetailsFresh = false;
    this.stableHeroProfile = null;
    this.showNetwork = true;
    this.isFetching = false;
    this.isFetchingFrame = false;
    this.speedDebounce = null;
    this.pollHandle = null;
    this.framePollHandle = null;
    this.frameEventSource = null;
    this.sseConnected = false;
    this.sseReconnectTimer = null;
    this.gameLoopHandle = null;
    this.prevGameSnapshot = null;
    this.currGameSnapshot = null;
    this.gameTransitionStart = 0;
    this.gameStepDurationMs = 125;
    this.lastGameFrameArrivalAt = 0;
    this.lastSettledGameSignature = "";
    this.currentPollMs = API_POLL_MS;
    this.restartPendingUntil = 0;
    this.selectedDeathStepIndex = -1;
    this.lastDeathSnapshotKey = "";
    this.lastDeathTimelineMarkup = "";
    this.lastNetworkRenderAt = -Infinity;
    this.lastChartRenderAt = -Infinity;
    this.lastNetworkPayloadAt = 0;
    this.lastDetailPayloadAt = 0;
    this.lastPanelRenderAt = -Infinity;
    this.lastSnapshotRenderAt = -Infinity;
    this.lastRenderedSnapshotKey = "";
    this.lastGameRenderSignature = "";
    this.lastNetworkRenderSignature = "";
    this.networkHoverState = null;
    this.networkHitCache = { signals: [], hidden1: [], hidden2: [], outputs: [] };
    this.lastChartRenderSignature = "";
    this.lastSnapshotRenderSignature = "";
    this.lastGameRenderTime = 0;
    this.lastRewardRulesMarkup = "";
    this.lastDecisionStatsMarkup = "";
    this.agentModeFocused = false;
    this.smartModelFocused = false;
    this.smartContextActive = false;
    this.lastSmartModelMarkup = "";
    this.renderFrameHandle = null;
    this.renderFrameScheduledAt = 0;
    this.liveFrameRenderHandle = null;
    this.liveFrameScheduledAt = 0;
    this.heavyFrameHandle = null;
    this.pendingHeavyTasks = [];
    this.pendingHeavyTaskNames = new Set();
    this.pendingControlSync = false;
    this.resetConfirmArmed = false;
    this.resetConfirmTimer = null;

    // Внедрение офскрин-рендеринга для кэширования сетки и градиента
    this.gameBgCanvas = document.createElement("canvas");
    this.gameBgCtx = this.gameBgCanvas.getContext("2d");
    this.boardLayout = { boardX: 0, boardY: 0, boardSize: 0, cell: 0 };
    this.bgDrawn = false;

    this.canvasDimensions = new WeakMap();
    this.initResizeObserver();

    this.setHidden(this.networkPanel, false);
    this.setDecisionText("Подключение к Python backend...");
    this.bindEvents();
    this.refreshState();
    this.resetPolling(this.currentPollMs);
    this.connectFrameStream();
    this.startGameRenderLoop();
  }

  getCanvasContext(canvas) {
    return canvas ? canvas.getContext("2d") : null;
  }

  initResizeObserver() {
    this.resizeObserver = new ResizeObserver((entries) => {
      let changed = false;
      for (const entry of entries) {
        const canvas = entry.target;
        const width = Math.max(1, Math.round(entry.contentRect.width || canvas.clientWidth || 1));
        const height = Math.max(1, Math.round(entry.contentRect.height || canvas.clientHeight || 1));
        
        const existing = this.canvasDimensions.get(canvas);
        if (!existing || existing.width !== width || existing.height !== height) {
          this.canvasDimensions.set(canvas, { width, height, dirty: true });
          changed = true;
        }
      }
      if (changed) {
        this.forceNextVisualRefresh();
        this.scheduleRender();
      }
    });

    [this.gameCanvas, this.networkCanvas, this.chartCanvas, this.deathSnapshotCanvas].forEach((canvas) => {
      if (canvas) {
        this.resizeObserver.observe(canvas);
        this.canvasDimensions.set(canvas, {
          width: canvas.width || 300,
          height: canvas.height || 150,
          dirty: true
        });
      }
    });
  }

  setText(element, value) {
    if (!element) return;
    const nextValue = String(value ?? "");
    if (element.textContent !== nextValue) element.textContent = nextValue;
  }

  setHtml(element, value) {
    if (!element) return;
    const nextValue = String(value ?? "");
    if (element.innerHTML !== nextValue) element.innerHTML = nextValue;
  }

  resetPolling(intervalMs) {
    const nextInterval = Math.max(40, Number(intervalMs) || API_POLL_MS);
    if (this.pollHandle && this.currentPollMs === nextInterval) return;
    if (this.pollHandle) window.clearInterval(this.pollHandle);
    this.currentPollMs = nextInterval;
    this.pollHandle = window.setInterval(() => this.refreshState(), nextInterval);
  }

  nextFramePollDelay() {
    if (document.hidden) return FRAME_HIDDEN_POLL_MS;
    if (!this.state || this.state.running) return 32;
    return FRAME_IDLE_POLL_MS;
  }

  stopFramePolling() {
    if (this.framePollHandle) {
      window.clearTimeout(this.framePollHandle);
      this.framePollHandle = null;
    }
  }

  resetFramePolling() {
    if (this.sseConnected || this.framePollHandle) return;
    const pollFrame = async () => {
      this.framePollHandle = null;
      if (this.sseConnected) return;
      if (!document.hidden) {
        await this.refreshFrame();
      }
      if (!this.sseConnected && !this.framePollHandle) {
        this.framePollHandle = window.setTimeout(pollFrame, this.nextFramePollDelay());
      }
    };
    this.framePollHandle = window.setTimeout(pollFrame, 0);
  }

  disconnectFrameStream() {
    if (this.sseReconnectTimer) {
      window.clearTimeout(this.sseReconnectTimer);
      this.sseReconnectTimer = null;
    }
    if (this.frameEventSource) {
      try { this.frameEventSource.close(); } catch (_) {}
      this.frameEventSource = null;
    }
    this.sseConnected = false;
  }

  connectFrameStream() {
    if (typeof window.EventSource !== "function") {
      this.resetFramePolling();
      return;
    }
    if (this.frameEventSource) return;
    try {
      const source = new EventSource("/api/frame_stream");
      this.frameEventSource = source;
      source.onopen = () => {
        this.sseConnected = true;
        this.stopFramePolling();
      };
      source.onmessage = (event) => {
        if (!event.data || document.hidden) return;
        try {
          const frameState = JSON.parse(event.data);
          this.applyIncomingFrameState(frameState);
        } catch (err) {
          console.warn("Snake SSE parse error:", err);
        }
      };
      source.onerror = () => {
        this.sseConnected = false;
        this.resetFramePolling();
        if (source.readyState === EventSource.CLOSED) {
          this.disconnectFrameStream();
          if (!this.sseReconnectTimer) {
            this.sseReconnectTimer = window.setTimeout(() => {
              this.sseReconnectTimer = null;
              this.connectFrameStream();
            }, 1200);
          }
        }
      };
    } catch (_) {
      this.sseConnected = false;
      this.resetFramePolling();
    }
  }

  applyIncomingFrameState(frameState) {
    if (!frameState || typeof frameState !== "object") return;
    const previousFrameId = Number(this.state?.frame_id ?? -1);
    const nextFrameId = Number(frameState?.frame_id ?? previousFrameId);
    if (Number.isFinite(previousFrameId) && Number.isFinite(nextFrameId) && nextFrameId <= previousFrameId && this.state) {
      return;
    }
    this.state = this.state ? this.mergeFrameState(frameState, this.state) : frameState;
    applyRuntimeReasonConfig(this.state);
    this.stateDetailsFresh = false;
    this.ingestGameFrame(this.state);
    this.updateGameHeaderMeta();
  }

  updateGameHeaderMeta() {
    if (!this.state) return;
    const score = Number(this.currGameSnapshot?.score ?? this.state?.game?.score ?? 0);
    this.setText(this.scoreStat, String(score));
    const isPpoTraining = Boolean(this.state.training && this.state.agent_kind === "ppo" && this.state.mode !== "smart");
    const ppoEnvs = Number(this.state.stats?.population ?? 0);
    const mbSummary = String(this.state.ppo_multiboard_summary ?? "");
    const kickerText = isPpoTraining && ppoEnvs > 0
      ? (this.state.ppo_multiboard && mbSummary ? `Поле • Мульти: ${mbSummary}` : `Поле • ${ppoEnvs} сред в фоне`)
      : "Поле";
    this.setText(this.gamePanelKicker, kickerText);
    this.renderEvalResultBar();
  }

  ingestGameFrame(state) {
    const nextGame = state?.game;
    if (!nextGame || !Array.isArray(nextGame.snake)) return;
    const now = performance.now();
    const visualSpeed = Math.max(1, Number(state?.visual_speed ?? state?.speed ?? 8));
    const nominalMs = clamp(1000 / visualSpeed, 16, 1000);
    const prev = this.currGameSnapshot;

    const headChanged = !prev
      || prev.size !== nextGame.size
      || prev.step_count !== nextGame.step_count
      || prev.score !== nextGame.score
      || prev.snake?.length !== nextGame.snake.length
      || prev.snake?.[0]?.x !== nextGame.snake?.[0]?.x
      || prev.snake?.[0]?.y !== nextGame.snake?.[0]?.y;

    if (headChanged) {
      this.prevGameSnapshot = prev;
      this.currGameSnapshot = nextGame;
      if (this.lastGameFrameArrivalAt > 0 && state?.running) {
        const dt = now - this.lastGameFrameArrivalAt;
        if (dt >= nominalMs * 0.5 && dt <= nominalMs * 2.2) {
          this.gameStepDurationMs = clamp(this.gameStepDurationMs * 0.6 + dt * 0.4, 18, 1000);
        } else {
          this.gameStepDurationMs = nominalMs;
        }
      } else {
        this.gameStepDurationMs = nominalMs;
      }
      this.gameTransitionStart = now;
      this.lastGameFrameArrivalAt = now;
      this.lastSettledGameSignature = "";
    }
  }

  canInterpolateSnake(prevGame, currGame) {
    if (!prevGame || !currGame) return false;
    if (prevGame.size !== currGame.size) return false;
    if (Number(currGame.step_count) !== Number(prevGame.step_count) + 1) return false;
    const pSnake = prevGame.snake;
    const cSnake = currGame.snake;
    if (!Array.isArray(pSnake) || !Array.isArray(cSnake) || pSnake.length < 2 || cSnake.length < 2) return false;
    if (cSnake.length !== pSnake.length && cSnake.length !== pSnake.length + 1) return false;
    if (cSnake[1].x !== pSnake[0].x || cSnake[1].y !== pSnake[0].y) return false;
    const manhattan = Math.abs(cSnake[0].x - pSnake[0].x) + Math.abs(cSnake[0].y - pSnake[0].y);
    return manhattan === 1;
  }

  buildInterpolatedSnake(prevGame, currGame, t) {
    const cSnake = currGame.snake;
    if (t >= 0.999 || !this.canInterpolateSnake(prevGame, currGame)) {
      return { snake: cSnake, interpolated: false };
    }
    const pSnake = prevGame.snake;
    const head = {
      x: pSnake[0].x + (cSnake[0].x - pSnake[0].x) * t,
      y: pSnake[0].y + (cSnake[0].y - pSnake[0].y) * t,
    };
    const grew = cSnake.length > pSnake.length;
    const result = [head];
    for (let i = 1; i < cSnake.length; i += 1) {
      result.push(cSnake[i]);
    }
    if (!grew) {
      const pTail = pSnake[pSnake.length - 1];
      const cTail = cSnake[cSnake.length - 1];
      result.push({
        x: pTail.x + (cTail.x - pTail.x) * t,
        y: pTail.y + (cTail.y - pTail.y) * t,
      });
    }
    return { snake: result, interpolated: true };
  }

  startGameRenderLoop() {
    if (this.gameLoopHandle) return;
    const tick = (now) => {
      this.gameLoopHandle = window.requestAnimationFrame(tick);
      if (document.hidden || !this.state) return;
      this.renderGame(now);
    };
    this.gameLoopHandle = window.requestAnimationFrame(tick);
  }

  setHidden(element, hidden) {
    if (element && element.hidden !== hidden) element.hidden = hidden;
  }

  toggleClass(element, className, enabled) {
    if (element) element.classList.toggle(className, enabled);
  }

  setDecisionText(text) {
    this.setText(this.decisionText, text);
  }

  outputLabels() {
    return this.state?.output_labels ?? DEFAULT_OUTPUT_LABELS;
  }

  outputDisplayOrder(labels = this.outputLabels()) {
    const preferred = OUTPUT_DISPLAY_ORDER.filter((index) => index < labels.length);
    const remaining = labels.map((_, index) => index).filter((index) => !preferred.includes(index));
    return [...preferred, ...remaining];
  }

  formatOrderedQValues(values = this.state?.q_values ?? []) {
    return this.outputDisplayOrder()
      .map((index) => values[index])
      .filter((value) => Number.isFinite(value))
      .map((value) => value.toFixed(2))
      .join(" / ");
  }

  arrowForAction(actionIndex) {
    if (actionIndex === 0) return "↑";
    if (actionIndex === 1) return "←";
    if (actionIndex === 2) return "→";
    return "-";
  }

  shouldUseNetworkVisuals() {
    return Boolean(this.showNetwork && !document.hidden);
  }

  resizeCanvas(canvas, ctx) {
    if (!canvas || !ctx) return { width: 0, height: 0, resized: false };
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    
    let dim = this.canvasDimensions.get(canvas);
    if (!dim) {
      dim = { width: canvas.width || 300, height: canvas.height || 150, dirty: true };
      this.canvasDimensions.set(canvas, dim);
    }

    const targetWidth = Math.round(dim.width * dpr);
    const targetHeight = Math.round(dim.height * dpr);
    const resized = dim.dirty || canvas.width !== targetWidth || canvas.height !== targetHeight;

    if (resized) {
      canvas.width = targetWidth;
      canvas.height = targetHeight;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      dim.dirty = false;
    }
    return { width: dim.width, height: dim.height, resized };
  }

  bindEvents() {
    window.addEventListener("keydown", (event) => this.handleKeydown(event));
    window.addEventListener("beforeunload", () => this.disconnectFrameStream());
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) return;
      this.forceNextVisualRefresh();
      if (!this.sseConnected) this.connectFrameStream();
      this.refreshState({ forceDetails: true, forceNetwork: this.shouldUseNetworkVisuals() });
      this.safeRenderAll();
    });
    this.agentModeSelect?.addEventListener("change", () => {
      const selectedAgent = this.agentModeSelect.value;
      this.agentModeFocused = false;
      this.agentModeSelect.blur();
      if (selectedAgent === "smart") { this.smartContextActive = true; this.sendCommand("set_mode", "smart"); }
      else { this.smartContextActive = false; this.sendCommand("set_agent_kind", selectedAgent); }
    });
    this.agentModeSelect?.addEventListener("focus", () => { this.agentModeFocused = true; });
    this.agentModeSelect?.addEventListener("blur", () => { this.agentModeFocused = false; });
    this.agentModeSelect?.addEventListener("pointerdown", () => { this.agentModeFocused = true; });
    this.smartModelSelect?.addEventListener("change", () => { const selectedModel = this.smartModelSelect.value; if (selectedModel) this.sendCommand("set_smart_model", selectedModel); });
    this.smartModelSelect?.addEventListener("focus", () => { this.smartModelFocused = true; });
    this.smartModelSelect?.addEventListener("blur", () => { this.smartModelFocused = false; });
    this.smartModelSelect?.addEventListener("pointerdown", () => { this.smartModelFocused = true; });
    this.boardSizeSelect?.addEventListener("change", () => { this.sendCommand("set_board_size", Number(this.boardSizeSelect.value)); });
    this.speedRange?.addEventListener("input", () => { const speed = Number(this.speedRange.value); window.clearTimeout(this.speedDebounce); this.speedDebounce = window.setTimeout(() => this.sendCommand("set_speed", speed), 40); });
    this.runToggle?.addEventListener("click", () => this.sendCommand("toggle_running"));
    this.aiModeButton?.addEventListener("click", () => { this.sendCommand("set_mode", this.smartContextActive ? "smart" : "ai"); });
    
    this.trainToggle?.addEventListener("click", () => { 
      if (this.trainToggle.disabled) return; 
      this.sendCommand("toggle_training", !this.state?.training); 
    });

    this.evalGreedyButton?.addEventListener("click", () => {
      if (this.evalGreedyButton.disabled) return;
      this.sendCommand("eval_greedy");
    });

    this.resetAgentButton?.addEventListener("click", () => {
      if (this.resetAgentButton.disabled) return;
      if (!this.resetConfirmArmed) {
        this.resetConfirmArmed = true;
        this.setText(this.resetAgentButton, "Точно?");
        this.toggleClass(this.resetAgentButton, "is-confirm", true);
        if (this.resetConfirmTimer) window.clearTimeout(this.resetConfirmTimer);
        this.resetConfirmTimer = window.setTimeout(() => {
          this.resetConfirmArmed = false;
          this.resetConfirmTimer = null;
          this.setText(this.resetAgentButton, "↺ Сброс");
          this.toggleClass(this.resetAgentButton, "is-confirm", false);
        }, 2600);
        return;
      }
      if (this.resetConfirmTimer) { window.clearTimeout(this.resetConfirmTimer); this.resetConfirmTimer = null; }
      this.resetConfirmArmed = false;
      this.setText(this.resetAgentButton, "↺ Сброс");
      this.toggleClass(this.resetAgentButton, "is-confirm", false);
      this.sendCommand("reset_agent");
    });
    this.restartBackendButton?.addEventListener("click", () => this.restartBackend());
    this.shutdownBackendButton?.addEventListener("click", () => this.shutdownBackend());
    this.recordReplaysCheckbox?.addEventListener("change", () => {
      this.sendCommand("set_record_replays", this.recordReplaysCheckbox.checked);
    });
    this.ppoMultiboardCheckbox?.addEventListener("change", () => {
      this.sendCommand("set_ppo_multiboard", this.ppoMultiboardCheckbox.checked);
    });
    this.deathStepTimeline?.addEventListener("click", (event) => {
      const button = event.target instanceof Element ? event.target.closest("button[data-death-step]") : null;
      if (!button) return;
      const nextIndex = Number(button.getAttribute("data-death-step"));
      if (Number.isFinite(nextIndex)) { this.selectedDeathStepIndex = nextIndex; this.renderLastDeathSnapshot(); }
    });
    this.networkCanvas?.addEventListener("mousemove", (event) => this.handleNetworkPointerMove(event));
    this.networkCanvas?.addEventListener("mouseleave", () => this.clearNetworkHover());
  }

  getNetworkHoverKey(hoverState) {
    if (!hoverState) return "";
    return `${hoverState.layer}:${hoverState.index}`;
  }

  hitTestNetworkGraph(x, y) {
    const cache = this.networkHitCache;
    if (!cache) return null;

    for (const circleList of [cache.hidden1, cache.hidden2]) {
      for (const item of circleList || []) {
        const dx = x - item.x;
        const dy = y - item.y;
        const hitRadius = item.r + 4;
        if (dx * dx + dy * dy <= hitRadius * hitRadius) {
          return { ...item, anchorX: x, anchorY: y };
        }
      }
    }

    for (const rectList of [cache.signals, cache.outputs]) {
      for (const item of rectList || []) {
        if (x >= item.x && x <= item.x + item.w && y >= item.y && y <= item.y + item.h) {
          return { ...item, anchorX: x, anchorY: y };
        }
      }
    }

    return null;
  }

  handleNetworkPointerMove(event) {
    if (!this.networkCanvas) return;
    const rect = this.networkCanvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;

    const x = event.clientX - rect.left;
    const y = event.clientY - rect.top;
    const nextHover = this.hitTestNetworkGraph(x, y);
    const prevKey = this.getNetworkHoverKey(this.networkHoverState);
    const nextKey = this.getNetworkHoverKey(nextHover);

    this.networkCanvas.style.cursor = nextHover ? "pointer" : "default";
    if (!nextHover && !this.networkHoverState) return;

    this.networkHoverState = nextHover;
    if (prevKey !== nextKey || nextHover) {
      this.renderNetworkGraph();
    }
  }

  clearNetworkHover() {
    if (!this.networkCanvas) return;
    this.networkCanvas.style.cursor = "default";
    if (!this.networkHoverState) return;
    this.networkHoverState = null;
    this.renderNetworkGraph();
  }

  async sendCommand(type, value = undefined) {
    try {
      const payload = { type };
      if (value !== undefined) payload.value = value;
      const response = await apiJson("/api/command", { method: "POST", body: JSON.stringify(payload), timeoutMs: type === "set_smart_model" ? 10000 : 2500 });
      if (response.busy && type === "toggle_training" && this.state) {
        this.state.training = Boolean(value);
        this.state.mode = "ai";
        if (this.state.training) this.state.running = true;
        this.safeRenderAll();
        return;
      }
      const forcePanelRefresh = this.shouldForcePanelRefresh(type);
      if (response.state) {
        this.state = this.isFrameOnlyState(response.state) && this.state ? this.mergeFrameState(response.state, this.state) : this.mergeDeferredState(response.state, this.state);
        applyRuntimeReasonConfig(this.state);
        this.ingestGameFrame(this.state);
        this.pendingControlSync = true;
        this.stateDetailsFresh = false;
        if (forcePanelRefresh) this.forceNextVisualRefresh();
        this.safeRenderAll();
        if (forcePanelRefresh) window.setTimeout(() => this.refreshState({ forceDetails: true, forceNetwork: this.shouldUseNetworkVisuals() }), 80);
        else if (type === "set_smart_model") window.setTimeout(() => this.refreshState({ forceDetails: true }), 60);
        else if (type === "set_mode" && value === "smart") window.setTimeout(() => this.refreshState({ forceDetails: true }), 60);
      } else {
        await this.refreshState({ forceDetails: forcePanelRefresh });
      }
    } catch (error) {
      console.error("Snake command error:", error);
      this.setDecisionText(`Ошибка команды: ${String(error.message || error)}`);
    }
  }

  shouldForcePanelRefresh(type) { return FORCE_PANEL_REFRESH_COMMANDS.has(type); }
  isFrameOnlyState(state) { return Boolean(state && !("available_devices" in state) && !("smart_model" in state)); }

  forceNextVisualRefresh() {
    this.lastPanelRenderAt = -Infinity; this.lastSnapshotRenderAt = -Infinity; this.lastChartRenderAt = -Infinity; this.lastNetworkRenderAt = -Infinity;
    this.lastNetworkPayloadAt = 0; this.lastDetailPayloadAt = 0; this.lastRenderedSnapshotKey = ""; this.lastDeathTimelineMarkup = "";
    this.lastGameRenderSignature = ""; this.lastSettledGameSignature = ""; this.lastNetworkRenderSignature = ""; this.lastChartRenderSignature = ""; this.lastSnapshotRenderSignature = "";
    this.lastGameRenderTime = 0; this.lastRewardRulesMarkup = ""; this.lastDecisionStatsMarkup = ""; this.lastSmartModelMarkup = "";
    this.pendingHeavyTasks = []; this.pendingHeavyTaskNames.clear();
    this.bgDrawn = false;
    
    [this.gameCanvas, this.networkCanvas, this.chartCanvas, this.deathSnapshotCanvas].forEach((canvas) => {
      if (canvas) {
        const dim = this.canvasDimensions.get(canvas);
        if (dim) dim.dirty = true;
      }
    });
  }

  mergeDeferredState(nextState, previousState) {
    if (!previousState) return nextState;
    const carryDeferredFields = () => {
      DEFERRED_STATE_FIELDS.forEach((key) => { if (!(key in nextState) && key in previousState) nextState[key] = previousState[key]; });
      if (previousState.network && !nextState.network) nextState.network = previousState.network;
    };
    const stateShapeChanged = previousState.agent_kind !== nextState.agent_kind || previousState.mode !== nextState.mode || previousState.game?.size !== nextState.game?.size;
    if (stateShapeChanged) { carryDeferredFields(); return nextState; }
    const previousFrameId = Number(previousState.frame_id ?? -1);
    const nextFrameId = Number(nextState.frame_id ?? previousFrameId);
    const staleFrame = Number.isFinite(previousFrameId) && Number.isFinite(nextFrameId) && nextFrameId < previousFrameId;
    if (nextState.backend_busy || staleFrame) STALE_FRAME_FIELDS.forEach((key) => { if (key in previousState) nextState[key] = previousState[key]; });
    carryDeferredFields();
    if (nextState.network && previousState.network) {
      const nextConnections = nextState.network.connections ?? {};
      const previousConnections = previousState.network.connections ?? {};
      const shouldReuseConnections = NETWORK_CONNECTION_KEYS.every((key) => Array.isArray(nextConnections[key]) && nextConnections[key].length === 0) && NETWORK_CONNECTION_KEYS.some((key) => Array.isArray(previousConnections[key]) && previousConnections[key].length > 0);
      if (shouldReuseConnections) nextState.network.connections = previousConnections;
    }
    return nextState;
  }

  mergeFrameState(frameState, previousState) {
    if (!previousState) return frameState;
    const previousFrameId = Number(previousState.frame_id ?? -1);
    const nextFrameId = Number(frameState.frame_id ?? previousFrameId);
    if (Number.isFinite(previousFrameId) && Number.isFinite(nextFrameId) && nextFrameId < previousFrameId) return previousState;
    const merged = { ...previousState, ...frameState };
    FRAME_PRESERVED_FIELDS.forEach((key) => { if (key in previousState) merged[key] = previousState[key]; });
    return merged;
  }

  async restartBackend() {
    this.restartPendingUntil = Date.now() + 12000;
    this.setDecisionText(this.buildDecisionText());
    try { await apiJson("/api/command", { method: "POST", body: JSON.stringify({ type: "restart_server" }), timeoutMs: 2500 }); } catch (error) { console.warn("Backend restart command interrupted:", error); }
    [1600, 2800, 4200, 6500].forEach((delay) => { window.setTimeout(() => this.refreshState(), delay); });
  }

  async shutdownBackend() {
    this.setDecisionText("Выключаю backend и закрываю консоль. Страница больше не будет обновляться.");
    try { await apiJson("/api/command", { method: "POST", body: JSON.stringify({ type: "shutdown_server", close_console: true }), timeoutMs: 1800 }); } catch (error) { console.warn("Backend shutdown command interrupted:", error); }
    if (this.pollHandle) { window.clearInterval(this.pollHandle); this.pollHandle = null; }
    this.stopFramePolling();
    this.disconnectFrameStream();
    if (this.gameLoopHandle) { window.cancelAnimationFrame(this.gameLoopHandle); this.gameLoopHandle = null; }
    if (this.renderFrameHandle) { window.cancelAnimationFrame(this.renderFrameHandle); this.renderFrameHandle = null; }
    if (this.liveFrameRenderHandle) { window.cancelAnimationFrame(this.liveFrameRenderHandle); this.liveFrameRenderHandle = null; }
    if (this.heavyFrameHandle) { window.cancelAnimationFrame(this.heavyFrameHandle); this.heavyFrameHandle = null; }
    this.pendingHeavyTasks = []; this.pendingHeavyTaskNames.clear(); this.isFetching = false; this.isFetchingFrame = false;
    this.setDecisionText("Backend выключен. Консоль закрывается, вкладку можно закрыть.");
    window.setTimeout(() => { try { window.close(); } catch (error) { console.warn("Window close blocked:", error); } }, 250);
  }

  async refreshState(options = {}) {
    if (this.isFetching) return;
    this.isFetching = true;
    try {
      const previousState = this.state;
      const previousNetwork = previousState?.network ?? null;
      const now = performance.now();
      const liveTraining = Boolean(previousState?.training && previousState?.running);
      const networkPayloadMs = liveTraining ? LIVE_NETWORK_PAYLOAD_MS : NETWORK_PAYLOAD_MS;
      const detailPayloadMs = liveTraining ? LIVE_DETAIL_PAYLOAD_MS : DETAIL_PAYLOAD_MS;
      const shouldFetchNetwork = Boolean(options.forceNetwork) || (this.shouldUseNetworkVisuals() && (!previousNetwork || !this.state?.running || now - this.lastNetworkPayloadAt >= networkPayloadMs));
      const shouldFetchDetails = Boolean(options.forceDetails) || !previousState || !this.state?.running || now - this.lastDetailPayloadAt >= detailPayloadMs;
      const nextState = await apiJson(`/api/state?network=${shouldFetchNetwork ? "1" : "0"}&details=${shouldFetchDetails ? "1" : "0"}`, { timeoutMs: 3500 });
      if (!nextState.network && previousNetwork) nextState.network = previousNetwork;
      else if (nextState.network) this.lastNetworkPayloadAt = now;
      if (shouldFetchDetails) this.lastDetailPayloadAt = now;
      this.state = this.mergeDeferredState(nextState, previousState);
      applyRuntimeReasonConfig(this.state);
      this.ingestGameFrame(this.state);
      this.pendingControlSync = true;
      this.stateDetailsFresh = Boolean(shouldFetchDetails && nextState.global_signals);
      if (this.stateDetailsFresh) this.captureStableHeroProfile();
      if (!this.state.restart_pending) this.restartPendingUntil = 0;
    } catch (error) {
      console.error("Snake state error:", error);
      if (Date.now() < this.restartPendingUntil) this.setDecisionText("Жду, пока Python backend поднимется заново...");
      else this.setDecisionText(`Нет связи с Python backend: ${String(error.message || error)}`);
      this.isFetching = false; return;
    }
    try { this.safeRenderAll(); } catch (error) { console.error("Snake render error:", error); this.setDecisionText(`Ошибка интерфейса: ${String(error.message || error)}`); } finally { this.isFetching = false; }
  }

  async refreshFrame() {
    if (this.isFetchingFrame || this.sseConnected || !this.state || document.hidden) return;
    this.isFetchingFrame = true;
    try {
      const frameState = await apiJson("/api/frame", { timeoutMs: 1200 });
      this.applyIncomingFrameState(frameState);
    } catch (error) {
      if (Date.now() >= this.restartPendingUntil) console.warn("Snake frame error:", error);
    } finally {
      this.isFetchingFrame = false;
    }
  }

  handleKeydown(event) {
    if (event.code === "Space") { event.preventDefault(); this.sendCommand("toggle_running"); }
  }

  scheduleRender() {
    if (this.renderFrameHandle) {
      if (performance.now() - this.renderFrameScheduledAt < 200) return;
      window.cancelAnimationFrame(this.renderFrameHandle);
      this.renderFrameHandle = null;
    }
    this.renderFrameScheduledAt = performance.now();
    this.renderFrameHandle = window.requestAnimationFrame(() => {
      this.renderFrameHandle = null;
      this.renderFrameScheduledAt = 0;
      try { this.renderAll(); } catch (error) { console.error("Snake render error:", error); this.setDecisionText(`Ошибка интерфейса: ${String(error.message || error)}`); }
    });
  }



  queueHeavyRender(name, callback) {
    if (this.pendingHeavyTaskNames.has(name)) return;
    this.pendingHeavyTaskNames.add(name);
    this.pendingHeavyTasks.push({ name, callback });
    this.scheduleNextHeavyRender();
  }

  scheduleNextHeavyRender() {
    if (this.heavyFrameHandle || !this.pendingHeavyTasks.length) return;
    this.heavyFrameHandle = window.requestAnimationFrame(() => {
      this.heavyFrameHandle = null;
      const task = this.pendingHeavyTasks.shift();
      if (!task) return;
      this.pendingHeavyTaskNames.delete(task.name);
      try { task.callback(); } catch (error) { console.error(`Snake ${task.name} render error:`, error); }
      this.scheduleNextHeavyRender();
    });
  }

  // Оптимизированное распределение рендеринга при активном воспроизведении
  renderAll() {
    if (!this.state) return;
    const now = performance.now();
    const isRunning = this.state.running;

    const shouldRenderPanels = now - this.lastPanelRenderAt >= (isRunning ? 300 : UI_PANEL_RENDER_MS);
    if (shouldRenderPanels || this.pendingControlSync) { 
      this.syncControlsFromState(); 
      this.updateHeroBadges(); 
      this.updateStats(); 
      this.lastPanelRenderAt = now; 
      this.pendingControlSync = false; 
    }
    
    const snapshot = this.state?.last_death_snapshot;
    const snapshotKey = snapshot ? `${snapshot.step_count ?? "-"}:${snapshot.score ?? "-"}:${snapshot.reason ?? "-"}` : "";
    if (snapshotKey !== this.lastRenderedSnapshotKey || now - this.lastSnapshotRenderAt >= SNAPSHOT_RENDER_MS) { 
      this.renderLastDeathSnapshot(); 
      this.lastRenderedSnapshotKey = snapshotKey; 
      this.lastSnapshotRenderAt = now; 
    }
    
    if (this.showNetwork) {
      const shouldRenderNetworkVisuals = this.shouldUseNetworkVisuals();
      
      // Троттлинг отрисовки сети при запущенной симуляции
      const networkInterval = isRunning ? (this.state?.training ? 900 : 400) : NETWORK_RENDER_MS;
      const shouldRenderNetwork = now - this.lastNetworkRenderAt >= networkInterval;
      
      // Троттлинг графиков при запущенной симуляции
      const chartInterval = isRunning ? 1500 : CHART_RENDER_MS;
      const shouldRenderChart = now - this.lastChartRenderAt >= chartInterval;

      if (shouldRenderNetworkVisuals && shouldRenderNetwork) {
        this.lastNetworkRenderAt = now;
        this.queueHeavyRender("networkGraph", () => { if (this.shouldUseNetworkVisuals()) this.renderNetworkGraph(); });
      }
      if (shouldRenderChart) {
        this.lastChartRenderAt = now;
        this.queueHeavyRender("chart", () => { if (this.showNetwork) this.renderChart(); });
      }
    }
  }

  safeRenderAll() {
    const preferredPollMs = this.state?.training ? TRAINING_API_POLL_MS : API_POLL_MS;
    this.resetPolling(preferredPollMs);
    if (document.hidden) return;
    if (!this.sseConnected) this.resetFramePolling();
    this.startGameRenderLoop();
    this.updateVisualizationVisibility();
    this.scheduleRender();
  }

  updateVisualizationVisibility() {
    const showVisuals = this.shouldUseNetworkVisuals();
    this.setHidden(this.networkPanel, false);
    this.setHidden(this.networkGraphCard, !showVisuals);
  }

  syncControlsFromState() {
    if (this.state.mode === "smart") this.smartContextActive = true;
    else if (this.state.mode === "ai") this.smartContextActive = false;
    if (this.agentModeSelect) {
      if (!this.agentModeFocused) {
        const nextAgentMode = this.smartContextActive ? "smart" : this.state.agent_kind ?? "dqn";
        if (this.agentModeSelect.value !== nextAgentMode) this.agentModeSelect.value = nextAgentMode;
        const smartOption = this.agentModeSelect.querySelector('option[value="smart"]');
        if (smartOption) { smartOption.disabled = false; smartOption.textContent = "Zmey-smart"; }
        const hasSmartState = Boolean(this.state.smart_model);
        const smartLoaded = Boolean(this.state.smart_model?.loaded);
        this.agentModeSelect.title = smartLoaded ? this.state.smart_model?.name ?? "Zmey-smart" : hasSmartState ? this.state.smart_model?.error ?? "Zmey-smart модель недоступна" : "Если режим не сработает, перезапустите backend";
      }
    }
    this.syncSmartModelSelect();
    if (this.boardSizeSelect) { const nextBoardSize = String(this.state.game.size); if (this.boardSizeSelect.value !== nextBoardSize) this.boardSizeSelect.value = nextBoardSize; }
    if (this.speedRange) { const nextSpeed = String(this.state.speed); if (this.speedRange.value !== nextSpeed) this.speedRange.value = nextSpeed; }
    if (this.recordReplaysCheckbox && typeof this.state.record_replays === "boolean") {
      if (this.recordReplaysCheckbox.checked !== this.state.record_replays) {
        this.recordReplaysCheckbox.checked = this.state.record_replays;
      }
    }
    const isSmartMode = this.smartContextActive;
    const isPpoMode = !isSmartMode && this.state.agent_kind === "ppo";
    this.setHidden(this.ppoMultiboardControl, !isPpoMode);
    if (this.ppoMultiboardCheckbox && typeof this.state.ppo_multiboard === "boolean") {
      if (this.ppoMultiboardCheckbox.checked !== this.state.ppo_multiboard) {
        this.ppoMultiboardCheckbox.checked = this.state.ppo_multiboard;
      }
    }
    if (this.ppoMultiboardControl) {
      const mbSummary = String(this.state.ppo_multiboard_summary ?? "");
      this.ppoMultiboardControl.title = this.state.ppo_multiboard && mbSummary
        ? `Мульти-поле PPO (${mbSummary}): упор на выбранную карту + якоря против забывания`
        : "Обучать PPO сразу на нескольких размерах карт (улучшает и главную 10×10, и большие поля)";
    }
    const isEvalRunning = Boolean(this.state.stats?.eval_running);
    const boardSize = Number(this.state.game?.size ?? 10);
    const evalEps = boardSize <= 20 ? 20 : 10;
    const agentBadgeLabel = isSmartMode ? "Zmey-smart" : this.state.agent_kind === "ppo" ? "PPO (Map)" : "DQN";
    this.setText(this.agentPillBadge, agentBadgeLabel);
    this.setText(this.runToggle, this.state.running ? "⏸ Пауза" : "▶ Старт");
    this.toggleClass(this.runToggle, "primary", Boolean(this.state.running));
    this.toggleClass(this.runToggle, "is-paused", !this.state.running);
    this.setText(this.trainToggle, "Обучение");
    if (this.trainToggle) { this.trainToggle.disabled = isSmartMode; this.trainToggle.title = isSmartMode ? "Zmey-smart не обучается в этом режиме" : (this.state.training ? "Выключить обучение" : "Включить обучение"); }
    if (this.evalGreedyButton) {
      const smartUnavailable = isSmartMode && !this.state.smart_model?.loaded;
      this.evalGreedyButton.disabled = isEvalRunning || smartUnavailable;
      this.setText(this.evalGreedyButton, isEvalRunning ? "⏳ Тест..." : `Тест (${evalEps})`);
      this.evalGreedyButton.title = smartUnavailable
        ? `Модель Zmey-smart не загружена: ${this.state.smart_model?.error ?? "нет .h5"}`
        : "Честная greedy-оценка текущей сети без влияния на обучение";
    }
    this.renderEvalResultBar();
    if (this.resetAgentButton) {
      this.resetAgentButton.disabled = false;
      this.resetAgentButton.title = isSmartMode ? "Сброс очистит JSON-статистику Zmey-smart (нажмите дважды)" : "Сбросить агента (нажмите дважды для подтверждения)";
      if (!this.resetConfirmArmed) {
        this.setText(this.resetAgentButton, "↺ Сброс");
        this.toggleClass(this.resetAgentButton, "is-confirm", false);
      }
    }
    this.toggleClass(this.aiModeButton, "is-active", ["ai", "smart"].includes(this.state.mode) && !this.state.training);
    this.toggleClass(this.trainToggle, "is-active", Boolean(this.state.training));
    this.updateVisualizationVisibility();
  }

  renderEvalResultBar() {
    if (!this.evalResultBar) return;
    const isRunning = Boolean(this.state?.stats?.eval_running);
    const res = this.state?.stats?.last_eval_result;
    if (isRunning) {
      this.setHidden(this.evalResultBar, false);
      this.setHtml(this.evalResultBar, "Выполняется честный greedy-тест...");
      return;
    }
    const activeKind = this.state?.mode === "smart" ? "smart" : String(this.state?.agent_kind ?? "");
    const activeBoard = Number(this.state?.game?.size ?? 10);
    if (
      !res
      || typeof res !== "object"
      || (res.agent_kind && res.agent_kind !== activeKind)
      || (res.board_size && Number(res.board_size) !== activeBoard)
    ) {
      this.setHidden(this.evalResultBar, true);
      return;
    }
    this.setHidden(this.evalResultBar, false);
    const bs = Number(res.board_size ?? 10);
    const eps = Number(res.episodes ?? 0);
    const winPct = Number(res.win_rate_pct ?? 0).toFixed(1);
    const avgScore = Number(res.avg_score ?? 0).toFixed(1);
    const maxV = Number(res.victory_score ?? (bs * bs - 3));
    const fillPct = Number(res.avg_fill_pct ?? 0).toFixed(1);
    const pureTag = res.pure_policy ? " • pure" : "";
    this.setHtml(
      this.evalResultBar,
      `Тест ${bs}×${bs} (${eps} игр${pureTag}): <strong>${winPct}% побед</strong> • ср. ${avgScore}/${maxV} (${fillPct}%)`
    );
  }

  syncSmartModelSelect() {
    const isSmartMode = this.state?.mode === "smart" || this.agentModeSelect?.value === "smart";
    this.setHidden(this.smartModelControl, !isSmartMode);
    if (!this.smartModelSelect || this.smartModelFocused) return;
    const models = Array.isArray(this.state?.smart_model?.models) ? this.state.smart_model.models : [];
    const activeName = this.state?.smart_model?.name ?? "";
    const markup = models.length ? models.map((model) => {
      const name = String(model.name ?? "");
      const label = name || String(model.label ?? model.epoch ?? "");
      const selected = name === activeName ? " selected" : "";
      return `<option value="${escapeText(name)}" title="${escapeText(name)}"${selected}>${escapeText(label)}</option>`;
    }).join("") : '<option value="">Нет .h5</option>';
    if (this.lastSmartModelMarkup !== markup) { this.lastSmartModelMarkup = markup; this.smartModelSelect.innerHTML = markup; }
    this.smartModelSelect.disabled = !models.length;
    if (activeName && this.smartModelSelect.value !== activeName) this.smartModelSelect.value = activeName;
    this.smartModelSelect.title = this.state?.smart_model?.error || activeName || "Файлы .h5 из .runtime";
  }

  captureStableHeroProfile() {
    if (!this.state) return;
    const signalCount = (this.state?.global_signals?.labels ?? []).length;
    const isSmartContext = this.smartContextActive || this.state.mode === "smart";
    const isPpo = !isSmartContext && this.state.agent_kind === "ppo";
    const ppoMapSize = Number(this.state?.ppo_map_size ?? this.state?.network?.ego_map?.[0]?.length ?? 11);
    const inputLabel = isPpo ? `${signalCount} сигн. + ${ppoMapSize}×${ppoMapSize}` : `${signalCount} сигналов`;
    this.stableHeroProfile = {
      device: this.state.device_kind ?? "CPU",
      boardSize: this.state?.game?.size ?? 10,
      inputLabel,
      networkSource: isSmartContext ? "Zmey-smart" : this.state.network_source ?? "DQN policy",
    };
  }

  updateHeroBadges() {
    if (this.stateDetailsFresh || !this.stableHeroProfile) this.captureStableHeroProfile();
    const profile = this.stableHeroProfile;
    if (!profile) return;
    this.setText(this.deviceBadge, profile.device);
    this.setText(this.boardSizeBadge, `Карта ${profile.boardSize}x${profile.boardSize}`);
    this.setText(this.inputGridBadge, `Вход: ${profile.inputLabel}`);
    this.setText(this.networkSourceBadge, `Нейросеть: ${profile.networkSource}`);
  }

  renderDecisionSummary() {
    if (!this.decisionText) return;
    if (this.state?.restart_pending || Date.now() < this.restartPendingUntil) {
      this.setHtml(this.decisionMeta, "");
      this.setText(this.decisionText, "Backend перезапускается. Состояние сохранено.");
      return;
    }
    const mode = this.state?.mode;
    const agentKind = this.state?.agent_kind;
    if (mode === "smart") {
      const smartLoaded = Boolean(this.state?.smart_model?.loaded);
      this.setHtml(this.decisionMeta, smartLoaded ? "<strong>Zmey-smart</strong>" : "не загружена");
      if (!smartLoaded) {
        const err = this.state?.smart_model?.error || "файл .h5 не загружен";
        this.setText(this.decisionText, `Zmey-smart недоступен: ${err}`);
        return;
      }
    } else if (agentKind === "ppo") {
      const valueEst = Number(this.state?.stats?.value_estimate ?? 0);
      this.setHtml(this.decisionMeta, `V(s) <strong>${valueEst >= 0 ? "+" : ""}${valueEst.toFixed(2)}</strong>`);
    } else {
      this.setHtml(this.decisionMeta, "Q-values");
    }

    const values = Array.isArray(this.state?.q_values) ? this.state.q_values : [];
    const chosen = Number.isInteger(this.state?.last_greedy_action) ? this.state.last_greedy_action : -1;
    const order = this.outputDisplayOrder();
    if (!values.length || !order.length) {
      this.setText(this.decisionText, "Ожидание первого шага...");
      return;
    }
    const pills = order.map((idx) => {
      const val = Number(values[idx]);
      const valText = Number.isFinite(val) ? val.toFixed(2) : "-";
      const cls = idx === chosen ? "decision-pill is-chosen" : "decision-pill";
      const label = this.outputLabels()[idx] ?? "";
      return `<span class="${cls}" title="${escapeText(label)}"><span class="dir">${this.arrowForAction(idx)}</span><span>${valText}</span></span>`;
    }).join("");
    this.setHtml(this.decisionText, `<div class="decision-pills">${pills}</div>`);
  }

  buildDecisionText() {
    if (this.state?.restart_pending || Date.now() < this.restartPendingUntil) {
      return "Backend перезапускается. Состояние сохранено.";
    }
    return this.formatOrderedQValues() || "-";
  }

  getDeathSteps(snapshot) {
    const recentSteps = Array.isArray(snapshot?.recent_steps) ? snapshot.recent_steps : [];
    return recentSteps.length ? recentSteps : snapshot ? [snapshot] : [];
  }

  renderDeathRiskTable(selectedStep) {
    if (!this.deathRiskTable) return;
    const risks = selectedStep?.move_risks ?? null;
    if (!risks) { this.setHtml(this.deathRiskTable, ""); return; }
    
    const percent = (value) => {
      const num = Number(value ?? 0);
      return `${Math.round(num * 100)}%`;
    };
    
    const getRiskValues = (dirKey) => {
      const raw = risks?.[dirKey];
      if (raw && typeof raw === "object") {
        return {
          danger: percent(raw.danger),
          tail_blocked: percent(raw.tail_blocked),
          space_tight: percent(raw.space_tight)
        };
      }
      return {
        danger: percent(raw),
        tail_blocked: "-",
        space_tight: "-"
      };
    };

    const rows = [
      { action: 0, key: "forward" },
      { action: 1, key: "left" },
      { action: 2, key: "right" }
    ].map(({ action, key }) => {
      const arrow = this.arrowForAction(action);
      const vals = getRiskValues(key);
      
      const isRandom = selectedStep && selectedStep.action_random === true;
      const isChosenAction = selectedStep && Number.isInteger(selectedStep.action) && action === Number(selectedStep.action);
      const isGreedyAction = selectedStep && Number.isInteger(selectedStep.greedy_action) && action === Number(selectedStep.greedy_action);

      let rowClass = "";
      let label = arrow;

      if (isRandom) {
        if (isChosenAction) {
          rowClass = ' class="is-selected-random"';
          const tooltip = isGreedyAction ? "Случайный ход (совпал с выбором сети)" : "Случайный ход (исследование)";
          label = `${arrow} <span style="cursor: help; margin-left: 4px; font-size: 0.85rem;" title="${tooltip}">🎲</span>`;
        } else if (isGreedyAction) {
          rowClass = ' class="is-greedy-choice"';
          label = `${arrow} <span style="cursor: help; margin-left: 4px; font-size: 0.85rem;" title="Выбор нейросети">🧠</span>`;
        }
      } else {
        if (isChosenAction) {
          rowClass = ' class="is-selected"';
        }
      }

      return [
        `<tr${rowClass}>`,
        `<td>${label}</td>`,
        `<td>${vals.danger}</td>`,
        `<td>${vals.tail_blocked}</td>`,
        `<td>${vals.space_tight}</td>`,
        "</tr>"
      ].join("");
    }).join("");
    
    this.setHtml(
      this.deathRiskTable,
      `<table class="death-risk-table"><thead><tr><th>выбор</th><th>СТЕНА</th><th>ХВОСТ</th><th>ТЕСНО</th></tr></thead><tbody>${rows}</tbody></table>`
    );
  }

  getSelectedDeathStep(snapshot) {
    const steps = this.getDeathSteps(snapshot);
    if (!steps.length) return { steps, selectedStep: null, selectedIndex: -1 };
    const maxIndex = steps.length - 1;
    const selectedIndex = this.selectedDeathStepIndex >= 0 ? Math.min(this.selectedDeathStepIndex, maxIndex) : maxIndex;
    return { steps, selectedStep: steps[selectedIndex], selectedIndex };
  }

  renderDeathStepTimeline(snapshot) {
    if (!this.deathStepTimeline) return;
    const { steps, selectedIndex } = this.getSelectedDeathStep(snapshot);
    if (!steps.length) { if (this.lastDeathTimelineMarkup) { this.lastDeathTimelineMarkup = ""; this.setHtml(this.deathStepTimeline, ""); } return; }
    const markup = steps.map((step, index) => {
      const offset = index - steps.length;
      const label = `${offset}`;
      const isActive = index === selectedIndex ? " is-active" : "";
      const stepCount = step?.step_count ?? "-";
      return `<button type="button" class="${isActive.trim()}" data-death-step="${index}" title="Шаг ${stepCount}">${label}</button>`;
    }).join("");
    if (markup !== this.lastDeathTimelineMarkup) { this.lastDeathTimelineMarkup = markup; this.setHtml(this.deathStepTimeline, markup); }
  }

  formatStatPercent(count, total) { if (!total) return "0%"; return `${((count / total) * 100).toFixed(0)}%`; }

  outcomeVisualMeta(key) {
    if (key === "victory") return { label: "Победа", color: "#5cd6ad" };
    if (key === "body_collision") return { label: "Тело", color: "#ff6b6b" };
    if (key === "wall_collision") return { label: "Стена", color: "#ff9b71" };
    if (key === "starvation") return { label: "Голод", color: "#ffb347" };
    return { label: formatReason(key), color: "#9eb6c0" };
  }

  renderDecisionStats() {
    if (!this.decisionStatsTable) return;
    const stats = this.state.decision_stats ?? {};
    const bucket = stats.deaths ?? {};
    const allCounts = bucket.all ?? {};
    const recentList = Array.isArray(bucket.recent) ? bucket.recent : [];
    const recentCounts = {};
    recentList.forEach((item) => { recentCounts[item] = (recentCounts[item] ?? 0) + 1; });
    const terminalReasonsOrdered = Array.from(TERMINAL_REASONS);
    const outcomeOrder = ["victory", "body_collision", "wall_collision", "starvation", ...terminalReasonsOrdered.filter((r) => !["victory", "body_collision", "wall_collision", "starvation"].includes(r))];
    const keys = [...Object.keys(allCounts)].filter((key) => Number(allCounts[key] ?? 0) > 0).sort((a, b) => {
      const left = outcomeOrder.includes(a) ? outcomeOrder.indexOf(a) : outcomeOrder.length;
      const right = outcomeOrder.includes(b) ? outcomeOrder.indexOf(b) : outcomeOrder.length;
      return left - right || a.localeCompare(b);
    });
    const allTotal = Object.values(allCounts).reduce((sum, value) => sum + Number(value || 0), 0);
    const recentTotal = recentList.length;
    const header = `<div class="ctrl-outcomes-head"><span>Итог</span><span>100 посл.</span><span>Все</span></div>`;
    if (!keys.length) {
      const emptyMarkup = `${header}<div class="ctrl-outcomes-empty">Еще нет данных о завершенных эпизодах.</div>`;
      if (emptyMarkup !== this.lastDecisionStatsMarkup) { this.lastDecisionStatsMarkup = emptyMarkup; this.setHtml(this.decisionStatsTable, emptyMarkup); }
      return;
    }
    const rows = keys.map((key) => {
      const allCount = Number(allCounts[key] ?? 0);
      const recentCount = Number(recentCounts[key] ?? 0);
      const recentPctNum = recentTotal ? Math.round((recentCount / recentTotal) * 100) : 0;
      const allPctText = this.formatStatPercent(allCount, allTotal);
      const meta = this.outcomeVisualMeta(key);
      const fullLabel = formatReason(key);
      return [
        `<div class="ctrl-outcome-row" title="${escapeText(fullLabel)}: ${recentCount}/${recentTotal || 0} за 100 посл., ${allCount}/${allTotal} всего">`,
        `<span class="ctrl-outcome-name"><span class="ctrl-outcome-dot" style="background:${meta.color}"></span>${escapeText(meta.label)}</span>`,
        `<span class="ctrl-outcome-recent"><span class="ctrl-outcome-bar"><span style="width:${recentPctNum}%;background:${meta.color}"></span></span><strong>${recentPctNum}%</strong></span>`,
        `<span class="ctrl-outcome-all">${allPctText} <small>(${allCount})</small></span>`,
        `</div>`,
      ].join("");
    });
    const markup = `${header}${rows.join("")}`;
    if (markup !== this.lastDecisionStatsMarkup) { this.lastDecisionStatsMarkup = markup; this.setHtml(this.decisionStatsTable, markup); }
  }

  renderLastDeathSnapshot() {
    if (!this.deathSnapshotCtx || !this.deathSnapshotCanvas || !this.state) return;

    const snapshot = this.state?.last_death_snapshot;
    const dim = this.canvasDimensions.get(this.deathSnapshotCanvas) || { width: this.deathSnapshotCanvas.width, height: this.deathSnapshotCanvas.height };
    const snapshotKey = snapshot ? `${snapshot.step_count ?? "-"}:${snapshot.score ?? "-"}:${snapshot.reason ?? "-"}` : "";
    const renderSignature = [dim.width, dim.height, snapshotKey, this.selectedDeathStepIndex].join("|");

    if (!dim.dirty && renderSignature === this.lastSnapshotRenderSignature) return;
    this.lastSnapshotRenderSignature = renderSignature;

    const ctx = this.deathSnapshotCtx;
    const { width, height } = this.resizeCanvas(this.deathSnapshotCanvas, ctx);

    if (snapshotKey !== this.lastDeathSnapshotKey) { this.lastDeathSnapshotKey = snapshotKey; this.selectedDeathStepIndex = -1; }
    const { selectedStep, selectedIndex } = this.getSelectedDeathStep(snapshot);
    this.renderDeathStepTimeline(snapshot);
    this.renderDeathRiskTable(selectedStep);
    const outcomeReason = selectedStep?.outcome_reason ?? selectedStep?.reason ?? snapshot?.reason;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0c1924";
    ctx.fillRect(0, 0, width, height);
    if (!selectedStep || !Array.isArray(selectedStep.snake) || !selectedStep.food) {
      this.renderDeathRiskTable(null);
      this.setHidden(this.deathChoiceMeta, true);
      this.setText(this.deathChoiceMeta, "");
      ctx.fillStyle = "#edf6f9";
      ctx.textAlign = "center";
      ctx.font = "600 18px Aptos, Trebuchet MS, sans-serif";
      ctx.fillText("Снимок пока недоступен", width / 2, height / 2 - 8);
      ctx.fillStyle = "rgba(180,199,207,0.82)";
      ctx.font = "500 14px Aptos, Trebuchet MS, sans-serif";
      drawCanvasLines(ctx, ["После первого эпизода", "здесь появится карта."], width / 2, height / 2 + 18, 18);
      ctx.textAlign = "start";
      this.setText(this.deathSnapshotMeta, "");
      return;
    }
    const size = Number(selectedStep.size ?? 0) || 1;
    const { boardX, boardY, boardSize, cell, snake, head } = SnakeRenderers.drawSnakeScene(ctx, width, height, { size, snake: selectedStep.snake ?? [], food: selectedStep.food, eatenFoodTrail: selectedStep.eaten_food_trail ?? [] }, { variant: "death", headOptions: { direction: selectedStep.direction } });
    const plannedTarget = selectedStep.planned_target;
    const hasPlannedTarget = plannedTarget && Number.isFinite(plannedTarget.x) && Number.isFinite(plannedTarget.y);
    if (head && hasPlannedTarget) {
      const startX = boardX + head.x * cell + cell / 2;
      const startY = boardY + head.y * cell + cell / 2;
      let endX = boardX + clamp(plannedTarget.x, 0, size - 1) * cell + cell / 2;
      let endY = boardY + clamp(plannedTarget.y, 0, size - 1) * cell + cell / 2;
      const edgeInset = Math.max(3, cell * 0.12);
      if (plannedTarget.x < 0) endX = boardX + edgeInset;
      else if (plannedTarget.x >= size) endX = boardX + boardSize - edgeInset;
      if (plannedTarget.y < 0) endY = boardY + edgeInset;
      else if (plannedTarget.y >= size) endY = boardY + boardSize - edgeInset;
      const angle = Math.atan2(endY - startY, endX - startX);
      const headLength = Math.max(7, cell * 0.24);
      ctx.save();
      ctx.strokeStyle = "rgba(255, 104, 124, 0.95)";
      ctx.fillStyle = "rgba(255, 104, 124, 0.95)";
      ctx.lineWidth = Math.max(2, cell * 0.08);
      ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(startX, startY); ctx.lineTo(endX, endY); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(endX, endY); ctx.lineTo(endX - headLength * Math.cos(angle - Math.PI / 6), endY - headLength * Math.sin(angle - Math.PI / 6)); ctx.lineTo(endX - headLength * Math.cos(angle + Math.PI / 6), endY - headLength * Math.sin(angle + Math.PI / 6)); ctx.closePath(); ctx.fill();
      ctx.restore();
    }
    this.setText(this.deathChoiceMeta, "");
    this.setHidden(this.deathChoiceMeta, true);

    const statsLines = [
      `Итог: ${formatReason(outcomeReason)}`,
      `Счет: ${selectedStep.score ?? 0} • Шагов: ${selectedStep.step_count ?? 0} • Длина: ${snake.length}`
    ];
    this.setText(this.deathSnapshotMeta, statsLines.join("\n"));
  }

  renderRewardRules() {
    if (!this.rewardRulesBody) return;
    const rules = this.state?.reward_rules ?? {};
    const bonuses = Array.isArray(rules.bonuses) ? rules.bonuses : [];
    const penalties = Array.isArray(rules.penalties) ? rules.penalties : [];
    const renderItems = (items, toneClass) => items.map((item) => {
      const title = item.formula ? ` title="${escapeText(item.formula)}"` : "";
      return `<span class="${toneClass}"${title}>${escapeText(item.label ?? "-")} <code>${escapeText(item.value ?? "")}</code></span>`;
    }).join("");
    const sections = [["Бонусы", bonuses, "is-good"], ["Штрафы", penalties, "is-bad"]].filter(([, items]) => items.length > 0);
    const noteText = String(rules.note ?? "").trim();
    const noteHtml = noteText ? `<div class="ctrl-reward-note">${escapeText(noteText)}</div>` : "";
    const markup = sections.map(([title, items, toneClass]) => `<div class="ctrl-reward-group"><span class="ctrl-reward-title">${title}</span><div class="reward-list">${renderItems(items, toneClass)}</div></div>`).join("") + noteHtml;
    if (markup !== this.lastRewardRulesMarkup) { this.lastRewardRulesMarkup = markup; this.setHtml(this.rewardRulesBody, markup); }
  }

  renderHungerStat() {
    if (!this.epsilonStat) return;
    const steps = Math.max(0, Number(this.state.stats?.hunger_steps ?? this.state.game?.steps_without_food ?? 0));
    const budget = Math.max(1, Number(this.state.stats?.hunger_budget ?? 1));
    const ratio = clamp(Number(this.state.stats?.hunger_ratio ?? steps / budget), 0, 1);
    const percentValue = Math.round(ratio * 100);
    const className = ratio >= 0.9 ? "is-danger" : ratio >= 0.65 ? "is-warn" : "";
    const text = `${Math.round(steps)}/${Math.round(budget)}`;
    this.setHtml(
      this.epsilonStat,
      `<span class="ctrl-hunger ${className}" title="Шагов без яблока: ${Math.round(steps)} из ${Math.round(budget)}"><span class="ctrl-hunger-label">Голод</span><span class="ctrl-hunger-track"><span class="ctrl-hunger-fill" style="--hunger-width: ${percentValue}%"></span></span><span class="ctrl-hunger-val">${escapeText(text)}</span></span>`
    );
  }

  updateStats() {
    const modeLabels = { ai: "ИИ", smart: "Zmey-smart" };
    const isSmart = this.state.mode === "smart";
    const isPpo = this.state.agent_kind === "ppo" && !isSmart;
    const isDqn = this.state.agent_kind === "dqn" && !isSmart;
    const modeText = this.state.training ? "Обучение" : modeLabels[this.state.mode] ?? "ИИ";
    this.setText(this.modeStat, modeText);
    this.setText(this.epochStat, String(this.state.stats.episode));
    this.setText(this.scoreStat, String(this.state.game.score));
    this.updateGameHeaderMeta();
    this.setText(this.bestScoreStat, String(this.state.stats.best_score));
    this.setText(this.avgScoreStat, Number(this.state.stats.average_score_100 ?? 0).toFixed(1));
    if (this.avgScoreStat) { const averageWindow = Number(this.state.stats.average_score_window ?? 0); this.avgScoreStat.title = averageWindow ? `Средний счет по ${averageWindow} эпизодам` : ""; }
    this.renderHungerStat();
    if (isSmart) {
      this.setText(this.updatesLabel, "Статус");
      this.setText(this.leaderStat, this.state.running ? "run" : "idle");
      this.setText(this.auxMetricLabel, "Модель");
      this.setText(this.auxMetricStat, this.state.smart_model?.loaded ? "ok" : "нет");
      if (this.auxMetricStat) this.auxMetricStat.title = this.state.smart_model?.name ?? "";
    } else if (isPpo) {
      this.setText(this.updatesLabel, "Updates");
      this.setText(this.leaderStat, String(this.state.stats.updates ?? 0));
      this.setText(this.auxMetricLabel, "Tail bias");
      const effBias = Number(this.state.stats.effective_tail_bias ?? 0);
      this.setText(this.auxMetricStat, effBias <= 0.001 ? "0.00" : effBias.toFixed(2));
      if (this.auxMetricStat) this.auxMetricStat.title = "Мягкий бутстрап tail_blocked: плавно зануляется 0.50 → 0.00 при росте заполнения поля";
    } else {
      this.setText(this.updatesLabel, "Updates");
      this.setText(this.leaderStat, String(this.state.stats.updates ?? 0));
      this.setText(this.auxMetricLabel, "Epsilon");
      this.setText(this.auxMetricStat, Number(this.state.stats.epsilon ?? 0).toFixed(3));
      if (this.auxMetricStat) this.auxMetricStat.title = "Доля случайных ходов (exploration) DQN";
    }
    const ppoMapSize = Number(this.state?.ppo_map_size ?? this.state?.network?.ego_map?.[0]?.length ?? 11);
    this.setText(this.chartMetricLabel, isSmart ? "Счет" : "Loss");
    this.setText(
      this.chartMetricDescription,
      isSmart
        ? "История Zmey-smart по выбранной модели: счет, награда и средний счет за последние эпизоды."
        : isPpo
          ? `Суммарный Loss (policy + value - entropy) при обновлении PPO с эгоцентричной картой ${ppoMapSize}×${ppoMapSize}.`
          : "Ошибка DQN при обновлении Q-сети: чем ниже значение, тем стабильнее прогнозы."
    );
    this.renderDecisionSummary();
    this.renderDecisionStats();
    this.renderRewardRules();
  }

  // Оптимизированный 60 FPS рендеринг игры с субклеточной интерполяцией и кэшем фона
  renderGame(now = performance.now()) {
    if (!this.gameCtx || !this.gameCanvas || !this.state?.game) return;
    const ctx = this.gameCtx;
    const game = this.currGameSnapshot ?? this.state.game;
    const size = game.size;

    const dim = this.canvasDimensions.get(this.gameCanvas) || { width: this.gameCanvas.width, height: this.gameCanvas.height };
    const canInterp = Boolean(this.state.running && this.canInterpolateSnake(this.prevGameSnapshot, game));
    const stepMs = Math.max(16, this.gameStepDurationMs || 125);
    const t = canInterp ? clamp((now - this.gameTransitionStart) / stepMs, 0, 1) : 1;

    const frameId = this.state.frame_id ?? `${game.score}:${game.step_count}:${game.direction}`;
    const settledSignature = [dim.width, dim.height, size, frameId, game.step_count, game.score, this.state.training ? 1 : 0].join("|");
    if (!dim.dirty && t >= 0.999 && settledSignature === this.lastSettledGameSignature) return;

    const { width, height, resized } = this.resizeCanvas(this.gameCanvas, ctx);

    if (resized || !this.bgDrawn || this.lastGameBoardSize !== size) {
      this.gameBgCanvas.width = this.gameCanvas.width;
      this.gameBgCanvas.height = this.gameCanvas.height;
      const dpr = Math.max(1, window.devicePixelRatio || 1);
      this.gameBgCtx.setTransform(dpr, 0, 0, dpr, 0, 0);

      this.boardLayout = SnakeRenderers.drawStaticBackground(this.gameBgCtx, width, height, size);
      this.lastGameBoardSize = size;
      this.bgDrawn = true;
    }

    ctx.clearRect(0, 0, width, height);

    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.drawImage(this.gameBgCanvas, 0, 0);
    ctx.restore();

    const rawSnake = game.snake ?? [];
    const { snake, interpolated } = canInterp
      ? this.buildInterpolatedSnake(this.prevGameSnapshot, game, t)
      : { snake: rawSnake, interpolated: false };
    const food = game.food;
    const eatenFoodTrail = game.eaten_food_trail ?? [];
    const occupiedBody = new Set(rawSnake.slice(1).map((segment) => `${segment.x}:${segment.y}`));

    SnakeRenderers.drawSnakeScene(
      ctx,
      width,
      height,
      { size, snake, food, eatenFoodTrail },
      {
        boardPadding: 18,
        skipFrame: true,
        interpolated,
        showTrail: !this.state.training,
        trailOptions: { guard: (marker) => occupiedBody.has(`${marker.x}:${marker.y}`) },
        headOptions: { direction: game.direction }
      }
    );

    this.lastGameRenderTime = now;
    if (t >= 0.999) {
      this.lastSettledGameSignature = settledSignature;
    }
  }

  renderNetworkGraph() {
    if (!this.networkCtx || !this.networkCanvas || !this.state) return;

    const dim = this.canvasDimensions.get(this.networkCanvas) || { width: this.networkCanvas.width, height: this.networkCanvas.height };
    const frameId = this.state.frame_id ?? "";
    const hoverSig = this.networkHoverState
      ? `${this.networkHoverState.layer}:${this.networkHoverState.index}:${Math.round(this.networkHoverState.anchorX)},${Math.round(this.networkHoverState.anchorY)}`
      : "";
    const renderSignature = [dim.width, dim.height, frameId, hoverSig].join("|");

    if (!dim.dirty && renderSignature === this.lastNetworkRenderSignature) return;
    this.lastNetworkRenderSignature = renderSignature;

    const ctx = this.networkCtx;
    const { width, height } = this.resizeCanvas(this.networkCanvas, ctx);
    const labels = this.outputLabels();
    SnakeRenderers.drawNetworkGraph(
      ctx,
      width,
      height,
      this.state,
      labels,
      this.outputDisplayOrder(labels),
      this.networkHoverState,
      this.networkHitCache
    );
  }

  renderChart() {
    if (!this.chartCtx || !this.chartCanvas || !this.state) return;

    const dim = this.canvasDimensions.get(this.chartCanvas) || { width: this.chartCanvas.width, height: this.chartCanvas.height };
    const points = this.state.history ?? [];
    const isSmart = this.state.mode === "smart";
    const usesLossMetric = !isSmart;
    const lastEpisode = points.length ? points[points.length - 1].episode ?? "" : "";
    const renderSignature = [dim.width, dim.height, points.length, lastEpisode, isSmart, usesLossMetric].join("|");

    if (!dim.dirty && renderSignature === this.lastChartRenderSignature) return;
    this.lastChartRenderSignature = renderSignature;

    const ctx = this.chartCtx;
    const { width, height } = this.resizeCanvas(this.chartCanvas, ctx);
    SnakeRenderers.drawTrainingChart(ctx, width, height, points, isSmart, usesLossMetric);
  }
}

new SnakeLabApp();

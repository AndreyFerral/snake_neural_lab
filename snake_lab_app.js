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

    this.inputModeSelect = document.getElementById("inputMode");
    this.agentModeSelect = document.getElementById("agentMode");
    this.smartModelControl = document.getElementById("smartModelControl");
    this.smartModelSelect = document.getElementById("smartModel");
    this.boardSizeSelect = document.getElementById("boardSize");
    this.speedRange = document.getElementById("speedRange");
    this.runToggle = document.getElementById("runToggle");
    this.manualModeButton = document.getElementById("manualMode");
    this.aiModeButton = document.getElementById("aiMode");
    this.trainToggle = document.getElementById("trainToggle");
    this.resetAgentButton = document.getElementById("resetAgent");
    this.restartBackendButton = document.getElementById("restartBackend");
    this.shutdownBackendButton = document.getElementById("shutdownBackend");

    this.networkPanel = document.getElementById("networkPanel");
    this.networkGraphCard = document.getElementById("networkGraphCard");

    this.deviceBadge = document.getElementById("deviceBadge");
    this.boardSizeBadge = document.getElementById("boardSizeBadge");
    this.inputGridBadge = document.getElementById("inputGridBadge");
    this.networkSourceBadge = document.getElementById("networkSourceBadge");

    this.modeStat = document.getElementById("modeStat");
    this.epochStat = document.getElementById("epochStat");
    this.scoreStat = document.getElementById("scoreStat");
    this.bestScoreStat = document.getElementById("bestScoreStat");
    this.avgScoreStat = document.getElementById("avgScoreStat");
    this.lossStat = document.getElementById("lossStat");
    this.epsilonStat = document.getElementById("epsilonStat");
    this.bufferStat = document.getElementById("bufferStat");
    this.leaderStat = document.getElementById("leaderStat");
    this.aliveStat = document.getElementById("aliveStat");
    this.chartMetricLabel = document.getElementById("chartMetricLabel");
    this.chartMetricDescription = document.getElementById("chartMetricDescription");
    this.stepStat = document.getElementById("stepStat");
    this.rewardRulesBody = document.getElementById("rewardRulesBody");

    this.stepReviewSlider = document.getElementById("stepReviewSlider");
    this.stepReviewLiveButton = document.getElementById("stepReviewLive");
    this.stepReviewPreviousButton = document.getElementById("stepReviewPrevious");
    this.stepReviewLabel = document.getElementById("stepReviewLabel");
    this.stepReviewBody = document.getElementById("stepReviewBody");
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
    this.lastStepReviewSignature = "";
    this.lastStepReviewReason = "";
    this.lastStepReviewHistoryId = "";
    this.lastGameRenderSignature = "";
    this.lastRewardRulesMarkup = "";
    this.lastDecisionStatsMarkup = "";
    this.lastSummaryMarkup = "";
    this.forceStepLogScroll = false;
    this.selectedReviewStepIndex = -1;
    this.selectedStepLogKey = "live";
    this.agentModeFocused = false;
    this.smartModelFocused = false;
    this.smartContextActive = false;
    this.lastSmartModelMarkup = "";
    this.resizeFrameHandle = null;
    this.renderFrameHandle = null;
    this.liveFrameRenderHandle = null;
    this.heavyFrameHandle = null;
    this.pendingHeavyTasks = [];
    this.pendingHeavyTaskNames = new Set();
    this.canvasSizeCache = new WeakMap();
    this.gameBackgroundCache = null;
    this.pendingControlSync = false;

    this.setHidden(this.networkPanel, false);
    this.setDecisionText("Подключение к Python backend...");
    this.bindEvents();
    this.refreshState();
    this.resetPolling(this.currentPollMs);
    this.resetFramePolling();
  }

  getCanvasContext(canvas) {
    return canvas ? canvas.getContext("2d") : null;
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
    if (!this.state || this.state.running) return FRAME_POLL_MS;
    return FRAME_IDLE_POLL_MS;
  }

  resetFramePolling() {
    if (this.framePollHandle) return;
    const pollFrame = async () => {
      this.framePollHandle = null;
      if (!document.hidden) await this.refreshFrame();
      if (!this.framePollHandle) this.framePollHandle = window.setTimeout(pollFrame, this.nextFramePollDelay());
    };
    this.framePollHandle = window.setTimeout(pollFrame, 0);
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

  arrowForAbsoluteDirection(direction) {
    if (direction === 0) return "↑";
    if (direction === 1) return "→";
    if (direction === 2) return "↓";
    if (direction === 3) return "←";
    return "-";
  }

  currentInputGrid() {
    return this.state?.input_grid ?? DEFAULT_INPUT_GRID;
  }

  currentInputLabel() {
    const inputGrid = this.currentInputGrid();
    return inputGrid > 0 ? `${inputGrid}x${inputGrid}` : "Без карты";
  }

  inputMode() {
    return this.state?.input_mode ?? "signals";
  }

  shouldShowNetworkGraph() {
    return this.inputMode() !== "map";
  }

  shouldUseNetworkVisuals() {
    return Boolean(this.showNetwork && !this.state?.training && !document.hidden);
  }

  resizeCanvas(canvas, ctx) {
    if (!canvas || !ctx) return { width: 0, height: 0, resized: false };
    const dpr = Math.max(1, window.devicePixelRatio || 1);
    const width = Math.max(1, Math.round(canvas.clientWidth || canvas.width || 1));
    const height = Math.max(1, Math.round(canvas.clientHeight || canvas.height || 1));
    const targetWidth = Math.round(width * dpr);
    const targetHeight = Math.round(height * dpr);
    const cached = this.canvasSizeCache.get(canvas);
    const resized = !cached || cached.width !== width || cached.height !== height || cached.dpr !== dpr || canvas.width !== targetWidth || canvas.height !== targetHeight;
    if (resized) {
      canvas.width = targetWidth;
      canvas.height = targetHeight;
      this.canvasSizeCache.set(canvas, { width, height, dpr });
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { width, height, resized };
  }

  pickTopIndices(values, count) {
    const len = values.length;
    if (len === 0) return [];
    const indexed = new Array(len);
    for (let i = 0; i < len; i++) indexed[i] = [Math.abs(values[i]), i];
    indexed.sort((a, b) => b[0] - a[0]);
    const result = [];
    for (let i = 0; i < count && i < len; i++) result.push(indexed[i][1]);
    result.sort((a, b) => a - b);
    return result;
  }

  roundedRect(ctx, x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + width, y, x + width, y + height, r);
    ctx.arcTo(x + width, y + height, x, y + height, r);
    ctx.arcTo(x, y + height, x, y, r);
    ctx.arcTo(x, y, x + width, y, r);
    ctx.closePath();
  }

  drawBoardFrame(ctx, boardX, boardY, boardSize, cells, options = {}) {
    const glow = Number(options.glow ?? 6);
    const strokeOffset = glow - 0.75;
    ctx.fillStyle = "rgba(255,255,255,0.04)";
    ctx.fillRect(boardX - glow, boardY - glow, boardSize + glow * 2, boardSize + glow * 2);
    ctx.strokeStyle = "rgba(255,255,255,0.09)";
    ctx.lineWidth = Number(options.outerLineWidth ?? 1.5);
    ctx.strokeRect(boardX - strokeOffset, boardY - strokeOffset, boardSize + strokeOffset * 2, boardSize + strokeOffset * 2);
    ctx.fillStyle = "rgba(13, 30, 41, 0.96)";
    ctx.fillRect(boardX, boardY, boardSize, boardSize);
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth = Number(options.innerLineWidth ?? 1.5);
    const inset = ctx.lineWidth / 2;
    ctx.strokeRect(boardX + inset, boardY + inset, boardSize - inset * 2, boardSize - inset * 2);
    const cell = boardSize / cells;
    ctx.strokeStyle = "rgba(255,255,255,0.05)";
    ctx.lineWidth = 1;
    for (let i = 1; i < cells; i += 1) {
      const offset = Math.round(i * cell) + 0.5;
      ctx.beginPath();
      ctx.moveTo(boardX + offset, boardY);
      ctx.lineTo(boardX + offset, boardY + boardSize);
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(boardX, boardY + offset);
      ctx.lineTo(boardX + boardSize, boardY + offset);
      ctx.stroke();
    }
    return cell;
  }

  fillBoardCell(ctx, boardX, boardY, cell, gridX, gridY, options = {}) {
    const inset = cell * Number(options.insetRatio ?? 0);
    const sizePx = cell - inset * 2;
    ctx.fillStyle = options.fill ?? "rgba(13, 30, 41, 0.68)";
    if (options.rounded === false) {
      ctx.fillRect(boardX + gridX * cell + inset, boardY + gridY * cell + inset, sizePx, sizePx);
      return;
    }
    this.roundedRect(
      ctx,
      boardX + gridX * cell + inset,
      boardY + gridY * cell + inset,
      sizePx,
      sizePx,
      Math.max(Number(options.minRadius ?? 4), cell * Number(options.radiusRatio ?? 0.24))
    );
    ctx.fill();
  }

  drawApple(ctx, boardX, boardY, cell, food, { includeStem = true } = {}) {
    if (!food) return;
    const appleSize = cell * 0.84;
    ctx.save();
    ctx.translate(boardX + food.x * cell + cell / 2, boardY + food.y * cell + cell * 0.52);
    this.roundedRect(ctx, -appleSize * 0.5, -appleSize * 0.5, appleSize, appleSize, appleSize * 0.4);
    ctx.fillStyle = "#df2338";
    ctx.fill();
    ctx.strokeStyle = "rgba(122, 16, 30, 0.95)";
    ctx.lineWidth = Math.max(1, cell * 0.05);
    ctx.stroke();
    if (includeStem) {
      ctx.strokeStyle = "#6b4728";
      ctx.lineWidth = Math.max(1.1, cell * 0.05);
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(0, -appleSize * 0.46);
      ctx.quadraticCurveTo(appleSize * 0.03, -appleSize * 0.60, appleSize * 0.10, -appleSize * 0.72);
      ctx.stroke();
      ctx.lineCap = "butt";
      ctx.fillStyle = "#5ba93a";
      ctx.beginPath();
      ctx.ellipse(appleSize * 0.18, -appleSize * 0.54, appleSize * 0.16, appleSize * 0.08, -0.35, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();
  }

  drawSnakeBody(ctx, boardX, boardY, cell, snake, options = {}) {
    if (!Array.isArray(snake) || !snake.length) return null;
    const centerOf = (segment) => ({ x: boardX + segment.x * cell + cell / 2, y: boardY + segment.y * cell + cell / 2 });
    if (snake.length > 1) {
      ctx.strokeStyle = "#44c8a2";
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.lineWidth = Math.max(Number(options.minLineWidth ?? 6), cell * Number(options.lineRatio ?? 0.58));
      ctx.beginPath();
      snake.forEach((segment, index) => {
        const center = centerOf(segment);
        if (index === 0) ctx.moveTo(center.x, center.y);
        else ctx.lineTo(center.x, center.y);
      });
      ctx.stroke();
    }
    snake.slice().reverse().forEach((segment, reverseIndex) => {
      const index = snake.length - 1 - reverseIndex;
      const x = boardX + segment.x * cell + cell * 0.04;
      const y = boardY + segment.y * cell + cell * 0.04;
      const sizePx = cell * 0.92;
      this.roundedRect(ctx, x, y, sizePx, sizePx, Math.max(Number(options.minRadius ?? 8), cell * Number(options.radiusRatio ?? 0.4)));
      ctx.fillStyle = index === 0 ? "#d7fff4" : index === snake.length - 1 ? "#36b28b" : "#5cd6ad";
      ctx.fill();
    });
    return snake[0];
  }

  drawFoodTrail(ctx, boardX, boardY, cell, trail, options = {}) {
    if (!Array.isArray(trail) || !trail.length) return;
    const guard = options.guard ?? (() => true);
    trail.forEach((marker, index) => {
      if (!marker || !guard(marker)) return;
      const freshness = (index + 1) / Math.max(1, trail.length);
      const alpha = Number(options.baseAlpha ?? 0.16) + freshness * Number(options.alphaScale ?? 0.20);
      const sizePx = cell * (Number(options.baseSizeRatio ?? 0.24) + freshness * Number(options.sizeScale ?? 0.10));
      const offset = (cell - sizePx) / 2;
      ctx.save();
      this.roundedRect(ctx, boardX + marker.x * cell + offset, boardY + marker.y * cell + offset, sizePx, sizePx, Math.max(3, sizePx * 0.30));
      ctx.fillStyle = `rgba(255, 82, 108, ${alpha.toFixed(3)})`;
      ctx.fill();
      ctx.strokeStyle = `rgba(${options.strokeRgb ?? "255, 214, 220"}, ${(alpha * Number(options.strokeAlphaScale ?? 0.7)).toFixed(3)})`;
      ctx.lineWidth = Math.max(Number(options.minLineWidth ?? 1), cell * Number(options.lineRatio ?? 0.03));
      ctx.stroke();
      ctx.restore();
    });
  }

  drawSnakeHead(ctx, boardX, boardY, cell, head, options = {}) {
    if (!head) return null;
    const headX = boardX + head.x * cell + cell * 0.04;
    const headY = boardY + head.y * cell + cell * 0.04;
    const sizePx = cell * 0.92;
    this.roundedRect(ctx, headX, headY, sizePx, sizePx, Math.max(Number(options.minRadius ?? 8), cell * Number(options.radiusRatio ?? 0.42)));
    ctx.fillStyle = "#d7fff4";
    ctx.fill();
    ctx.strokeStyle = "rgba(9, 19, 27, 0.25)";
    ctx.lineWidth = 1;
    ctx.stroke();
    if (Number.isInteger(options.direction)) {
      const eyeRadius = Math.max(1.5, cell * 0.06);
      const eyeLayouts = [
        [[0.3, 0.32], [0.7, 0.32]],
        [[0.68, 0.3], [0.68, 0.7]],
        [[0.3, 0.68], [0.7, 0.68]],
        [[0.32, 0.3], [0.32, 0.7]]
      ];
      ctx.fillStyle = "#08131d";
      eyeLayouts[options.direction]?.forEach(([ex, ey]) => {
        ctx.beginPath();
        ctx.arc(headX + sizePx * ex, headY + sizePx * ey, eyeRadius, 0, Math.PI * 2);
        ctx.fill();
      });
    }
    return { headX, headY, sizePx };
  }

  drawSnakeScene(ctx, width, height, scene, options = {}) {
    const deathPreset = options.variant === "death" ? {
      boardPadding: 14,
      frameOptions: { glow: 5, outerLineWidth: 1.25, innerLineWidth: 1 },
      clearOccupiedCells: true,
      clearCellOptions: { fill: "rgba(13, 30, 41, 0.98)", insetRatio: 0, rounded: false },
      bodyOptions: { minLineWidth: 4, lineRatio: 0.54, minRadius: 6, radiusRatio: 0.38 },
      trailOptions: { baseAlpha: 0.24, alphaScale: 0.30, baseSizeRatio: 0.28, sizeScale: 0.14, strokeRgb: "255, 233, 237", strokeAlphaScale: 0.92, minLineWidth: 1.2, lineRatio: 0.04 },
      headOptions: { minRadius: 6, radiusRatio: 0.4 }
    } : {};
    const mergeSceneOption = (key) => (deathPreset[key] || options[key] ? { ...(deathPreset[key] ?? {}), ...(options[key] ?? {}) } : undefined);
    const sceneOptions = {
      ...deathPreset, ...options,
      frameOptions: mergeSceneOption("frameOptions"),
      foodCellOptions: mergeSceneOption("foodCellOptions"),
      clearCellOptions: mergeSceneOption("clearCellOptions"),
      appleOptions: mergeSceneOption("appleOptions"),
      bodyOptions: mergeSceneOption("bodyOptions"),
      trailOptions: mergeSceneOption("trailOptions"),
      headOptions: mergeSceneOption("headOptions")
    };
    const size = Number(scene.size ?? 0) || 1;
    const snake = Array.isArray(scene.snake) ? scene.snake : [];
    const food = scene.food;
    const trail = Array.isArray(scene.eatenFoodTrail) ? scene.eatenFoodTrail : [];
    const boardPadding = Number(sceneOptions.boardPadding ?? 18);
    const boardSize = Math.min(width, height) - boardPadding * 2;
    const boardX = Math.round((width - boardSize) / 2);
    const boardY = Math.round((height - boardSize) / 2);
    const cell = boardSize / size;
    this.drawBoardFrame(ctx, boardX, boardY, boardSize, size, sceneOptions.frameOptions ?? {});
    if (food) this.fillBoardCell(ctx, boardX, boardY, cell, food.x, food.y, sceneOptions.foodCellOptions ?? { insetRatio: 0.08, radiusRatio: 0.24 });
    if (sceneOptions.clearOccupiedCells) {
      const clearOptions = sceneOptions.clearCellOptions ?? { fill: "rgba(13, 30, 41, 0.98)", insetRatio: 0, rounded: false };
      if (food) this.fillBoardCell(ctx, boardX, boardY, cell, food.x, food.y, clearOptions);
      snake.forEach((segment) => this.fillBoardCell(ctx, boardX, boardY, cell, segment.x, segment.y, clearOptions));
    } else {
      snake.forEach((segment, index) => {
        const insetRatio = index === 0 ? 0.04 : 0.05;
        const radiusRatio = index === 0 ? 0.26 : 0.24;
        this.fillBoardCell(ctx, boardX, boardY, cell, segment.x, segment.y, { insetRatio, radiusRatio });
      });
    }
    this.drawApple(ctx, boardX, boardY, cell, food, sceneOptions.appleOptions ?? {});
    this.drawSnakeBody(ctx, boardX, boardY, cell, snake, sceneOptions.bodyOptions ?? {});
    if (sceneOptions.showTrail !== false && trail.length) {
      this.drawFoodTrail(ctx, boardX, boardY, cell, trail, sceneOptions.trailOptions ?? {});
    }
    const head = snake[0];
    if (head) this.drawSnakeHead(ctx, boardX, boardY, cell, head, sceneOptions.headOptions ?? {});
    return { boardX, boardY, boardSize, cell, snake, head, food };
  }

  getObservationWindowOrigin(head, boardCells, inputGrid = this.currentInputGrid()) {
    if (boardCells >= inputGrid) {
      const halfWindow = Math.floor(inputGrid / 2);
      return { x: clamp(head.x - halfWindow, 0, boardCells - inputGrid), y: clamp(head.y - halfWindow, 0, boardCells - inputGrid) };
    }
    return { x: 0, y: 0 };
  }

  bindEvents() {
    window.addEventListener("keydown", (event) => this.handleKeydown(event));
    window.addEventListener("resize", () => {
      if (this.resizeFrameHandle !== null) window.cancelAnimationFrame(this.resizeFrameHandle);
      this.resizeFrameHandle = window.requestAnimationFrame(() => {
        this.resizeFrameHandle = null;
        this.lastGameRenderSignature = "";
        this.safeRenderAll();
      });
    });
    document.addEventListener("visibilitychange", () => {
      if (document.hidden) return;
      this.forceNextVisualRefresh();
      this.refreshState({ forceDetails: true, forceNetwork: this.shouldUseNetworkVisuals() });
      this.safeRenderAll();
    });
    this.inputModeSelect?.addEventListener("change", () => { this.sendCommand("set_input_mode", this.inputModeSelect.value); });
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
    this.stepReviewLiveButton?.addEventListener("click", () => { this.selectedStepLogKey = "live"; this.lastStepReviewSignature = ""; this.renderStepReview({ force: true }); });
    this.stepReviewPreviousButton?.addEventListener("click", () => {
      if (this.stepReviewPreviousButton.disabled) return;
      this.selectedStepLogKey = "previous"; this.lastStepReviewSignature = ""; this.renderStepReview({ force: true });
    });
    this.runToggle?.addEventListener("click", () => this.sendCommand("toggle_running"));
    this.manualModeButton?.addEventListener("click", () => { if (this.state?.mode === "smart") this.smartContextActive = true; this.sendCommand("set_mode", "manual"); });
    this.aiModeButton?.addEventListener("click", () => { this.sendCommand("set_mode", this.smartContextActive ? "smart" : "ai"); });
    this.trainToggle?.addEventListener("click", () => { if (this.trainToggle.disabled) return; this.sendCommand("toggle_training", !this.state?.training); });
    this.resetAgentButton?.addEventListener("click", () => { if (this.resetAgentButton.disabled) return; this.sendCommand("reset_agent"); });
    this.restartBackendButton?.addEventListener("click", () => this.restartBackend());
    this.shutdownBackendButton?.addEventListener("click", () => this.shutdownBackend());
    this.deathStepTimeline?.addEventListener("click", (event) => {
      const button = event.target instanceof Element ? event.target.closest("button[data-death-step]") : null;
      if (!button) return;
      const nextIndex = Number(button.getAttribute("data-death-step"));
      if (Number.isFinite(nextIndex)) { this.selectedDeathStepIndex = nextIndex; this.renderLastDeathSnapshot(); }
    });
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
        this.pendingControlSync = true;
        this.stateDetailsFresh = false;
        if (forcePanelRefresh) this.forceNextVisualRefresh();
        if (type === "toggle_running" && this.state.running) this.forceStepLogScroll = true;
        this.safeRenderAll();
        if (forcePanelRefresh) window.setTimeout(() => this.refreshState({ forceDetails: true, forceNetwork: this.shouldUseNetworkVisuals() }), 80);
        else if (type === "set_smart_model") window.setTimeout(() => this.refreshState({ forceDetails: true }), 60);
      } else if (type !== "manual_action") {
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
    this.lastNetworkPayloadAt = 0; this.lastDetailPayloadAt = 0; this.lastRenderedSnapshotKey = ""; this.lastDeathTimelineMarkup = ""; this.lastStepReviewSignature = "";
    this.lastGameRenderSignature = ""; this.lastRewardRulesMarkup = ""; this.lastDecisionStatsMarkup = ""; this.lastSummaryMarkup = ""; this.lastSmartModelMarkup = "";
    this.pendingHeavyTasks = []; this.pendingHeavyTaskNames.clear();
  }

  mergeDeferredState(nextState, previousState) {
    if (!previousState) return nextState;
    const carryDeferredFields = () => {
      DEFERRED_STATE_FIELDS.forEach((key) => { if (!(key in nextState) && key in previousState) nextState[key] = previousState[key]; });
      if (previousState.network && !nextState.network) nextState.network = previousState.network;
    };
    const stateShapeChanged = previousState.input_grid !== nextState.input_grid || previousState.agent_kind !== nextState.agent_kind || previousState.mode !== nextState.mode || previousState.game?.size !== nextState.game?.size;
    if (stateShapeChanged) { carryDeferredFields(); return nextState; }
    const previousFrameId = Number(previousState.frame_id ?? -1);
    const nextFrameId = Number(nextState.frame_id ?? previousFrameId);
    const staleFrame = Number.isFinite(previousFrameId) && Number.isFinite(nextFrameId) && nextFrameId < previousFrameId;
    if (nextState.backend_busy || staleFrame) STALE_FRAME_FIELDS.forEach((key) => { if (key in previousState) nextState[key] = previousState[key]; });
    carryDeferredFields();
    if (previousState.network && !nextState.network) nextState.network = previousState.network;
    else if (nextState.network && previousState.network) {
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
    if (this.pollHandle) { window.clearTimeout(this.pollHandle); this.pollHandle = null; }
    if (this.framePollHandle) { window.clearTimeout(this.framePollHandle); this.framePollHandle = null; }
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
    if (this.isFetchingFrame || !this.state || document.hidden) return;
    this.isFetchingFrame = true;
    try {
      const frameState = await apiJson("/api/frame", { timeoutMs: 700 });
      const previousFrameId = Number(this.state?.frame_id ?? -1);
      const nextFrameId = Number(frameState?.frame_id ?? previousFrameId);
      if (Number.isFinite(previousFrameId) && Number.isFinite(nextFrameId) && nextFrameId === previousFrameId) return;
      this.state = this.mergeFrameState(frameState, this.state);
      this.stateDetailsFresh = false;
      this.scheduleLiveFrameRender();
    } catch (error) { if (Date.now() >= this.restartPendingUntil) console.warn("Snake frame error:", error); } finally { this.isFetchingFrame = false; }
  }

  handleKeydown(event) {
    const key = event.key.toLowerCase();
    if (key in KEY_DIRECTIONS) { event.preventDefault(); this.sendCommand("manual_action", KEY_DIRECTIONS[key]); return; }
    if (event.code === "Space") { event.preventDefault(); this.sendCommand("toggle_running"); }
  }

  scheduleRender() {
    if (this.renderFrameHandle) return;
    this.renderFrameHandle = window.requestAnimationFrame(() => {
      this.renderFrameHandle = null;
      try { this.renderAll(); } catch (error) { console.error("Snake render error:", error); this.setDecisionText(`Ошибка интерфейса: ${String(error.message || error)}`); }
    });
  }

  scheduleLiveFrameRender() {
    if (this.liveFrameRenderHandle) return;
    this.liveFrameRenderHandle = window.requestAnimationFrame(() => {
      this.liveFrameRenderHandle = null;
      try {
        const now = performance.now();
        if (now - this.lastPanelRenderAt >= UI_PANEL_RENDER_MS) { this.updateStats(); this.lastPanelRenderAt = now; }
        this.renderGame();
      } catch (error) { console.error("Snake live frame render error:", error); this.setDecisionText(`Ошибка интерфейса: ${String(error.message || error)}`); }
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

  renderAll() {
    if (!this.state) return;
    const now = performance.now();
    const shouldRenderPanels = now - this.lastPanelRenderAt >= UI_PANEL_RENDER_MS;
    if (shouldRenderPanels || this.pendingControlSync) { this.syncControlsFromState(); this.updateHeroBadges(); this.updateStats(); this.lastPanelRenderAt = now; this.pendingControlSync = false; }
    this.renderGame();
    const snapshot = this.state?.last_death_snapshot;
    const snapshotKey = snapshot ? `${snapshot.step_count ?? "-"}:${snapshot.score ?? "-"}:${snapshot.reason ?? "-"}` : "";
    if (snapshotKey !== this.lastRenderedSnapshotKey || now - this.lastSnapshotRenderAt >= SNAPSHOT_RENDER_MS) { this.renderLastDeathSnapshot(); this.lastRenderedSnapshotKey = snapshotKey; this.lastSnapshotRenderAt = now; }
    if (this.showNetwork) {
      const shouldRenderNetworkVisuals = this.shouldUseNetworkVisuals();
      const shouldRenderNetwork = now - this.lastNetworkRenderAt >= NETWORK_RENDER_MS;
      const shouldRenderChart = now - this.lastChartRenderAt >= CHART_RENDER_MS;
      if (shouldRenderNetworkVisuals && shouldRenderNetwork) {
        this.lastNetworkRenderAt = now;
        this.queueHeavyRender("networkGraph", () => { if (this.shouldUseNetworkVisuals() && this.shouldShowNetworkGraph()) this.renderNetworkGraph(); });
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
    this.resetFramePolling();
    if (document.hidden) return;
    this.updateVisualizationVisibility();
    this.scheduleRender();
  }

  updateVisualizationVisibility() {
    const showVisuals = this.shouldUseNetworkVisuals();
    this.setHidden(this.networkPanel, false);
    this.setHidden(this.networkGraphCard, !showVisuals || !this.shouldShowNetworkGraph());
  }

  syncControlsFromState() {
    if (this.state.mode === "smart") this.smartContextActive = true;
    else if (this.state.mode === "ai") this.smartContextActive = false;
    if (this.inputModeSelect) {
      const nextInputMode = this.state?.input_mode ?? "signals";
      if (this.inputModeSelect.value !== nextInputMode) this.inputModeSelect.value = nextInputMode;
    }
    if (this.agentModeSelect) {
      if (!this.agentModeFocused) {
        const nextAgentMode = this.smartContextActive ? "smart" : this.state.agent_kind ?? "evolution";
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
    const isSmartMode = this.smartContextActive;
    this.setText(this.runToggle, this.state.running ? "Пауза" : "Старт");
    this.setText(this.trainToggle, this.state.training ? "Обучение: вкл" : "Обучение: выкл");
    if (this.trainToggle) { this.trainToggle.disabled = isSmartMode; this.trainToggle.title = isSmartMode ? "Zmey-smart не обучается в этом режиме" : ""; }
    if (this.resetAgentButton) { this.resetAgentButton.disabled = isSmartMode; this.resetAgentButton.title = isSmartMode ? "Сброс агента доступен только для DQN и эволюции" : ""; }
    this.toggleClass(this.manualModeButton, "is-active", this.state.mode === "manual" && !this.state.training);
    this.toggleClass(this.aiModeButton, "is-active", ["ai", "smart"].includes(this.state.mode) && !this.state.training);
    this.toggleClass(this.trainToggle, "is-active", this.state.training);
    this.updateVisualizationVisibility();
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
    const mode = this.inputMode();
    const signalCount = (this.state?.global_signals?.labels ?? []).length;
    const grid = this.currentInputGrid();
    const inputLabel = mode === "signals" ? `${signalCount} сигналов` : mode === "map" ? `Карта ${grid}x${grid}` : `Карта ${grid}x${grid} + ${signalCount} сигналов`;
    const isSmartContext = this.smartContextActive || this.state.mode === "smart";
    this.stableHeroProfile = { device: this.state.device_kind ?? "CPU", boardSize: this.state?.game?.size ?? 25, inputLabel, networkSource: isSmartContext ? "Zmey-smart" : this.state.network_source ?? "Evolution policy" };
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

  buildDecisionText() {
    if (this.state?.restart_pending || Date.now() < this.restartPendingUntil) return ["Backend перезапускается.", "Состояние сохранено.", "Связь может пропасть на несколько секунд."].join("\n");
    const qValues = this.formatOrderedQValues();
    const penaltyReasons = Array.isArray(this.state.last_penalty_reasons) ? this.state.last_penalty_reasons.map((reason) => this.penaltyShortLabel(reason)).join(", ") : "";
    const networkChoice = Number.isInteger(this.state.last_greedy_action) ? `${this.arrowForAction(this.state.last_greedy_action)} ${this.outputLabels()[this.state.last_greedy_action] ?? ""}`.trim() : "-";
    const reward = Number.isFinite(this.state.stats?.reward) ? this.state.stats.reward : 0;
    const rewardValue = reward.toFixed(3);
    let rewardReason = "Нет";
    if (this.state.last_reason === "victory") rewardReason = "Победа";
    else if (this.state.last_reason === "apple") rewardReason = "Яблоко";
    else if (this.state.last_reason === "looping") rewardReason = "Цикл";
    else if (TERMINAL_REASONS.has(this.state.last_reason)) rewardReason = "Смерть";
    else if (reward > 0.02) rewardReason = "Положительный ход";
    else if (reward > -0.02 && !penaltyReasons) rewardReason = "Базовый шаг";
    let actionHeader = "Q-values (↑ / ← / →): ";
    const mode = this.state.mode;
    const agentKind = this.state.agent_kind;
    if (mode === "smart") actionHeader = "Оценки действий (↑ / ← / →): ";
    else if (agentKind === "evolution") actionHeader = "Оценки действий (↑ / ← / →): ";
    else if (agentKind === "dqn") actionHeader = "Q-values (↑ / ← / →): ";
    const lines = [`${actionHeader}${qValues || "-"}`, `Выбор сети: ${networkChoice}`, `Награда: ${rewardReason} (${rewardValue})`, `Штрафы: ${penaltyReasons || "-"}`];
    return lines.join("\n");
  }

  penaltyCatalogById() {
    const rules = this.state?.reward_rules ?? {};
    const penalties = Array.isArray(rules.penalties) ? rules.penalties : [];
    return new Map(penalties.map((item) => [String(item.id ?? ""), item]));
  }

  penaltyShortLabel(reason) {
    const item = this.penaltyCatalogById().get(String(reason));
    return String(item?.short_label ?? item?.label ?? formatReason(reason));
  }

  penaltyEffectTag(reason) {
    const item = this.penaltyCatalogById().get(String(reason));
    return String(item?.short_label ?? item?.label ?? formatReason(reason)).toLowerCase();
  }

  getDeathSteps(snapshot) {
    const recentSteps = Array.isArray(snapshot?.recent_steps) ? snapshot.recent_steps : [];
    return recentSteps.length ? recentSteps : snapshot ? [snapshot] : [];
  }

  renderDeathRiskTable(selectedStep) {
    if (!this.deathRiskTable) return;
    const safety = selectedStep?.move_safety ?? null;
    const risks = selectedStep?.move_risks ?? null;
    if (!safety && !risks) { this.setHtml(this.deathRiskTable, ""); return; }
    const baseDirection = Number(selectedStep?.direction ?? 1);
    const selectedDirection = Number(selectedStep?.planned_direction ?? selectedStep?.direction ?? 1);
    const percent = (value) => `${Math.round(Number(value ?? 0) * 100)}%`;
    const absoluteForRelativeAction = (action) => { if (action === 1) return (baseDirection + 3) % 4; if (action === 2) return (baseDirection + 1) % 4; return baseDirection; };
    const rows = [{ action: 0, key: "forward" }, { action: 1, key: "left" }, { action: 2, key: "right" }].map(({ action, key }) => {
      const absoluteDirection = absoluteForRelativeAction(action);
      const label = this.arrowForAbsoluteDirection(absoluteDirection);
      const metrics = safety?.[key] ?? {};
      const risk = Number(metrics.risk ?? risks?.[key] ?? 0);
      const rowClass = absoluteDirection === selectedDirection ? ' class="is-selected"' : "";
      return [`<tr${rowClass}>`, `<td>${label}</td>`, `<td>${percent(risk)}</td>`, `<td>${percent(metrics.food)}</td>`, `<td>${percent(metrics.tail)}</td>`, `<td>${percent(metrics.space)}</td>`, `<td>${percent(metrics.safe)}</td>`, "</tr>"].join("");
    }).join("");
    this.setHtml(this.deathRiskTable, `<table class="death-risk-table"><thead><tr><th>Ход</th><th>Риск</th><th>Еда</th><th>Хвост</th><th>Место</th><th>После ябл.</th></tr></thead><tbody>${rows}</tbody></table>`);
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

  renderDecisionStats() {
    if (!this.decisionStatsTable) return;
    const stats = this.state.decision_stats ?? {};
    const bucket = stats.deaths ?? {};
    const allCounts = bucket.all ?? {};
    const recentList = Array.isArray(bucket.recent) ? bucket.recent : [];
    const recentCounts = {};
    recentList.forEach((item) => { recentCounts[item] = (recentCounts[item] ?? 0) + 1; });
    const outcomeOrder = ["victory", "wall_collision", "body_collision", "starvation", "starvation_food_unreachable", "starvation_no_relayout_path", "starvation_food_available", "starvation_safe_loop", "starvation_endgame_trap"];
    const keys = [...Object.keys(allCounts)].filter((key) => Number(allCounts[key] ?? 0) > 0).sort((a, b) => { const left = outcomeOrder.includes(a) ? outcomeOrder.indexOf(a) : outcomeOrder.length; const right = outcomeOrder.includes(b) ? outcomeOrder.indexOf(b) : outcomeOrder.length; return left - right || a.localeCompare(b); });
    const allTotal = Object.values(allCounts).reduce((sum, value) => sum + Number(value || 0), 0);
    const recentTotal = recentList.length;
    if (!keys.length) {
      const emptyMarkup = `<table><thead><tr><th>Итог</th><th>Все</th><th>100 посл.</th></tr></thead><tbody><tr><td colspan="3" class="muted">Еще нет данных о завершенных эпизодах.</td></tr></tbody></table>`;
      if (emptyMarkup !== this.lastDecisionStatsMarkup) { this.lastDecisionStatsMarkup = emptyMarkup; this.setHtml(this.decisionStatsTable, emptyMarkup); }
      return;
    }
    const rows = keys.map((key) => {
      const allCount = Number(allCounts[key] ?? 0);
      const recentCount = Number(recentCounts[key] ?? 0);
      return ["<tr>", `<td>${formatReason(key)}</td>`, '<td><span class="stat-value">', `<span>${this.formatStatPercent(allCount, allTotal)}</span>`, `<span class="muted">(${allCount})</span>`, "</span></td>", '<td><span class="stat-value">', `<span>${this.formatStatPercent(recentCount, recentTotal)}</span>`, "</span></td>", "</tr>"].join("");
    });
    const markup = `<table><thead><tr><th>Итог</th><th>Все</th><th>100 посл.</th></tr></thead><tbody>${rows.join("")}</tbody></table>`;
    if (markup !== this.lastDecisionStatsMarkup) { this.lastDecisionStatsMarkup = markup; this.setHtml(this.decisionStatsTable, markup); }
  }

  renderLastDeathSnapshot() {
    if (!this.deathSnapshotCtx || !this.deathSnapshotCanvas) return;
    const snapshot = this.state?.last_death_snapshot;
    const ctx = this.deathSnapshotCtx;
    const { width, height } = this.resizeCanvas(this.deathSnapshotCanvas, ctx);
    const snapshotKey = snapshot ? `${snapshot.step_count ?? "-"}:${snapshot.score ?? "-"}:${snapshot.reason ?? "-"}` : "";
    if (snapshotKey !== this.lastDeathSnapshotKey) { this.lastDeathSnapshotKey = snapshotKey; this.selectedDeathStepIndex = -1; }
    const { selectedStep, selectedIndex } = this.getSelectedDeathStep(snapshot);
    this.renderDeathStepTimeline(snapshot);
    this.renderDeathRiskTable(selectedStep);
    const stepOffset = selectedIndex - this.getDeathSteps(snapshot).length;
    const outcomeReason = selectedStep?.outcome_reason ?? selectedStep?.reason ?? snapshot?.reason;
    const outcomeTarget = outcomeReason === "victory" ? "победы" : "исхода";
    const stepLabel = `Ход ${stepOffset} до ${outcomeTarget}`;
    ctx.clearRect(0, 0, width, height);
    const background = ctx.createLinearGradient(0, 0, width, height);
    background.addColorStop(0, "#09131b");
    background.addColorStop(1, "#143244");
    ctx.fillStyle = background;
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
      this.setText(this.deathSnapshotMeta, "Снимок появится после первого эпизода.");
      return;
    }
    const size = Number(selectedStep.size ?? 0) || 1;
    const { boardX, boardY, boardSize, cell, snake, head } = this.drawSnakeScene(ctx, width, height, { size, snake: selectedStep.snake ?? [], food: selectedStep.food, eatenFoodTrail: selectedStep.eaten_food_trail ?? [] }, { variant: "death", headOptions: { direction: selectedStep.direction } });
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
    const actionIndex = Number.isInteger(selectedStep.action) ? selectedStep.action : null;
    const greedyIndex = Number.isInteger(selectedStep.greedy_action) ? selectedStep.greedy_action : null;
    const actionArrow = actionIndex === null ? "-" : this.arrowForAction(actionIndex);
    const greedyArrow = greedyIndex === null ? "-" : this.arrowForAction(greedyIndex);
    const choiceLine = actionIndex === null ? null : selectedStep.action_random ? `Выбор: случайный ${actionArrow}; сеть выбрала бы ${greedyArrow}` : `Выбранный ход сети: ${actionArrow}`;
    this.setText(this.deathChoiceMeta, choiceLine ?? "");
    this.setHidden(this.deathChoiceMeta, !choiceLine);

    // === ИЗМЕНЕНИЕ: Формат голода и фикс 0 шагов ===
    const stepsWithoutFood = selectedStep.steps_without_food ?? selectedStep.last_steps_without_food ?? 0;
    const hungerInfo = (typeof outcomeReason === "string" && outcomeReason.startsWith("starvation")) ? `• ${stepsWithoutFood} шагов` : "";
    this.setText(this.deathSnapshotMeta, [`${stepLabel}`, `Итог: ${formatReason(outcomeReason)}${hungerInfo}`, `Счет: ${selectedStep.score ?? 0} • Шагов: ${selectedStep.step_count ?? 0} • Длина: ${snake.length}`].filter(Boolean).join("\n"));
  }

  renderRewardRules() {
    if (!this.rewardRulesBody) return;
    const rules = this.state?.reward_rules ?? {};
    const bonuses = Array.isArray(rules.bonuses) ? rules.bonuses : [];
    const penalties = Array.isArray(rules.penalties) ? rules.penalties : [];
    const renderItems = (items) => items.map((item) => { const title = item.formula ? `title="${escapeText(item.formula)}"` : ""; return `<span${title}>${escapeText(item.label ?? "-")} <code>${escapeText(item.value ?? "")}</code></span>`; }).join("");
    const sections = [["Бонусы", bonuses], ["Штрафы", penalties]].filter(([, items]) => items.length > 0);
    const markup = sections.map(([title, items]) => `<tr><th>${title}</th></tr><tr><td><div class="reward-list">${renderItems(items)}</div></td></tr>`).join("");
    if (markup !== this.lastRewardRulesMarkup) { this.lastRewardRulesMarkup = markup; this.setHtml(this.rewardRulesBody, markup); }
  }

  renderHungerStat() {
    if (this.epsilonStat?.previousElementSibling) this.epsilonStat.previousElementSibling.textContent = "";
    const steps = Math.max(0, Number(this.state.stats?.hunger_steps ?? 0));
    const budget = Math.max(1, Number(this.state.stats?.hunger_budget ?? 1));
    const ratio = clamp(Number(this.state.stats?.hunger_ratio ?? steps / budget), 0, 1);
    const percentValue = Math.round(ratio * 100);
    const className = ratio >= 0.9 ? "hunger-meter is-danger" : ratio >= 0.65 ? "hunger-meter is-warn" : "hunger-meter";
    const text = `${Math.round(steps)}/${Math.round(budget)}`;
    this.setHtml(this.epsilonStat, `<span class="${className}" title="Шагов без яблока: ${Math.round(steps)} из ${Math.round(budget)}"><span class="hunger-meter__text"><span class="hunger-meter__label">Голод</span><span class="hunger-meter__value">${escapeText(text)}</span></span><span class="hunger-meter__track"><span class="hunger-meter__fill" style="--hunger-width: ${percentValue}%"></span></span></span>`);
  }

  updateStats() {
    const modeLabels = { manual: "Игрок", ai: "ИИ", smart: "Zmey-smart" };
    const isSmart = this.state.mode === "smart";
    const isDqn = this.state.agent_kind === "dqn" && !isSmart;
    const modeText = this.state.training ? "Обучение" : modeLabels[this.state.mode] ?? "ИИ";
    const smartModelName = this.state.smart_model?.label || this.state.smart_model?.name || "-";
    const smartModelShort = String(smartModelName).length > 14 ? `${String(smartModelName).slice(0, 13)}...` : String(smartModelName);
    this.setText(this.modeStat, modeText);
    this.setText(this.epochStat, String(this.state.stats.episode));
    this.setText(this.scoreStat, String(this.state.game.score));
    this.setText(this.bestScoreStat, String(this.state.stats.best_score));
    this.setText(this.avgScoreStat, Number(this.state.stats.average_score_100 ?? 0).toFixed(1));
    if (this.avgScoreStat) { const averageWindow = Number(this.state.stats.average_score_window ?? 0); this.avgScoreStat.title = averageWindow ? `Средний счет по ${averageWindow} эпизодам` : ""; }
    this.setText(this.lossStat?.previousElementSibling, isDqn ? "Epsilon" : isSmart ? "Модель" : "Fitness");
    this.setText(this.epsilonStat?.previousElementSibling, "");
    this.setText(this.bufferStat?.previousElementSibling, isDqn ? "Replay" : isSmart ? "Эпизоды" : "Поколение");
    this.setText(this.leaderStat?.previousElementSibling, isDqn ? "Updates" : isSmart ? "Режим" : "Лидер");
    this.setText(this.aliveStat?.previousElementSibling, isDqn ? "Статус" : isSmart ? "Статус" : "Живых");
    if (isSmart) { this.setText(this.lossStat, this.state.smart_model?.loaded ? "loaded" : "нет"); this.renderHungerStat(); this.setText(this.bufferStat, String(this.state.stats.episode)); this.setText(this.leaderStat, this.state.running ? "run" : "idle"); this.setText(this.aliveStat, this.state.smart_model?.loaded ? "ok" : "нет"); }
    else if (isDqn) { this.setText(this.lossStat, Number(this.state.stats.epsilon ?? 0).toFixed(3)); this.renderHungerStat(); this.setText(this.bufferStat, String(this.state.stats.buffer ?? 0)); this.setText(this.leaderStat, String(this.state.stats.updates ?? 0)); this.setText(this.aliveStat, this.state.training ? "train" : "idle"); }
    else { this.setText(this.lossStat, Number(this.state.stats.fitness ?? 0).toFixed(3)); this.renderHungerStat(); this.setText(this.bufferStat, String(this.state.stats.generation ?? 0)); this.setText(this.leaderStat, this.state.training ? `${this.state.stats.leader ?? 0}/${this.state.stats.population ?? 0}` : "best"); this.setText(this.aliveStat, this.state.training ? `${this.state.stats.alive ?? 0}/${this.state.stats.population ?? 0}` : "-"); }
    this.setText(this.chartMetricLabel, isDqn ? "Loss" : isSmart ? "Счет" : "Fitness");
    this.setText(this.chartMetricDescription, isSmart ? "История Zmey-smart по выбранной модели: счет, награда и средний счет за последние эпизоды." : isDqn ? "Ошибка DQN при обновлении Q-сети: чем ниже значение, тем стабильнее прогнозы." : "Оценка особи для селекции: учитывает счет, выживание, награду, ловушки и причину завершения.");
    this.setText(this.stepStat, String(this.state.stats.steps));
    this.setDecisionText(this.buildDecisionText());
    this.renderDecisionStats();
    this.renderRewardRules();
    this.renderStepReview({ force: false });
  }

  renderStepReview({ force = false } = {}) {
    if (!this.stepReviewLabel || !this.stepReviewBody) return;
    const history = Array.isArray(this.state?.step_review_history) ? this.state.step_review_history : [];
    const previousLog = history[0] ?? null;
    const previousId = String(previousLog?.id ?? "");
    if (!previousLog && this.selectedStepLogKey === "previous") this.selectedStepLogKey = "live";
    if (previousId !== this.lastStepReviewHistoryId) { this.lastStepReviewHistoryId = previousId; if (!previousLog && this.selectedStepLogKey === "previous") this.selectedStepLogKey = "live"; }
    this.toggleClass(this.stepReviewLiveButton, "is-active", this.selectedStepLogKey === "live");
    this.toggleClass(this.stepReviewPreviousButton, "is-active", this.selectedStepLogKey === "previous");
    if (this.stepReviewPreviousButton) { this.stepReviewPreviousButton.disabled = !previousLog; this.stepReviewPreviousButton.title = previousLog ? `Эпизод ${previousLog.episode ?? "?"}, счет ${previousLog.score ?? 0}, ${formatReason(previousLog.reason)}` : "Прошлого лога пока нет"; }
    const selectedHistory = this.selectedStepLogKey === "previous" ? previousLog : null;
    const steps = selectedHistory ? Array.isArray(selectedHistory.entries) ? selectedHistory.entries : [] : Array.isArray(this.state?.step_review) ? this.state.step_review : [];
    if (!steps.length) { this.selectedReviewStepIndex = -1; this.setText(this.stepReviewLabel, "Нет шагов"); this.setHtml(this.stepReviewBody, '<div class="muted">После первого шага здесь появятся награда и штрафы по ходу эпизода.</div>'); this.lastStepReviewSignature = ""; this.lastStepReviewReason = ""; return; }
    const visibleSteps = steps.slice(-100);
    const lastStep = steps[steps.length - 1];
    const stepReviewSource = selectedHistory ? "history" : this.state?.step_review_source ?? "current";
    const isDeathLog = stepReviewSource === "last_death";
    const sourceLabel = selectedHistory ? `• эп. ${selectedHistory.episode ?? "?"}` : isDeathLog ? `• ${this.state?.last_outcome_reason === "victory" ? "последняя победа" : "итог эпизода"}` : "";
    const signature = [this.selectedStepLogKey, stepReviewSource, this.state?.last_outcome_reason ?? "", steps.length, lastStep.step, lastStep.reward, lastStep.train_reward, lastStep.reason, (lastStep.penalties ?? []).join(", "), JSON.stringify(lastStep.apple_opportunity ?? null)].join(":");
    const shouldUpdateDuringTraining = selectedHistory || !this.state?.training || !this.state?.running || stepReviewSource === "current" || isDeathLog || TERMINAL_REASONS.has(lastStep.reason);
    if (!force && this.state?.training && !shouldUpdateDuringTraining && this.lastStepReviewSignature) { this.setText(this.stepReviewLabel, `${visibleSteps.length} строк • последний шаг ${lastStep.step} • обновление на паузе/смерти`); return; }
    if (!force && signature === this.lastStepReviewSignature && lastStep.reason === this.lastStepReviewReason) return;
    const stepReasonLabel = (reason) => { if (reason === "apple") return "яблоко"; if (reason === "victory") return "win"; if (reason === "wall_collision") return "стена"; if (reason === "body_collision") return "тело"; if (typeof reason === "string" && reason.startsWith("starvation")) return "голод"; return "  "; };
    const forceScroll = this.forceStepLogScroll;
    const nearBottom = this.stepReviewBody.scrollHeight - this.stepReviewBody.scrollTop - this.stepReviewBody.clientHeight < 24;
    this.setText(this.stepReviewLabel, `${visibleSteps.length} строк • последний шаг ${steps[steps.length - 1].step}${sourceLabel}`);

	this.setHtml(this.stepReviewBody, visibleSteps.map((entry) => {
		const sourceLabel = entry.random ? "random" : "q";
		const reward = Number(entry.reward ?? 0);
		const rewardText = reward.toFixed(3);
		const moveArrow = this.arrowForAction(Number(entry.action ?? 0));
		const stepStr = `#${String(entry.step).padStart(4, "0")}`;

		// 1. Собираем теги штрафов и причин
		const penalties = Array.isArray(entry.penalties) ? entry.penalties : [];
		const effectTags = penalties.map((reason) => this.penaltyEffectTag(reason)).filter(Boolean);
		const reasonTag = stepReasonLabel(entry.reason);
		if (reasonTag && reasonTag.trim() && !effectTags.includes(reasonTag)) {
			effectTags.unshift(reasonTag);
		}

		// 2. Формируем заметку о яблоке
		const apple = entry.apple_opportunity;
		if (apple?.adjacent) {
			const appleDir = this.arrowForAction(Number(apple.action ?? 0));
			const verdict = apple.safe ? "безопасно" : formatAppleVerdictTag(apple.verdict ?? "none");
			const appleTag = `🍎 ${appleDir} ${verdict}`.trim();
			if (!effectTags.some(t => t.includes("🍎"))) {
				effectTags.push(appleTag);
			}
		}

		// 3. Очищаем от пустых значений и дефисов
		const cleanEvents = effectTags
			.map(t => String(t).trim())
			.filter(t => t && t !== "-" && t !== "  ");

		// 4. Собираем строку по шаблону: #0135 4.000 ↑ яблоко • 🍎 ↑ мало места • q
		const headerParts = [stepStr, rewardText, moveArrow];
		const extraParts = [...cleanEvents, sourceLabel];
		const logText = headerParts.join(" ") + (extraParts.length ? " • " + extraParts.join(" • ") : "");

		// 5. ПРИОРИТЕТ ЦВЕТОВ (строгий порядок: красный > зелёный > синий > жёлтый)
		const hasTailLost = penalties.includes("route_dead_end") || penalties.includes("tail_path_lost") || apple?.verdict === "no_tail_after_apple";
		const isCollision = penalties.includes("wall_collision") || penalties.includes("body_collision");
		const tookApple = entry.reason === "apple";

		let rowClass = "step-log-line log-yellow"; // Q по умолчанию
		if (hasTailLost || isCollision || reward < -1.0) {
			rowClass = "step-log-line log-red";    // ← Хвост/столкновение/штраф всегда красные, даже если взято яблоко
		} else if (tookApple || reward > 1.0) {
			rowClass = "step-log-line log-green";  // ← Яблоко зелёное, только если нет красных условий
		} else if (entry.random) {
			rowClass = "step-log-line log-blue";   // ← Random синий на нейтральных шагах
		}

		return `<div class="${rowClass}">${logText}</div>`;
	}).join(""));

    this.lastStepReviewSignature = signature;
    this.lastStepReviewReason = lastStep.reason;
    if (nearBottom || force || forceScroll) this.stepReviewBody.scrollTop = this.stepReviewBody.scrollHeight;
    this.forceStepLogScroll = false;
  }

  renderGame() {
    if (!this.gameCtx || !this.gameCanvas) return;
    const ctx = this.gameCtx;
    const size = this.state.game.size;
    const { width, height, resized } = this.resizeCanvas(this.gameCanvas, ctx);
    const snake = this.state.game.snake ?? [];
    const food = this.state.game.food;
    const eatenFoodTrail = this.state.game.eaten_food_trail ?? [];
    const frameId = this.state.frame_id ?? `${this.state.game.score}:${this.state.game.step_count}:${this.state.game.direction}`;
    const gameSignature = [width, height, size, frameId, this.state.training ? 1 : 0, this.currentInputGrid()].join("|");
    if (!resized && gameSignature === this.lastGameRenderSignature) return;
    this.lastGameRenderSignature = gameSignature;
    ctx.clearRect(0, 0, width, height);
    const backgroundKey = `${width}x${height}`;
    if (!this.gameBackgroundCache || this.gameBackgroundCache.key !== backgroundKey) { const background = ctx.createLinearGradient(0, 0, width, height); background.addColorStop(0, "#09131b"); background.addColorStop(1, "#143244"); this.gameBackgroundCache = { key: backgroundKey, value: background }; }
    ctx.fillStyle = this.gameBackgroundCache.value;
    ctx.fillRect(0, 0, width, height);
    const occupiedBody = new Set(snake.slice(1).map((segment) => `${segment.x}:${segment.y}`));
    this.drawSnakeScene(ctx, width, height, { size, snake, food, eatenFoodTrail }, { boardPadding: 18, showTrail: !this.state.training, trailOptions: { guard: (marker) => occupiedBody.has(`${marker.x}:${marker.y}`) }, headOptions: { direction: this.state.game.direction } });
    ctx.fillStyle = "rgba(4, 14, 20, 0.72)";
    ctx.fillRect(14, 14, 280, 66);
    ctx.strokeStyle = "rgba(255,255,255,0.1)";
    ctx.strokeRect(14, 14, 280, 66);
    ctx.fillStyle = "#edf6f9";
    ctx.font = "600 22px Aptos, Trebuchet MS, sans-serif";
    ctx.fillText(`Счет: ${this.state.game.score}`, 28, 44);
    ctx.font = "500 14px Aptos, Trebuchet MS, sans-serif";
    ctx.fillStyle = "#b4c7cf";
    ctx.fillText(`Поле: ${size}x${size} | Вход: ${this.currentInputLabel()}`, 28, 68);
  }

  renderNetworkGraph() {
    if (!this.networkCtx || !this.networkCanvas) return;
    const ctx = this.networkCtx;
    const { width, height } = this.resizeCanvas(this.networkCanvas, ctx);
    const network = this.state.network ?? {};
    const globalSignalPayload = network.global_signals ?? this.state.global_signals ?? {};
    const globalSignalLabels = globalSignalPayload.labels ?? [];
    const globalSignalValues = globalSignalPayload.values ?? [];
    const hidden1 = network.hidden1 ?? [];
    const hidden2 = network.hidden2 ?? [];
    const output = network.output ?? [];
    const connections = network.connections ?? {};
    const labels = this.outputLabels();
    if (hidden1.length > 500 || hidden2.length > 500) { ctx.fillStyle = "#edf6f9"; ctx.font = "600 18px Aptos, Trebuchet MS, sans-serif"; ctx.fillText("Сеть слишком большая для визуализации", 28, 42); return; }
    ctx.clearRect(0, 0, width, height);
    const gradient = ctx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, "#08131d");
    gradient.addColorStop(1, "#102333");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);
    if (!Array.isArray(globalSignalLabels) || !Array.isArray(globalSignalValues) || globalSignalLabels.length !== globalSignalValues.length) {
      ctx.fillStyle = "#edf6f9"; ctx.font = "600 18px Aptos, Trebuchet MS, sans-serif"; ctx.fillText("Граф сети временно недоступен", 28, 42);
      ctx.fillStyle = "#b4c7cf"; ctx.font = "500 14px Aptos, Trebuchet MS, sans-serif"; ctx.fillText(`Ожидались актуальные сигналы, получено ${Array.isArray(globalSignalValues) ? globalSignalValues.length : 0}.`, 28, 70); return;
    }
    const globalSignalLabelSet = new Set(globalSignalLabels);
    const globalSignalIndexByLabel = new Map(globalSignalLabels.map((label, index) => [label, index]));
    const padX = 26; const titleY = 32; const contentTop = 54; const contentBottom = height - 20; const contentHeight = Math.max(160, contentBottom - contentTop);
    const signalGroups = [
      { title: "Еда", labels: ["food_ahead", "food_left", "food_right", "apple_step_forward", "apple_step_left", "apple_step_right"] },
      { title: "Риск", labels: ["danger_forward", "danger_left", "danger_right"] },
      { title: "Безопасность", labels: ["forward_food_reachable", "left_food_reachable", "right_food_reachable", "forward_tail_reachable", "left_tail_reachable", "right_tail_reachable", "forward_enough_space", "left_enough_space", "right_enough_space", "forward_food_tail_safe", "left_food_tail_safe", "right_food_tail_safe"] },
      { title: "Маршрут", labels: ["forward_route_follow", "left_route_follow", "right_route_follow", "forward_route_apple_line", "left_route_apple_line", "right_route_apple_line", "forward_route_return", "left_route_return", "right_route_return"] },
    ].map((group) => ({ ...group, labels: group.labels.filter((label) => globalSignalLabelSet.has(label)) })).filter((group) => group.labels.length > 0);
    const inputWidth = Math.max(220, Math.min(255, width * 0.27)); const inputX = padX; const inputY = contentTop + 4;
    const hidden1X = Math.max(inputX + inputWidth + 76, width * 0.44); const hidden2X = Math.max(hidden1X + 160, width * 0.62); const outputX = width - padX - 74;
    const hidden1Indices = this.pickTopIndices(hidden1, 14); const hidden2Indices = this.pickTopIndices(hidden2, 10);
    const outputIndices = this.outputDisplayOrder(labels).filter((index) => index < output.length);
    const hidden1Positions = this.createColumnPositions(hidden1X, contentTop + contentHeight / 2, hidden1Indices.length, Math.min(32, contentHeight / Math.max(1, hidden1Indices.length)));
    const hidden2Positions = this.createColumnPositions(hidden2X, contentTop + contentHeight / 2, hidden2Indices.length, Math.min(38, contentHeight / Math.max(1, hidden2Indices.length)));
    const outputCenterY = contentTop + contentHeight / 2; const outputPositions = this.createColumnPositions(outputX, outputCenterY, outputIndices.length, Math.min(64, contentHeight / Math.max(1, outputIndices.length)));
    const hidden1Map = new Map(hidden1Indices.map((index, position) => [index, hidden1Positions[position]])); const hidden2Map = new Map(hidden2Indices.map((index, position) => [index, hidden2Positions[position]]));
    const outputMap = new Map(outputIndices.map((index, position) => [index, outputPositions[position]])); const signalMap = new Map();
    if (globalSignalLabels.length) {
      const groupGap = 5; const groupPadding = 8; const pillGap = 6; const pillColumns = 3; const pillWidth = (inputWidth - groupPadding * 2 - pillGap * (pillColumns - 1)) / pillColumns; const pillHeight = 22; const pillRowGap = 23;
      const signalLayouts = signalGroups.map((group) => ({ ...group, rows: Math.ceil(group.labels.length / pillColumns) }));
      const totalSignalHeight = signalLayouts.reduce((sum, group) => sum + 22 + group.rows * pillRowGap + 6, Math.max(0, signalLayouts.length - 1) * groupGap);
      let currentY = inputY + Math.max(0, (contentHeight - totalSignalHeight) / 2);
      signalLayouts.forEach((group) => {
        const groupHeight = 22 + group.rows * pillRowGap + 6; this.roundedRect(ctx, inputX, currentY, inputWidth, groupHeight, 10);
        ctx.fillStyle = "rgba(255,255,255,0.045)"; ctx.fill(); ctx.strokeStyle = "rgba(255,255,255,0.075)"; ctx.lineWidth = 1; ctx.stroke();
        ctx.fillStyle = "#ffcf66"; ctx.font = "700 10.5px Aptos, Trebuchet MS, sans-serif"; ctx.fillText(group.title.toUpperCase(), inputX + groupPadding, currentY + 14.5);
        group.labels.forEach((label, labelOffset) => {
          const signalIndex = globalSignalIndexByLabel.get(label) ?? -1; if (signalIndex < 0) return;
          const colIndex = labelOffset % pillColumns; const rowIndex = Math.floor(labelOffset / pillColumns);
          const x = inputX + groupPadding + colIndex * (pillWidth + pillGap); const y = currentY + 21 + rowIndex * pillRowGap;
          const value = globalSignalValues[signalIndex] ?? 0; this.roundedRect(ctx, x, y, pillWidth, pillHeight, 7);
          ctx.fillStyle = value > 0.66 ? "rgba(255, 179, 71, 0.24)" : value > 0.05 ? "rgba(92, 214, 173, 0.18)" : "rgba(255,255,255,0.06)"; ctx.fill(); ctx.strokeStyle = "rgba(255,255,255,0.055)"; ctx.stroke();
          ctx.fillStyle = "#edf6f9"; ctx.font = "700 11px Aptos, Trebuchet MS, sans-serif";
          const displayValue = Math.abs(value - Math.round(value)) < 0.005 ? String(Math.round(value)) : value.toFixed(2);
          ctx.fillText(`${formatSignalLabel(label)} ${displayValue}`, x + 5, y + 15); signalMap.set(signalIndex, { x: x + pillWidth, y: y + pillHeight / 2 });
        });
        currentY += groupHeight + groupGap;
      });
    }
    (connections.global_hidden1 ?? []).forEach((connection) => { const from = signalMap.get(connection.from_index); const to = hidden1Map.get(connection.to_index); if (!from || !to) return; const alpha = clamp(connection.influence * 4.2, 0.20, 0.78); ctx.beginPath(); ctx.moveTo(from.x, from.y); ctx.bezierCurveTo(from.x + 34, from.y, to.x - 48, to.y, to.x - 10, to.y); ctx.strokeStyle = connection.weight >= 0 ? `rgba(127, 208, 255, ${alpha})` : `rgba(255, 179, 71, ${alpha})`; ctx.lineWidth = clamp(Math.abs(connection.weight) * 2.2, 1.05, 3.0); ctx.stroke(); });
    (connections.hidden1_hidden2 ?? []).forEach((connection) => { const from = hidden1Map.get(connection.from_index); const to = hidden2Map.get(connection.to_index); if (!from || !to) return; const alpha = clamp(connection.influence * 4.4, 0.16, 0.76); ctx.beginPath(); ctx.moveTo(from.x + 10, from.y); ctx.bezierCurveTo(from.x + 55, from.y, to.x - 55, to.y, to.x - 10, to.y); ctx.strokeStyle = connection.weight >= 0 ? `rgba(92, 214, 173, ${alpha})` : `rgba(255, 179, 71, ${alpha})`; ctx.lineWidth = clamp(Math.abs(connection.weight) * 3.0, 1.0, 3.2); ctx.stroke(); });
    (connections.hidden2_output ?? []).forEach((connection) => { const from = hidden2Map.get(connection.from_index); const to = outputMap.get(connection.to_index); if (!from || !to) return; const alpha = clamp(connection.influence * 4.6, 0.18, 0.82); ctx.beginPath(); ctx.moveTo(from.x + 12, from.y); ctx.bezierCurveTo(from.x + 46, from.y, to.x - 46, to.y, to.x - 12, to.y); ctx.strokeStyle = connection.weight >= 0 ? `rgba(92, 214, 173, ${alpha})` : `rgba(255, 179, 71, ${alpha})`; ctx.lineWidth = clamp(Math.abs(connection.weight) * 3.2, 1.1, 3.6); ctx.stroke(); });
    this.drawIndexedLayer(ctx, hidden1Indices, hidden1Positions, hidden1, 10, "H1"); this.drawIndexedLayer(ctx, hidden2Indices, hidden2Positions, hidden2, 12, "H2"); this.drawOutputLayer(ctx, outputIndices, outputPositions, output, labels, "q");
    ctx.fillStyle = "#edf6f9"; ctx.font = "600 14px Aptos, Trebuchet MS, sans-serif"; ctx.fillText("Актуальные сигналы", inputX - 2, titleY); ctx.fillText("Активные нейроны H1", hidden1X - 44, titleY); ctx.fillText("Активные нейроны H2", hidden2X - 42, titleY); ctx.fillText("Решения", outputX - 18, titleY);
  }

  createColumnPositions(centerX, centerY, count, spacing) { const positions = []; const startY = centerY - ((count - 1) * spacing) / 2; for (let index = 0; index < count; index += 1) positions.push({ x: centerX, y: startY + index * spacing }); return positions; }
  drawIndexedLayer(ctx, indices, positions, values, radius, prefix) { positions.forEach((position, offset) => { const index = indices[offset]; const value = values[index] ?? 0; ctx.beginPath(); ctx.arc(position.x, position.y, radius, 0, Math.PI * 2); ctx.fillStyle = value === 0 ? "rgba(255,255,255,0.08)" : colorForActivation(value); ctx.fill(); ctx.strokeStyle = "rgba(255,255,255,0.14)"; ctx.stroke(); ctx.fillStyle = "#b4c7cf"; ctx.font = "600 12.5px Aptos, Trebuchet MS, sans-serif"; ctx.fillText(`${prefix}-${index}`, position.x + 16, position.y + 4.5); }); }
  drawOutputLayer(ctx, indices, positions, values, labels, mode = "q") { positions.forEach((position, offset) => { const index = indices[offset]; const value = values[index] ?? 0; this.roundedRect(ctx, position.x - 26, position.y - 18, 116, 36, 18); ctx.fillStyle = value === 0 ? "rgba(255,255,255,0.06)" : colorForActivation(value); ctx.fill(); ctx.strokeStyle = "rgba(255,255,255,0.12)"; ctx.stroke(); ctx.fillStyle = "#08131d"; ctx.font = "700 14.5px Aptos, Trebuchet MS, sans-serif"; ctx.fillText(`${labels[index] ?? `Out-${index}`}`, position.x - 10, position.y + 5); }); }

  renderChart() {
    if (!this.chartCtx || !this.chartCanvas) return;
    const ctx = this.chartCtx;
    const { width, height } = this.resizeCanvas(this.chartCanvas, ctx);
    const points = this.state.history ?? [];
    const isSmart = this.state.mode === "smart";
    const isDqn = this.state.agent_kind === "dqn" && !isSmart;
    const metricLabel = isDqn ? "Loss" : isSmart ? "Счет" : "Fitness";
    ctx.clearRect(0, 0, width, height); const gradient = ctx.createLinearGradient(0, 0, width, height); gradient.addColorStop(0, "#08131d"); gradient.addColorStop(1, "#0f2231"); ctx.fillStyle = gradient; ctx.fillRect(0, 0, width, height);
    if (points.length < 2) { ctx.fillStyle = "#b4c7cf"; ctx.textAlign = "center"; ctx.font = "500 15px Aptos, Trebuchet MS, sans-serif"; drawCanvasLines(ctx, ["После нескольких эпизодов", `здесь появятся счет, награда и ${metricLabel.toLowerCase()}.`], width / 2, height / 2 - 8, 20); ctx.textAlign = "start"; return; }
    const scores = points.map((point) => point.score); const rewards = points.map((point) => point.reward);
    const fitnesses = points.map((point) => { const value = Number(isDqn ? point.loss ?? 0 : isSmart ? point.score ?? 0 : point.fitness ?? 0); return Number.isFinite(value) ? value : 0; });
    const finiteFitnesses = fitnesses.filter(Number.isFinite); const movingAverage = (series, windowSize) => series.map((_, index) => { const start = Math.max(0, index - windowSize + 1); return average(series.slice(start, index + 1)); });
    const scoreAverage20 = movingAverage(scores, 20); const mainMax = Math.max(1, ...scores, ...rewards, ...scoreAverage20); const mainMin = Math.min(-2, ...scores, ...rewards, ...scoreAverage20); const mainRange = Math.max(1, mainMax - mainMin);
    const fitnessLow = quantile(finiteFitnesses, 0.05); const fitnessHigh = quantile(finiteFitnesses, 0.95); const fitnessPad = Math.max(5, (fitnessHigh - fitnessLow) * 0.18); const fitnessMax = Math.max(0.1, fitnessHigh + fitnessPad); const fitnessMin = Math.min(0, fitnessLow - fitnessPad); const fitnessRange = Math.max(0.1, fitnessMax - fitnessMin);
    const padLeft = 56; const padRight = 52; const padTop = 38; const padBottom = 34; const chartWidth = width - padLeft - padRight; const chartHeight = height - padTop - padBottom;
    const mapMainY = (value) => padTop + chartHeight - ((value - mainMin) / mainRange) * chartHeight;
    const mapFitnessY = (value) => { const clipped = clamp(value, fitnessMin, fitnessMax); return padTop + chartHeight - ((clipped - fitnessMin) / fitnessRange) * chartHeight; };
    const mapX = (index, total) => padLeft + (index / Math.max(1, total - 1)) * chartWidth;
    for (let index = 0; index <= 4; index += 1) { const value = mainMin + (mainRange / 4) * index; const y = mapMainY(value); ctx.beginPath(); ctx.moveTo(padLeft, y); ctx.lineTo(width - padRight, y); ctx.strokeStyle = "rgba(255,255,255,0.08)"; ctx.lineWidth = 1; ctx.stroke(); ctx.fillStyle = "rgba(180,199,207,0.75)"; ctx.font = "11px Aptos, Trebuchet MS, sans-serif"; ctx.fillText(value.toFixed(1), 12, y + 4); }
    for (let index = 0; index <= 4; index += 1) { const value = fitnessMin + (fitnessRange / 4) * index; const y = mapFitnessY(value); ctx.fillStyle = "rgba(127,208,255,0.78)"; ctx.font = "11px Aptos, Trebuchet MS, sans-serif"; ctx.fillText(value.toFixed(2), width - padRight + 8, y + 4); }
    const drawSeries = (series, stroke, fillAlpha, mapY) => { const points2d = series.map((value, index) => ({ x: mapX(index, series.length), y: mapY(value) })); ctx.beginPath(); ctx.moveTo(points2d[0].x, points2d[0].y); for (let index = 1; index < points2d.length; index += 1) { const prev = points2d[index - 1]; const current = points2d[index]; const controlX = (prev.x + current.x) / 2; ctx.quadraticCurveTo(controlX, prev.y, current.x, current.y); } ctx.strokeStyle = stroke; ctx.lineWidth = 2.4; ctx.stroke(); ctx.beginPath(); ctx.moveTo(points2d[0].x, chartHeight + padTop); ctx.lineTo(points2d[0].x, points2d[0].y); for (let index = 1; index < points2d.length; index += 1) { const prev = points2d[index - 1]; const current = points2d[index]; const controlX = (prev.x + current.x) / 2; ctx.quadraticCurveTo(controlX, prev.y, current.x, current.y); } ctx.lineTo(points2d[points2d.length - 1].x, chartHeight + padTop); ctx.closePath(); ctx.fillStyle = `${stroke}${fillAlpha}`; ctx.fill(); };
    const drawLineSeries = (series, stroke, mapY, lineWidth = 3.2) => { const points2d = series.map((value, index) => ({ x: mapX(index, series.length), y: mapY(value) })); ctx.save(); ctx.beginPath(); ctx.moveTo(points2d[0].x, points2d[0].y); for (let index = 1; index < points2d.length; index += 1) { const prev = points2d[index - 1]; const current = points2d[index]; const controlX = (prev.x + current.x) / 2; ctx.quadraticCurveTo(controlX, prev.y, current.x, current.y); } ctx.strokeStyle = stroke; ctx.lineWidth = lineWidth; ctx.lineCap = "round"; ctx.shadowColor = "rgba(237, 246, 249, 0.42)"; ctx.shadowBlur = 8; ctx.stroke(); ctx.restore(); };
    drawSeries(scores, "#5cd6ad", "18", mapMainY); drawSeries(rewards, "#ffb347", "12", mapMainY); drawSeries(fitnesses, "#7fd0ff", "0f", mapFitnessY); drawLineSeries(scoreAverage20, "#edf6f9", mapMainY);
    [["Счет", "#5cd6ad", padLeft], ["Награда", "#ffb347", padLeft + 84], ["Срд счет", "#edf6f9", padLeft + 190], [metricLabel, "#7fd0ff", width - padRight - 76]].forEach(([label, color, x]) => { ctx.fillStyle = color; ctx.beginPath(); ctx.arc(x, 18, 5, 0, Math.PI * 2); ctx.fill(); ctx.fillStyle = "#edf6f9"; ctx.font = "600 13px Aptos, Trebuchet MS, sans-serif"; ctx.fillText(label, x + 12, 22); });
  }
}

new SnakeLabApp();

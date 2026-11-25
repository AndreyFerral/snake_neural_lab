const SnakeRenderers = {
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
  },

  roundedRect(ctx, x, y, width, height, radius) {
    const r = Math.min(radius, width / 2, height / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + width, y, x + width, y + height, r);
    ctx.arcTo(x + width, y + height, x, y + height, r);
    ctx.arcTo(x, y + height, x, y, r);
    ctx.arcTo(x, y, x + width, y, r);
    ctx.closePath();
  },

  drawBoardFrame(ctx, boardX, boardY, boardSize, cells, options = {}) {
    const glow = options.glow ?? 6;
    const strokeOffset = glow - 0.75;
    ctx.fillStyle = "rgba(255,255,255,0.04)";
    ctx.fillRect(boardX - glow, boardY - glow, boardSize + glow * 2, boardSize + glow * 2);
    ctx.strokeStyle = "rgba(255,255,255,0.09)";
    ctx.lineWidth = options.outerLineWidth ?? 1.5;
    ctx.strokeRect(boardX - strokeOffset, boardY - strokeOffset, boardSize + strokeOffset * 2, boardSize + strokeOffset * 2);
    ctx.fillStyle = "rgba(13, 30, 41, 0.96)";
    ctx.fillRect(boardX, boardY, boardSize, boardSize);
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth = options.innerLineWidth ?? 1.5;
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
  },

  fillBoardCell(ctx, boardX, boardY, cell, gridX, gridY, options = {}) {
    const insetRatio = options.insetRatio ?? 0;
    const inset = cell * insetRatio;
    const sizePx = cell - inset * 2;
    ctx.fillStyle = options.fill ?? "rgba(13, 30, 41, 0.68)";
    if (options.rounded === false) {
      ctx.fillRect(boardX + gridX * cell + inset, boardY + gridY * cell + inset, sizePx, sizePx);
      return;
    }
    const radiusRatio = options.radiusRatio ?? 0.24;
    const minRadius = options.minRadius ?? 4;
    this.roundedRect(
      ctx,
      boardX + gridX * cell + inset,
      boardY + gridY * cell + inset,
      sizePx,
      sizePx,
      Math.max(minRadius, cell * radiusRatio)
    );
    ctx.fill();
  },

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
  },

  drawSnakeBody(ctx, boardX, boardY, cell, snake, options = {}) {
    if (!Array.isArray(snake) || !snake.length) return null;
    const centerOf = (segment) => ({ x: boardX + segment.x * cell + cell / 2, y: boardY + segment.y * cell + cell / 2 });
    if (snake.length > 1) {
      ctx.strokeStyle = "#44c8a2";
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.lineWidth = Math.max(options.minLineWidth ?? 6, cell * (options.lineRatio ?? 0.58));
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
      this.roundedRect(ctx, x, y, sizePx, sizePx, Math.max(options.minRadius ?? 8, cell * (options.radiusRatio ?? 0.4)));
      ctx.fillStyle = index === 0 ? "#d7fff4" : index === snake.length - 1 ? "#36b28b" : "#5cd6ad";
      ctx.fill();
    });
    return snake[0];
  },

  drawFoodTrail(ctx, boardX, boardY, cell, trail, options = {}) {
    if (!Array.isArray(trail) || !trail.length) return;
    const guard = options.guard ?? (() => true);
    const baseAlpha = options.baseAlpha ?? 0.16;
    const alphaScale = options.alphaScale ?? 0.20;
    const baseSizeRatio = options.baseSizeRatio ?? 0.24;
    const sizeScale = options.sizeScale ?? 0.10;
    const strokeRgb = options.strokeRgb ?? "255, 214, 220";
    const strokeAlphaScale = options.strokeAlphaScale ?? 0.7;

    ctx.save();
    ctx.fillStyle = "#ff526c";
    ctx.strokeStyle = `rgb(${strokeRgb})`;
    ctx.lineWidth = Math.max(options.minLineWidth ?? 1, cell * (options.lineRatio ?? 0.03));

    trail.forEach((marker, index) => {
      if (!marker || !guard(marker)) return;
      const freshness = (index + 1) / Math.max(1, trail.length);
      const alpha = baseAlpha + freshness * alphaScale;
      const sizePx = cell * (baseSizeRatio + freshness * sizeScale);
      const offset = (cell - sizePx) / 2;

      this.roundedRect(ctx, boardX + marker.x * cell + offset, boardY + marker.y * cell + offset, sizePx, sizePx, Math.max(3, sizePx * 0.30));
      ctx.globalAlpha = alpha;
      ctx.fill();
      ctx.globalAlpha = alpha * strokeAlphaScale;
      ctx.stroke();
    });
    ctx.restore();
  },

  drawSnakeHead(ctx, boardX, boardY, cell, head, options = {}) {
    if (!head) return null;
    const headX = boardX + head.x * cell + cell * 0.04;
    const headY = boardY + head.y * cell + cell * 0.04;
    const sizePx = cell * 0.92;
    this.roundedRect(ctx, headX, headY, sizePx, sizePx, Math.max(options.minRadius ?? 8, cell * (options.radiusRatio ?? 0.42)));
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
  },

  drawSnakeScene(ctx, width, height, scene, options = {}) {
    const deathPreset = options.variant === "death" ? {
      boardPadding: 2,
      frameOptions: { glow: 0, outerLineWidth: 1.25, innerLineWidth: 1 },
      clearOccupiedCells: true,
      clearCellOptions: { fill: "rgba(13, 30, 41, 0.96)", insetRatio: 0.05, rounded: false },
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

    if (!options.skipFrame) {
      this.drawBoardFrame(ctx, boardX, boardY, boardSize, size, sceneOptions.frameOptions ?? {});
    }

    if (food) this.fillBoardCell(ctx, boardX, boardY, cell, food.x, food.y, sceneOptions.foodCellOptions ?? { insetRatio: 0.08, radiusRatio: 0.24 });
    if (sceneOptions.clearOccupiedCells) {
      const clearOptions = sceneOptions.clearCellOptions ?? { fill: "rgba(13, 30, 41, 0.96)", insetRatio: 0.05, rounded: false };
      if (food) this.fillBoardCell(ctx, boardX, boardY, cell, food.x, food.y, clearOptions);
      snake.forEach((segment) => this.fillBoardCell(ctx, boardX, boardY, cell, Math.round(segment.x), Math.round(segment.y), clearOptions));
    } else if (!sceneOptions.interpolated) {
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
  },

  drawStaticBackground(ctx, width, height, size) {
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = "#0c1924";
    ctx.fillRect(0, 0, width, height);
    const boardPadding = 18;
    const boardSize = Math.min(width, height) - boardPadding * 2;
    const boardX = Math.round((width - boardSize) / 2);
    const boardY = Math.round((height - boardSize) / 2);
    const cell = this.drawBoardFrame(ctx, boardX, boardY, boardSize, size, {
      glow: 6,
      outerLineWidth: 1.5,
      innerLineWidth: 1.5
    });
    return { boardX, boardY, boardSize, cell };
  },

  SIGNAL_DESCRIPTIONS: {
    food_ahead: "Еда спереди от головы",
    food_left: "Еда слева от головы",
    food_right: "Еда справа от головы",
    food_behind: "Еда позади головы",
    apple_step_forward: "Яблоко на клетке прямо",
    apple_step_left: "Яблоко на клетке слева",
    apple_step_right: "Яблоко на клетке справа",
    food_reachable: "К еде есть путь (BFS)",
    danger_forward: "Стена или тело прямо",
    danger_left: "Стена или тело слева",
    danger_right: "Стена или тело справа",
    tail_ahead: "Хвост спереди по координатам",
    tail_left: "Хвост слева по координатам",
    tail_right: "Хвост справа по координатам",
    tail_behind: "Хвост позади по координатам",
    tail_blocked_forward: "Ход прямо отрезает путь к хвосту",
    tail_blocked_left: "Ход влево отрезает путь к хвосту",
    tail_blocked_right: "Ход вправо отрезает путь к хвосту",
    tail_close_forward: "Сближение вплотную с хвостом прямо",
    tail_close_left: "Сближение вплотную с хвостом слева",
    tail_close_right: "Сближение вплотную с хвостом справа",
    partition_forward: "Ход прямо отрезает доступ к еде",
    partition_left: "Ход влево отрезает доступ к еде",
    partition_right: "Ход вправо отрезает доступ к еде",
    endgame: "Эндшпиль (заполнено >= 75% поля)",
    pocket_split_forward: "Ход прямо дробит пустое поле на карманы",
    pocket_split_left: "Ход влево дробит пустое поле на карманы",
    pocket_split_right: "Ход вправо дробит пустое поле на карманы",
    hunger_high: "Высокий голод (>= 65% лимита без еды)",
    hunger_critical: "Критический голод (>= 85% лимита)",
    board_split: "Поле уже разделено на >= 2 кармана",
    pocket_merge_forward: "Ход прямо склеивает карманы обратно",
    pocket_merge_left: "Ход влево склеивает карманы обратно",
    pocket_merge_right: "Ход вправо склеивает карманы обратно",
    area_ratio_forward: "Доля доступной свободной площади после хода прямо",
    area_ratio_left: "Доля доступной свободной площади после хода влево",
    area_ratio_right: "Доля доступной свободной площади после хода вправо",
    fill_ratio: "Доля заполнения поля змейкой (длина / площадь)",
    food_dist_norm: "Нормированное манхэттенское расстояние до яблока",
    tail_dist_norm: "Нормированное манхэттенское расстояние до хвоста",
  },

  createColumnPositions(centerX, centerY, count, spacing) {
    const positions = [];
    const startY = centerY - ((count - 1) * spacing) / 2;
    for (let index = 0; index < count; index += 1) positions.push({ x: centerX, y: startY + index * spacing });
    return positions;
  },

  drawIndexedLayer(ctx, indices, positions, values, radius, prefix, hoveredTarget = null) {
    positions.forEach((position, offset) => {
      const index = indices[offset];
      const value = values[index] ?? 0;
      const isHovered = Boolean(hoveredTarget && hoveredTarget.layer === prefix && hoveredTarget.index === index);
      const r = isHovered ? radius + 2.5 : radius;
      ctx.beginPath();
      ctx.arc(position.x, position.y, r, 0, Math.PI * 2);
      ctx.fillStyle = value === 0 ? "rgba(255,255,255,0.10)" : colorForActivation(value);
      ctx.fill();
      ctx.lineWidth = isHovered ? 2.2 : 1.1;
      ctx.strokeStyle = isHovered ? "#ffffff" : "rgba(255,255,255,0.22)";
      ctx.stroke();
      ctx.fillStyle = value > 0.15 ? "#07141f" : "#edf6f9";
      ctx.font = "700 10.5px Aptos, Trebuchet MS, sans-serif";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(String(index), position.x, position.y + 0.5);
    });
    ctx.textAlign = "start";
    ctx.textBaseline = "alphabetic";
  },

  drawOutputLayer(ctx, indices, positions, values, labels, hoveredTarget = null, pillOptions = null) {
    const boxW = pillOptions?.width ?? 148;
    const boxH = pillOptions?.height ?? 40;
    const radius = Math.round(boxH / 2);
    const bestIndex = indices.reduce((best, idx) => ((values[idx] ?? -Infinity) > (values[best] ?? -Infinity) ? idx : best), indices[0]);
    positions.forEach((position, offset) => {
      const index = indices[offset];
      const value = values[index] ?? 0;
      const isHovered = Boolean(hoveredTarget && hoveredTarget.layer === "OUT" && hoveredTarget.index === index);
      const isBest = index === bestIndex && indices.length > 0;
      const boxX = position.x - 16;
      const boxY = position.y - boxH / 2;
      this.roundedRect(ctx, boxX, boxY, boxW, boxH, radius);
      const isNearZero = Math.abs(value) < 0.005;
      ctx.fillStyle = isNearZero ? "rgba(255,255,255,0.06)" : colorForActivation(value);
      ctx.fill();
      ctx.lineWidth = isHovered ? 2.2 : isBest ? 1.8 : 1.0;
      ctx.strokeStyle = isHovered ? "#ffffff" : isBest ? "rgba(255,255,255,0.65)" : "rgba(255,255,255,0.14)";
      ctx.stroke();

      const isLightFill = !isNearZero && value > 0.15;
      const textColor = isLightFill ? "#08131d" : "#edf6f9";
      ctx.fillStyle = textColor;
      ctx.font = "700 13.5px Aptos, Trebuchet MS, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(`${labels[index] ?? `Out-${index}`}`, boxX + 13, position.y);

      if (Number.isFinite(value)) {
        const valStr = `${value >= 0 ? "+" : ""}${value.toFixed(2)}`;
        ctx.font = "700 12px Aptos, Trebuchet MS, sans-serif";
        ctx.textAlign = "right";
        ctx.fillStyle = isLightFill ? "rgba(8,19,29,0.78)" : "rgba(237,246,249,0.85)";
        ctx.fillText(valStr, boxX + boxW - 13, position.y);
      }
    });
    ctx.textAlign = "start";
    ctx.textBaseline = "alphabetic";
  },

  drawConnections(ctx, connectionList, fromMap, toMap, options) {
    if (!Array.isArray(connectionList) || !connectionList.length) return;
    const xOffsetFrom = options.xOffsetFrom ?? 0;
    const xOffsetTo = options.xOffsetTo ?? 0;
    const cpX1 = options.cpX1 ?? 0;
    const cpX2 = options.cpX2 ?? 0;
    const alphaMult = options.alphaMult ?? 1.0;
    const alphaMin = options.alphaMin ?? 0.20;
    const alphaMax = options.alphaMax ?? 0.80;
    const widthMult = options.widthMult ?? 1.0;
    const widthMin = options.widthMin ?? 1.0;
    const widthMax = options.widthMax ?? 3.0;
    const posRgb = options.posRgb ?? "127, 208, 255";
    const negRgb = options.negRgb ?? "255, 179, 71";
    const highlightFrom = options.highlightFrom ?? null;
    const highlightTo = options.highlightTo ?? null;
    const hasFilter = highlightFrom !== null || highlightTo !== null;
    const routedFromSignals = Boolean(options.routedFromSignals);

    const groups = {};
    connectionList.forEach((connection) => {
      const from = fromMap.get(connection.from_index);
      const to = toMap.get(connection.to_index);
      if (!from || !to) return;

      const matchesHighlight = (highlightFrom !== null && connection.from_index === highlightFrom) ||
        (highlightTo !== null && connection.to_index === highlightTo);
      let alpha = clamp(connection.influence * alphaMult, alphaMin, alphaMax);
      const weight = connection.weight ?? 0;
      let lineWidth = clamp(Math.abs(weight) * widthMult, widthMin, widthMax);

      if (hasFilter) {
        if (matchesHighlight) {
          alpha = Math.max(0.75, alpha);
          lineWidth = Math.max(2.0, lineWidth * 1.35);
        } else {
          alpha = 0.05;
          lineWidth = 0.8;
        }
      }

      const roundedAlpha = Math.round(alpha * 20) / 20;
      const roundedLineWidth = Math.round(lineWidth * 10) / 10;
      const isPos = weight >= 0 ? 1 : 0;

      const key = `${isPos}|${roundedAlpha}|${roundedLineWidth}`;
      if (!groups[key]) groups[key] = [];
      groups[key].push({ from, to });
    });

    for (const [key, paths] of Object.entries(groups)) {
      const [isPosStr, alphaStr, lineWidthStr] = key.split("|");
      const isPos = isPosStr === "1";
      const alpha = parseFloat(alphaStr);
      const lineWidth = parseFloat(lineWidthStr);
      const rgb = isPos ? posRgb : negRgb;

      ctx.strokeStyle = `rgba(${rgb}, ${alpha})`;
      ctx.lineWidth = lineWidth;
      ctx.lineCap = "round";
      ctx.lineJoin = "round";
      ctx.beginPath();
      paths.forEach(({ from, to }) => {
        if (routedFromSignals && from.laneY !== undefined && from.busX !== undefined && !from.isLastInRow) {
          const sx = from.x + xOffsetFrom;
          const sy = from.y;
          const stubX = sx + 4;
          const bx = from.busX;
          const ly = from.laneY;
          const tx = to.x + xOffsetTo;
          const ty = to.y;
          ctx.moveTo(sx, sy);
          ctx.lineTo(stubX, ly);
          ctx.lineTo(bx, ly);
          ctx.bezierCurveTo(bx + cpX1, ly, tx + cpX2, ty, tx, ty);
        } else {
          const sx = (from.busX ?? from.x) + xOffsetFrom;
          const sy = from.y;
          const tx = to.x + xOffsetTo;
          const ty = to.y;
          if (routedFromSignals && from.busX !== undefined) {
            ctx.moveTo(from.x + xOffsetFrom, sy);
            ctx.lineTo(sx, sy);
          } else {
            ctx.moveTo(sx, sy);
          }
          ctx.bezierCurveTo(sx + cpX1, sy, tx + cpX2, ty, tx, ty);
        }
      });
      ctx.stroke();
    }
  },

  drawNetworkGraph(ctx, width, height, state, labels, outputIndices, hoverState = null, hitCache = null) {
    const network = state.network ?? {};
    const globalSignalPayload = network.global_signals ?? state.global_signals ?? {};
    const globalSignalLabels = globalSignalPayload.labels ?? [];
    const globalSignalValues = globalSignalPayload.values ?? [];
    const hidden1 = network.hidden1 ?? [];
    const hidden2 = network.hidden2 ?? [];
    const output = network.output ?? [];
    const connections = network.connections ?? {};
    if (hitCache) {
      hitCache.signals = [];
      hitCache.hidden1 = [];
      hitCache.hidden2 = [];
      hitCache.outputs = [];
    }
    if (hidden1.length > 500 || hidden2.length > 500) {
      ctx.fillStyle = "#edf6f9";
      ctx.font = "600 18px Aptos, Trebuchet MS, sans-serif";
      ctx.fillText("Сеть слишком большая для визуализации", 28, 42);
      return;
    }
    ctx.clearRect(0, 0, width, height);
    const gradient = ctx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, "#08131d");
    gradient.addColorStop(1, "#102333");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);
    if (!Array.isArray(globalSignalLabels) || !Array.isArray(globalSignalValues) || globalSignalLabels.length !== globalSignalValues.length) {
      ctx.fillStyle = "#edf6f9";
      ctx.font = "600 18px Aptos, Trebuchet MS, sans-serif";
      ctx.fillText("Граф сети временно недоступен", 28, 42);
      ctx.fillStyle = "#b4c7cf";
      ctx.font = "500 14px Aptos, Trebuchet MS, sans-serif";
      ctx.fillText(`Ожидались актуальные сигналы, получено ${Array.isArray(globalSignalValues) ? globalSignalValues.length : 0}.`, 28, 70);
      return;
    }

    const globalSignalLabelSet = new Set(globalSignalLabels);
    const globalSignalIndexByLabel = new Map(globalSignalLabels.map((label, index) => [label, index]));
    const padX = 20;
    const titleY = 26;
    const contentTop = 42;
    const contentBottom = height - 14;
    const contentHeight = Math.max(200, contentBottom - contentTop);

    const inputWidth = clamp(Math.round(width * 0.30), 236, 336);
    const inputX = padX;
    const groupPadding = 8;
    const pillGap = 7;
    const minPillWidth = 60;
    const innerWidth = inputWidth - groupPadding * 2;
    const maxColsPerRow = Math.max(1, Math.floor((innerWidth + pillGap) / (minPillWidth + pillGap)));

    const rawGroups = [
      {
        title: "Еда",
        rows: [
          ["food_ahead", "food_left", "food_right", "food_behind"],
          ["apple_step_forward", "apple_step_left", "apple_step_right", "food_reachable"]
        ]
      },
      {
        title: "Состояние",
        rows: [
          ["endgame", "hunger_high", "hunger_critical", "board_split"],
          ["fill_ratio", "food_dist_norm", "tail_dist_norm"]
        ]
      },
      {
        title: "Хвост",
        rows: [
          ["tail_ahead", "tail_left", "tail_right", "tail_behind"],
          ["tail_close_forward", "tail_close_left", "tail_close_right"]
        ]
      },
      {
        title: "Риск",
        rows: [
          ["danger_forward", "danger_left", "danger_right"],
          ["tail_blocked_forward", "tail_blocked_left", "tail_blocked_right"],
          ["partition_forward", "partition_left", "partition_right"],
          ["pocket_split_forward", "pocket_split_left", "pocket_split_right"],
          ["pocket_merge_forward", "pocket_merge_left", "pocket_merge_right"],
          ["area_ratio_forward", "area_ratio_left", "area_ratio_right"]
        ]
      }
    ];

    const assignedLabels = new Set();
    const signalGroups = rawGroups.map((group) => {
      const wrappedRows = [];
      group.rows.forEach((row) => {
        const present = row.filter((label) => globalSignalLabelSet.has(label));
        present.forEach((label) => assignedLabels.add(label));
        for (let i = 0; i < present.length; i += maxColsPerRow) {
          wrappedRows.push(present.slice(i, i + maxColsPerRow));
        }
      });
      return { title: group.title, rows: wrappedRows };
    }).filter((group) => group.rows.length > 0);

    const unassigned = globalSignalLabels.filter((label) => !assignedLabels.has(label));
    if (unassigned.length > 0) {
      const extraRows = [];
      for (let i = 0; i < unassigned.length; i += maxColsPerRow) {
        extraRows.push(unassigned.slice(i, i + maxColsPerRow));
      }
      signalGroups.push({ title: "Прочее", rows: extraRows });
    }

    const egoMap = Array.isArray(network.ego_map) && network.ego_map.length >= 3 ? network.ego_map : null;
    const isPpo = state.agent_kind === "ppo" && state.mode !== "smart";
    const showEgoMapCard = Boolean(egoMap && width >= 680);
    const inputBusX = inputX + inputWidth + 12;
    const outputBoxLeftOffset = 16;
    const rightCardPad = 8;
    const rightCardWidth = showEgoMapCard ? 160 : 148;
    const rightCardX = width - padX - rightCardWidth;
    const outputBoxWidth = showEgoMapCard ? (rightCardWidth - rightCardPad * 2) : 148;
    const outputPillHeight = showEgoMapCard ? 36 : 40;
    const outputX = (showEgoMapCard ? rightCardX + rightCardPad : rightCardX) + outputBoxLeftOffset;
    const graphSpan = (outputX - outputBoxLeftOffset) - inputBusX;
    const hidden1X = inputBusX + graphSpan * 0.34;
    const hidden2X = inputBusX + graphSpan * 0.68;

    const hidden1Indices = this.pickTopIndices(hidden1, 14);
    const hidden2Indices = this.pickTopIndices(hidden2, 10);
    const validOutputIndices = outputIndices.filter((index) => index < output.length);
    const hidden1Positions = this.createColumnPositions(hidden1X, contentTop + contentHeight / 2, hidden1Indices.length, Math.min(35, contentHeight / Math.max(1, hidden1Indices.length)));
    const hidden2Positions = this.createColumnPositions(hidden2X, contentTop + contentHeight / 2, hidden2Indices.length, Math.min(44, contentHeight / Math.max(1, hidden2Indices.length)));

    const networkValue = Array.isArray(network.value) && network.value.length > 0 ? Number(network.value[0]) : null;
    const showPpoValueBadge = Boolean(isPpo && networkValue !== null && Number.isFinite(networkValue) && validOutputIndices.length > 0);
    const policyHeaderH = 22;
    const outputGap = 7;
    const outputStep = outputPillHeight + outputGap;
    const outputsBlockH = validOutputIndices.length > 0
      ? validOutputIndices.length * outputPillHeight + (validOutputIndices.length - 1) * outputGap
      : 0;
    const valueBadgeGap = showPpoValueBadge ? 8 : 0;
    const valueBadgeH = showPpoValueBadge ? 28 : 0;
    const policyCardH = policyHeaderH + outputsBlockH + valueBadgeGap + valueBadgeH + 8;
    const rightGroupGap = 14;
    const egoMapCols = showEgoMapCard ? Math.max(3, (egoMap[0] && egoMap[0].length) || 11) : 11;
    const rawMapGridSize = showEgoMapCard
      ? clamp(Math.floor(Math.min(rightCardWidth - 28, contentHeight - policyCardH - rightGroupGap - 54)), 88, 132)
      : 0;
    const mapGridSize = showEgoMapCard
      ? Math.max(egoMapCols * 8, Math.floor(rawMapGridSize / egoMapCols) * egoMapCols)
      : 0;
    const mapCardH = showEgoMapCard ? (24 + mapGridSize + 22) : 0;
    const totalRightStackH = showEgoMapCard ? (policyCardH + rightGroupGap + mapCardH) : 0;
    const rightStartY = contentTop + Math.max(0, Math.round((contentHeight - totalRightStackH) / 2));

    const outputCenterY = showEgoMapCard
      ? rightStartY + policyHeaderH + outputsBlockH / 2
      : contentTop + contentHeight / 2;
    const outputSpacing = showEgoMapCard
      ? outputStep
      : Math.min(76, contentHeight / Math.max(1, validOutputIndices.length));
    const outputPositions = this.createColumnPositions(outputX, outputCenterY, validOutputIndices.length, outputSpacing);

    const hidden1Map = new Map(hidden1Indices.map((index, position) => [index, hidden1Positions[position]]));
    const hidden2Map = new Map(hidden2Indices.map((index, position) => [index, hidden2Positions[position]]));
    const outputMap = new Map(validOutputIndices.map((index, position) => [index, outputPositions[position]]));
    const signalMap = new Map();

    // Подсчёт активных соединений для индикаторов портов на чипах
    const connectedSignalIndices = new Set();
    if (Array.isArray(connections.global_hidden1)) {
      connections.global_hidden1.forEach((c) => {
        if (hidden1Map.has(c.to_index)) connectedSignalIndices.add(c.from_index);
      });
    }

    // Расчёт раскладки карточек входов под доступную высоту холста
    if (globalSignalLabels.length) {
      const totalRows = signalGroups.reduce((sum, g) => sum + g.rows.length, 0);
      const groupCount = signalGroups.length;
      const groupGap = totalRows > 10 ? 6 : 8;
      const groupHeaderH = 21;
      const groupBottomPad = 4;
      const availableForRows = contentHeight - groupCount * (groupHeaderH + groupBottomPad) - Math.max(0, groupCount - 1) * groupGap;
      const rowUnit = Math.min(31, Math.max(24, Math.floor(availableForRows / Math.max(1, totalRows))));
      const pillVerticalGap = rowUnit >= 28 ? 10 : 7;
      const pillHeight = rowUnit - pillVerticalGap;
      const totalSignalHeight = totalRows * rowUnit + groupCount * (groupHeaderH + groupBottomPad) + Math.max(0, groupCount - 1) * groupGap;
      let currentY = contentTop + Math.max(0, (contentHeight - totalSignalHeight) / 2);

      signalGroups.forEach((group) => {
        const groupHeight = groupHeaderH + group.rows.length * rowUnit + groupBottomPad;
        this.roundedRect(ctx, inputX, currentY, inputWidth, groupHeight, 10);
        ctx.fillStyle = "rgba(255,255,255,0.042)";
        ctx.fill();
        ctx.strokeStyle = "rgba(255,255,255,0.08)";
        ctx.lineWidth = 1;
        ctx.stroke();

        ctx.textAlign = "left";
        ctx.textBaseline = "alphabetic";
        ctx.fillStyle = "#ffcf66";
        ctx.font = "700 10.5px Aptos, Trebuchet MS, sans-serif";
        ctx.fillText(group.title.toUpperCase(), inputX + groupPadding, currentY + 14.5);

        group.rows.forEach((rowLabels, rowIndex) => {
          const cols = rowLabels.length;
          const pillWidth = (innerWidth - pillGap * (cols - 1)) / cols;
          const y = currentY + groupHeaderH + rowIndex * rowUnit;

          rowLabels.forEach((label, colIndex) => {
            const signalIndex = globalSignalIndexByLabel.get(label) ?? -1;
            if (signalIndex < 0) return;
            const x = inputX + groupPadding + colIndex * (pillWidth + pillGap);
            const value = globalSignalValues[signalIndex] ?? 0;
            const isHovered = Boolean(hoverState && hoverState.layer === "IN" && hoverState.index === signalIndex);

            this.roundedRect(ctx, x, y, pillWidth, pillHeight, 7);
            ctx.fillStyle = isHovered
              ? "rgba(127, 208, 255, 0.28)"
              : value > 0.66
                ? "rgba(255, 179, 71, 0.25)"
                : value > 0.05
                  ? "rgba(92, 214, 173, 0.20)"
                  : "rgba(255,255,255,0.065)";
            ctx.fill();
            ctx.lineWidth = isHovered ? 1.6 : 1;
            ctx.strokeStyle = isHovered ? "#ffffff" : value > 0.66 ? "rgba(255, 179, 71, 0.45)" : "rgba(255,255,255,0.08)";
            ctx.stroke();

            const displayValue = Math.abs(value - Math.round(value)) < 0.005 ? String(Math.round(value)) : value.toFixed(2);
            const shortText = formatSignalLabel(label);
            ctx.font = "700 11px Aptos, Trebuchet MS, sans-serif";
            ctx.textBaseline = "middle";
            ctx.textAlign = "left";
            ctx.fillStyle = "#edf6f9";
            ctx.fillText(shortText, x + 6, y + pillHeight / 2 + 0.5);

            ctx.textAlign = "right";
            ctx.fillStyle = value > 0.66 ? "#ffd185" : value > 0.05 ? "#7fe4c1" : "#9db3be";
            ctx.fillText(displayValue, x + pillWidth - 6, y + pillHeight / 2 + 0.5);

            const isLastInRow = colIndex === cols - 1;
            // Маршрутизируем линии не по тексту чипов, а по межстрочным желобкам
            let laneY = y + pillHeight / 2;
            if (!isLastInRow) {
              if (colIndex < Math.ceil((cols - 1) / 2)) {
                laneY = y - Math.min(4.5, pillVerticalGap * 0.45) + colIndex * 1.6;
              } else {
                laneY = y + pillHeight + Math.min(4.5, pillVerticalGap * 0.45);
              }
            }
            const portX = x + pillWidth;
            const portY = y + pillHeight / 2;
            if (connectedSignalIndices.has(signalIndex)) {
              ctx.beginPath();
              ctx.arc(portX, portY, isHovered ? 3.2 : 2.3, 0, Math.PI * 2);
              ctx.fillStyle = value > 0.66 ? "#ffb347" : "#7fd0ff";
              ctx.fill();
            }

            signalMap.set(signalIndex, {
              x: portX,
              y: portY,
              laneY,
              busX: inputBusX,
              isLastInRow,
            });
            if (hitCache) {
              hitCache.signals.push({
                x, y, w: pillWidth, h: pillHeight,
                layer: "IN", index: signalIndex, label, value,
              });
            }
          });
        });
        currentY += groupHeight + groupGap;
      });
      ctx.textAlign = "start";
      ctx.textBaseline = "alphabetic";
    }

    const hFromIn = hoverState?.layer === "IN" ? hoverState.index : null;
    const hToH1 = hoverState?.layer === "H1" ? hoverState.index : null;
    const hFromH1 = hoverState?.layer === "H1" ? hoverState.index : null;
    const hToH2 = hoverState?.layer === "H2" ? hoverState.index : null;
    const hFromH2 = hoverState?.layer === "H2" ? hoverState.index : null;
    const hToOut = hoverState?.layer === "OUT" ? hoverState.index : null;

    this.drawConnections(ctx, connections.global_hidden1, signalMap, hidden1Map, {
      xOffsetFrom: 0, xOffsetTo: -13, cpX1: 42, cpX2: -54,
      alphaMult: 4.2, alphaMin: 0.22, alphaMax: 0.82,
      widthMult: 2.2, widthMin: 1.05, widthMax: 3.0,
      posRgb: "127, 208, 255", negRgb: "255, 179, 71",
      highlightFrom: hFromIn, highlightTo: hToH1,
      routedFromSignals: true,
    });
    this.drawConnections(ctx, connections.hidden1_hidden2, hidden1Map, hidden2Map, {
      xOffsetFrom: 13, xOffsetTo: -13, cpX1: 62, cpX2: -62,
      alphaMult: 4.4, alphaMin: 0.16, alphaMax: 0.78,
      widthMult: 3.0, widthMin: 1.0, widthMax: 3.2,
      posRgb: "92, 214, 173", negRgb: "255, 179, 71",
      highlightFrom: hFromH1, highlightTo: hToH2,
    });
    this.drawConnections(ctx, connections.hidden2_output, hidden2Map, outputMap, {
      xOffsetFrom: 13, xOffsetTo: -16, cpX1: 52, cpX2: -52,
      alphaMult: 4.6, alphaMin: 0.18, alphaMax: 0.84,
      widthMult: 3.2, widthMin: 1.1, widthMax: 3.6,
      posRgb: "92, 214, 173", negRgb: "255, 179, 71",
      highlightFrom: hFromH2, highlightTo: hToOut,
    });

    this.drawIndexedLayer(ctx, hidden1Indices, hidden1Positions, hidden1, 12.5, "H1", hoverState);
    this.drawIndexedLayer(ctx, hidden2Indices, hidden2Positions, hidden2, 13.5, "H2", hoverState);

    if (showEgoMapCard && validOutputIndices.length > 0) {
      this.roundedRect(ctx, rightCardX, rightStartY, rightCardWidth, policyCardH, 10);
      ctx.fillStyle = "rgba(255,255,255,0.042)";
      ctx.fill();
      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.lineWidth = 1;
      ctx.stroke();

      ctx.textAlign = "left";
      ctx.textBaseline = "alphabetic";
      ctx.fillStyle = "#ffcf66";
      ctx.font = "700 10.5px Aptos, Trebuchet MS, sans-serif";
      ctx.fillText("ДЕЙСТВИЯ", rightCardX + rightCardPad, rightStartY + 14.5);
    }

    this.drawOutputLayer(ctx, validOutputIndices, outputPositions, output, labels, hoverState, {
      width: outputBoxWidth,
      height: outputPillHeight,
    });

    if (showPpoValueBadge && validOutputIndices.length > 0) {
      const lastOut = outputPositions[outputPositions.length - 1];
      const badgeX = showEgoMapCard ? rightCardX + rightCardPad : outputX - 16;
      const badgeY = Math.round(lastOut.y + outputPillHeight / 2 + valueBadgeGap);
      this.roundedRect(ctx, badgeX, badgeY, outputBoxWidth, valueBadgeH, Math.round(valueBadgeH / 2));
      ctx.fillStyle = "rgba(127, 208, 255, 0.14)";
      ctx.fill();
      ctx.strokeStyle = "rgba(127, 208, 255, 0.40)";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = "#9fd8ff";
      ctx.font = "600 12px Aptos, Trebuchet MS, sans-serif";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText("V(s)", badgeX + 12, badgeY + valueBadgeH / 2);
      ctx.textAlign = "right";
      ctx.fillStyle = "#edf6f9";
      ctx.font = "700 12.5px Aptos, Trebuchet MS, sans-serif";
      ctx.fillText(`${networkValue >= 0 ? "+" : ""}${networkValue.toFixed(2)}`, badgeX + outputBoxWidth - 12, badgeY + valueBadgeH / 2);
      ctx.textAlign = "start";
      ctx.textBaseline = "alphabetic";
    }

    if (showEgoMapCard && egoMap) {
      const mapCardY = rightStartY + policyCardH + rightGroupGap;
      this.drawEgoMapCard(ctx, rightCardX, mapCardY, rightCardWidth, mapCardH, mapGridSize, egoMap);
    }

    if (hitCache) {
      hidden1Indices.forEach((index, offset) => {
        const pos = hidden1Positions[offset];
        hitCache.hidden1.push({ x: pos.x, y: pos.y, r: 15, layer: "H1", index, value: hidden1[index] ?? 0 });
      });
      hidden2Indices.forEach((index, offset) => {
        const pos = hidden2Positions[offset];
        hitCache.hidden2.push({ x: pos.x, y: pos.y, r: 16, layer: "H2", index, value: hidden2[index] ?? 0 });
      });
      validOutputIndices.forEach((index, offset) => {
        const pos = outputPositions[offset];
        hitCache.outputs.push({
          x: pos.x - 16, y: pos.y - outputPillHeight / 2, w: outputBoxWidth, h: outputPillHeight,
          layer: "OUT", index, label: labels[index] ?? `Out-${index}`, value: output[index] ?? 0,
        });
      });
    }

    const isMobile = width < 680;
    const headerInput = isMobile ? "Вход" : `Входные признаки (${globalSignalLabels.length})`;
    const headerH1 = isMobile ? "H1" : `Скрытый H1 (${hidden1.length})`;
    const headerH2 = isMobile ? "H2" : `Скрытый H2 (${hidden2.length})`;
    const headerOutput = isMobile ? "Выход" : (isPpo ? "Политика и карта" : "Решения");

    ctx.fillStyle = "#edf6f9";
    ctx.font = "600 13.5px Aptos, Trebuchet MS, sans-serif";
    ctx.textAlign = "start";
    ctx.fillText(headerInput, inputX, titleY);
    ctx.textAlign = "center";
    ctx.fillText(headerH1, hidden1X, titleY);
    ctx.fillText(headerH2, hidden2X, titleY);
    ctx.fillText(headerOutput, rightCardX + rightCardWidth / 2, titleY);
    ctx.textAlign = "start";

    if (hoverState) {
      this.drawNetworkTooltip(ctx, width, height, hoverState, connections, globalSignalLabels, labels);
    }
  },

  drawEgoMapCard(ctx, x, y, cardW, cardH, gridSize, egoMap) {
    const chWall = egoMap[0] || [];
    const chBody = egoMap[1] || [];
    const chFood = egoMap[2] || [];
    const chReach = egoMap[3] || [];
    const rows = chWall.length || 11;
    const cols = (chWall[0] && chWall[0].length) || 11;
    const gx = x + Math.round((cardW - gridSize) / 2);
    const gy = y + 24;
    const cell = gridSize / cols;

    ctx.save();
    this.roundedRect(ctx, x, y, cardW, cardH, 10);
    ctx.fillStyle = "rgba(255,255,255,0.042)";
    ctx.fill();
    ctx.strokeStyle = "rgba(127, 208, 255, 0.24)";
    ctx.lineWidth = 1;
    ctx.stroke();

    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillStyle = "#7fd0ff";
    ctx.font = "700 10.5px Aptos, Trebuchet MS, sans-serif";
    ctx.fillText(`КАРТА ${cols}×${rows} (↑)`, x + 8, y + 15);

    const centerR = Math.floor(rows / 2);
    const centerC = Math.floor(cols / 2);
    for (let r = 0; r < rows; r += 1) {
      for (let c = 0; c < cols; c += 1) {
        const wall = Number(chWall?.[r]?.[c] ?? 0);
        const body = Number(chBody?.[r]?.[c] ?? 0);
        const food = Number(chFood?.[r]?.[c] ?? 0);
        const reach = Number(chReach?.[r]?.[c] ?? 0);

        let fill = reach > 0.5 ? "rgba(127, 208, 255, 0.14)" : "rgba(255, 104, 124, 0.10)";
        if (wall > 0.5) fill = "#2b3e50";
        else if (food > 0.5) fill = "#ff687c";
        else if (r === centerR && c === centerC) fill = "#ffcf66";
        else if (body > 0.0) {
          const alpha = clamp(0.28 + body * 0.72, 0.28, 1.0);
          fill = `rgba(92, 214, 173, ${alpha.toFixed(2)})`;
        }

        const cx = gx + c * cell;
        const cy = gy + r * cell;
        ctx.fillStyle = fill;
        ctx.fillRect(cx + 0.5, cy + 0.5, Math.max(1, cell - 1), Math.max(1, cell - 1));
      }
    }

    ctx.strokeStyle = "rgba(255,255,255,0.14)";
    ctx.lineWidth = 1;
    ctx.strokeRect(gx, gy, gridSize, gridSize);

    const legendY = gy + gridSize + 12;
    const legendItems = [
      ["#ffcf66", "Гол."],
      ["#ff687c", "Еда"],
      ["#5cd6ad", "Тело"],
    ];
    const stepX = Math.floor(gridSize / legendItems.length);
    ctx.font = "500 9.5px Aptos, Trebuchet MS, sans-serif";
    ctx.textBaseline = "middle";
    legendItems.forEach(([color, label], idx) => {
      const lx = gx + idx * stepX + 3;
      ctx.fillStyle = color;
      ctx.fillRect(lx, legendY - 3, 6, 6);
      ctx.fillStyle = "#b4c7cf";
      ctx.fillText(label, lx + 9, legendY);
    });
    ctx.restore();
  },

  drawNetworkTooltip(ctx, width, height, hoverState, connections, globalSignalLabels, outputLabels) {
    const lines = [];
    if (hoverState.layer === "IN") {
      const desc = this.SIGNAL_DESCRIPTIONS[hoverState.label] || hoverState.label;
      const numVal = Number(hoverState.value);
      const valText = Math.abs(numVal - Math.round(numVal)) < 0.005 ? numVal.toFixed(0) : numVal.toFixed(2);
      lines.push(`${formatSignalLabel(hoverState.label)}  ${hoverState.label} = ${valText}`);
      lines.push(desc);
      const topConns = (connections.global_hidden1 || [])
        .filter((c) => c.from_index === hoverState.index)
        .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
        .slice(0, 3);
      topConns.forEach((c) => {
        const w = c.weight ?? 0;
        lines.push(`→ H1-${c.to_index}: w=${w >= 0 ? "+" : ""}${w.toFixed(2)}`);
      });
    } else if (hoverState.layer === "H1" || hoverState.layer === "H2") {
      lines.push(`${hoverState.layer}-${hoverState.index}: активация = ${Number(hoverState.value).toFixed(3)}`);
      const nextList = hoverState.layer === "H1" ? connections.hidden1_hidden2 : connections.hidden2_output;
      const targetPrefix = hoverState.layer === "H1" ? "H2-" : "";
      const topConns = (nextList || [])
        .filter((c) => c.from_index === hoverState.index)
        .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
        .slice(0, 3);
      topConns.forEach((c) => {
        const w = c.weight ?? 0;
        const targetLabel = hoverState.layer === "H2"
          ? (outputLabels[c.to_index] ?? `Out-${c.to_index}`)
          : `${targetPrefix}${c.to_index}`;
        lines.push(`→ ${targetLabel}: w=${w >= 0 ? "+" : ""}${w.toFixed(2)}`);
      });
    } else if (hoverState.layer === "OUT") {
      lines.push(`Выход «${hoverState.label}»: ${Number(hoverState.value).toFixed(3)}`);
      const topConns = (connections.hidden2_output || [])
        .filter((c) => c.to_index === hoverState.index)
        .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
        .slice(0, 3);
      topConns.forEach((c) => {
        const w = c.weight ?? 0;
        lines.push(`← H2-${c.from_index}: w=${w >= 0 ? "+" : ""}${w.toFixed(2)}`);
      });
    }
    if (!lines.length) return;

    ctx.save();
    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.font = "600 12px Aptos, Trebuchet MS, sans-serif";
    const maxLineW = lines.reduce((m, line, i) => {
      ctx.font = i === 0 ? "700 12.5px Aptos, Trebuchet MS, sans-serif" : "500 11.5px Aptos, Trebuchet MS, sans-serif";
      return Math.max(m, ctx.measureText(line).width);
    }, 0);
    const boxW = Math.min(width - 24, maxLineW + 24);
    const boxH = 18 + lines.length * 17;
    const anchorX = hoverState.anchorX ?? width / 2;
    const anchorY = hoverState.anchorY ?? height / 2;
    const boxX = clamp(anchorX + 16, 12, width - boxW - 12);
    const boxY = anchorY + boxH + 18 <= height - 10 ? anchorY + 14 : Math.max(10, anchorY - boxH - 14);

    this.roundedRect(ctx, boxX, boxY, boxW, boxH, 10);
    ctx.fillStyle = "rgba(6, 16, 26, 0.94)";
    ctx.fill();
    ctx.strokeStyle = "rgba(127, 208, 255, 0.45)";
    ctx.lineWidth = 1.2;
    ctx.stroke();

    lines.forEach((line, idx) => {
      ctx.font = idx === 0 ? "700 12.5px Aptos, Trebuchet MS, sans-serif" : "500 11.5px Aptos, Trebuchet MS, sans-serif";
      ctx.fillStyle = idx === 0 ? "#ffcf66" : idx === 1 && hoverState.layer === "IN" ? "#edf6f9" : "#b4c7cf";
      ctx.fillText(line, boxX + 12, boxY + 19 + idx * 17);
    });
    ctx.restore();
  },

  drawTrainingChart(ctx, width, height, points, isSmart, isDqn) {
    const metricLabel = isDqn ? "Loss" : isSmart ? "Счет" : "Fitness";
    ctx.clearRect(0, 0, width, height);
    const gradient = ctx.createLinearGradient(0, 0, width, height);
    gradient.addColorStop(0, "#08131d");
    gradient.addColorStop(1, "#0f2231");
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, width, height);

    if (points.length < 2) {
      ctx.fillStyle = "#b4c7cf";
      ctx.textAlign = "center";
      ctx.font = "500 15px Aptos, Trebuchet MS, sans-serif";
      drawCanvasLines(ctx, ["После нескольких эпизодов", `здесь появятся счет, награда и ${metricLabel.toLowerCase()}.`], width / 2, height / 2 - 8, 20);
      ctx.textAlign = "start";
      return;
    }

    const scores = points.map((point) => point.score);
    const rewards = points.map((point) => point.reward);
    const fitnesses = points.map((point) => {
      const value = Number(isDqn ? point.loss ?? 0 : isSmart ? point.score ?? 0 : point.fitness ?? 0);
      return Number.isFinite(value) ? value : 0;
    });
    const finiteFitnesses = fitnesses.filter(Number.isFinite);
    const movingAverage = (series, windowSize) => series.map((_, index) => average(series.slice(Math.max(0, index - windowSize + 1), index + 1)));
    const scoreAverage20 = movingAverage(scores, 20);
    const mainMax = Math.max(1, ...scores, ...rewards, ...scoreAverage20);
    const mainMin = Math.min(-2, ...scores, ...rewards, ...scoreAverage20);
    const mainRange = Math.max(1, mainMax - mainMin);
    const fitnessLow = quantile(finiteFitnesses, 0.05);
    const fitnessHigh = quantile(finiteFitnesses, 0.95);
    const fitnessPad = Math.max(5, (fitnessHigh - fitnessLow) * 0.18);
    const fitnessMax = Math.max(0.1, fitnessHigh + fitnessPad);
    const fitnessMin = Math.min(0, fitnessLow - fitnessPad);
    const fitnessRange = Math.max(0.1, fitnessMax - fitnessMin);
    const padLeft = 56;
    const padRight = 52;
    const padTop = 38;
    const padBottom = 34;
    const chartWidth = width - padLeft - padRight;
    const chartHeight = height - padTop - padBottom;
    const mapMainY = (value) => padTop + chartHeight - ((value - mainMin) / mainRange) * chartHeight;
    const mapFitnessY = (value) => {
      const clipped = clamp(value, fitnessMin, fitnessMax);
      return padTop + chartHeight - ((clipped - fitnessMin) / fitnessRange) * chartHeight;
    };
    const episodeSeries = points.map((point, index) => {
      const value = Number(point?.episode);
      return Number.isFinite(value) ? value : (index + 1);
    });
    const minEpisode = Math.min(...episodeSeries);
    const maxEpisode = Math.max(...episodeSeries);
    const episodeRange = Math.max(1, maxEpisode - minEpisode);
    const mapX = (episodeValue) => padLeft + ((episodeValue - minEpisode) / episodeRange) * chartWidth;

    for (let index = 0; index <= 4; index += 1) {
      const value = mainMin + (mainRange / 4) * index;
      const y = mapMainY(value);
      ctx.beginPath();
      ctx.moveTo(padLeft, y);
      ctx.lineTo(width - padRight, y);
      ctx.strokeStyle = "rgba(255,255,255,0.08)";
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.fillStyle = "rgba(180,199,207,0.75)";
      ctx.font = "11px Aptos, Trebuchet MS, sans-serif";
      ctx.fillText(value.toFixed(1), 12, y + 4);
    }
    for (let index = 0; index <= 4; index += 1) {
      const value = fitnessMin + (fitnessRange / 4) * index;
      const y = mapFitnessY(value);
      ctx.fillStyle = "rgba(127,208,255,0.78)";
      ctx.font = "11px Aptos, Trebuchet MS, sans-serif";
      ctx.fillText(value.toFixed(2), width - padRight + 8, y + 4);
    }

    const drawSeries = (series, stroke, fillAlpha, mapY) => {
      const points2d = series.map((value, index) => ({ x: mapX(episodeSeries[index]), y: mapY(value) }));
      ctx.beginPath();
      ctx.moveTo(points2d[0].x, points2d[0].y);
      for (let index = 1; index < points2d.length; index += 1) {
        const prev = points2d[index - 1];
        const current = points2d[index];
        const controlX = (prev.x + current.x) / 2;
        ctx.quadraticCurveTo(controlX, prev.y, current.x, current.y);
      }
      ctx.strokeStyle = stroke;
      ctx.lineWidth = 2.4;
      ctx.stroke();
      ctx.beginPath();
      ctx.moveTo(points2d[0].x, chartHeight + padTop);
      ctx.lineTo(points2d[0].x, points2d[0].y);
      for (let index = 1; index < points2d.length; index += 1) {
        const prev = points2d[index - 1];
        const current = points2d[index];
        const controlX = (prev.x + current.x) / 2;
        ctx.quadraticCurveTo(controlX, prev.y, current.x, current.y);
      }
      ctx.lineTo(points2d[points2d.length - 1].x, chartHeight + padTop);
      ctx.closePath();
      ctx.fillStyle = `${stroke}${fillAlpha}`;
      ctx.fill();
    };

    const drawLineSeries = (series, stroke, mapY, lineWidth = 3.2) => {
      const points2d = series.map((value, index) => ({ x: mapX(episodeSeries[index]), y: mapY(value) }));
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(points2d[0].x, points2d[0].y);
      for (let index = 1; index < points2d.length; index += 1) {
        const prev = points2d[index - 1];
        const current = points2d[index];
        const controlX = (prev.x + current.x) / 2;
        ctx.quadraticCurveTo(controlX, prev.y, current.x, current.y);
      }
      ctx.strokeStyle = stroke;
      ctx.lineWidth = lineWidth;
      ctx.lineCap = "round";
      ctx.stroke();
      ctx.restore();
    };

    drawSeries(scores, "#5cd6ad", "18", mapMainY);
    drawSeries(rewards, "#ffb347", "12", mapMainY);
    drawSeries(fitnesses, "#7fd0ff", "0f", mapFitnessY);
    drawLineSeries(scoreAverage20, "#edf6f9", mapMainY);

    [["Счет", "#5cd6ad", padLeft], ["Награда", "#ffb347", padLeft + 84], ["Срд счет", "#edf6f9", padLeft + 190], [metricLabel, "#7fd0ff", width - padRight - 76]].forEach(([label, color, x]) => {
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.arc(x, 18, 5, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#edf6f9";
      ctx.font = "600 13px Aptos, Trebuchet MS, sans-serif";
      ctx.fillText(label, x + 12, 22);
    });
  }
};

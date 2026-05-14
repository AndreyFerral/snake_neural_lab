function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}
function escapeText(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
function average(values) {
  if (!values.length) { return 0; }
  return values.reduce((sum, value) => sum + value, 0) / values.length;
}
function quantile(values, ratio) {
  if (!values.length) { return 0; }
  const sorted = [...values].sort((a, b) => a - b);
  const index = clamp((sorted.length - 1) * ratio, 0, sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);
  if (lower === upper) { return sorted[lower]; }
  const fraction = index - lower;
  return sorted[lower] * (1 - fraction) + sorted[upper] * fraction;
}
function colorForActivation(value) {
  const normalized = clamp((value + 1) / 2, 0, 1);
  const hue = 32 + normalized * 128;
  const lightness = 28 + normalized * 42;
  return `hsl(${hue}, 78%, ${lightness}%)`;
}
function drawCanvasLines(ctx, lines, x, y, lineHeight) {
  lines.forEach((line, index) => {
    ctx.fillText(line, x, y + index * lineHeight);
  });
}
function formatReason(reason) {
  const map = {
    new_game: "Новая игра",
    step: "Шаг",
    apple: "Яблоко",
    victory: "Победа",
    looping:  "Ходит по кругу",
    missed_safe_adjacent_apple:  "Пропустила безопасное яблоко",
    available_food_ignored:  "Игнорирует доступную еду",
    trap_after_apple:  "Тупик после яблока",
    route_dead_end:  "Нет пути к хвосту",
    tail_path_lost:  "Потерян путь к хвосту",
    endgame_space_efficient:  "Экономный ход",
    endgame_space_cut:  "Срез пространства",
    endgame_pocket_fill:  "Заполняет карман",
    endgame_pocket_skip:  "Пропуск кармана",
    avoidable_collision:  "Избежимое столкновение",
    blocked:  "ход заблокирован",
    no_tail_after_apple:  "нет пути к хвосту",
    not_enough_space:  "мало места",
    safe:  "безопасно",
    none:  "нет",
    wall_collision:  "Столкновение со стеной",
    body_collision:  "Столкновение с телом",
    starvation:  "Слишком долго без еды",
    starvation_food_unreachable:  "Голод: еда недостижима",
    starvation_no_relayout_path:  "Голод: нет перестройки",
    starvation_food_available:  "Голод: еда доступна",
    starvation_safe_loop:  "Голод: безопасная петля",
    starvation_endgame_trap:  "Голод: эндшпильная ловушка",
    evolution_hunger_cut:  "Отсев: голод",
  };
  Object.assign(map, {
    route_follow: "На маршруте",
    route_deviation: "Сход с маршрута",
  });
  return map[reason] ?? reason ?? "Ожидание";
}
function formatCompactReason(reason) {
  const map = {
    new_game:  "Старт",
    step:  "Шаг",
    apple:  "Яблоко",
    victory:  "Победа",
    looping:  "Цикл",
    missed_safe_adjacent_apple:  "Пропуск яблока",
    available_food_ignored:  "Еда игнор",
    trap_after_apple:  "Тупик после яблока",
    route_dead_end:  "Нет хвоста",
    tail_path_lost:  "Потеря хвоста",
    endgame_space_efficient:  "Экономно",
    endgame_space_cut:  "Срез места",
    endgame_pocket_fill:  "Карман",
    endgame_pocket_skip:  "Пропуск кармана",
    avoidable_collision:  "Избежимое столкновение",
    wall_collision:  "Стена",
    body_collision:  "Тело",
    starvation:  "Голод",
    starvation_food_unreachable:  "Еда недоступна",
    starvation_no_relayout_path:  "Нет перестройки",
    starvation_food_available:  "Еда доступна",
    starvation_safe_loop:  "Петля",
    starvation_endgame_trap:  "Ловушка",
    evolution_hunger_cut:  "Отсев",
  };
  Object.assign(map, {
    route_follow:  "Маршрут",
    route_deviation:  "Сход",
  });
  return map[reason] ?? formatReason(reason);
}
function appleSticker() {
  return '<span class="step-sticker" title="яблоко">🍎</span>';
}
function formatStepEffectTag(reason) {
  const apple = appleSticker();
  const map = {
    looping:  "цикл",
    missed_safe_adjacent_apple:  `пропуск ${apple}`,
    available_food_ignored:  `${apple} не взята`,
    trap_after_apple:  `тупик после ${apple}`,
    route_dead_end:  "нет хвоста",
    tail_path_lost:  "потеря хвоста",
    evolution_hunger_cut:  "отсев: голод",
    endgame_space_efficient:  "экономно",
    endgame_space_cut:  "срез места",
    endgame_pocket_fill:  "карман",
    endgame_pocket_skip:  "пропуск кармана",
    avoidable_collision:  "избежимая смерть",
    wall_collision:  "стена",
    body_collision:  "тело",
    starvation:  "голод",
    starvation_food_unreachable:  "еда недоступна",
    starvation_no_relayout_path:  "нет перестройки",
    starvation_food_available:  "еда доступна",
    starvation_safe_loop:  "петля",
    starvation_endgame_trap:  "ловушка",
  };
  Object.assign(map, {
    route_follow:  "маршрут",
    route_deviation:  "сход",
  });
  return map[reason] ?? formatCompactReason(reason);
}
function formatAppleVerdictTag(verdict) {
  const map = {
    safe: "безопасно",
    blocked: "блок",
    no_tail_after_apple: "нет хвоста",
    not_enough_space: "мало места",
    none: "-",
  };
  return map[verdict] ?? formatStepEffectTag(verdict);
}
function formatSignalLabel(label) {
  const labels = {
    food_ahead:  "🍎↑",
    food_right:  "🍎→",
    food_left:  "🍎←",
    danger_forward:  "💀 ↑",
    danger_right:  "💀 →",
    danger_left:  "💀 ←",
    apple_step_forward:  "🍎+↑",
    apple_step_right:  "🍎+→",
    apple_step_left:  "🍎+←",
    forward_food_reachable:  "↑ food",
    forward_tail_reachable:  "↑ tail",
    forward_enough_space:  "↑ space",
    forward_food_tail_safe:  "↑ safe",
    left_food_reachable:  "← food",
    left_tail_reachable:  "← tail",
    left_enough_space:  "← space",
    left_food_tail_safe:  "← safe",
    right_food_reachable:  "→ food",
    right_tail_reachable:  "→ tail",
    right_enough_space:  "→ space",
    right_food_tail_safe:  "→ safe",
  };
  return labels[label] ?? label;
}
async function apiJson(path, options = {}) {
  const timeoutMs = Number(options.timeoutMs ?? 6000);
  const controller = new AbortController();
  const timeout = window.setTimeout(() => controller.abort(), timeoutMs);
  const { timeoutMs: _timeoutMs, headers: optionHeaders, ...fetchOptions } = options;
  const headers = { ...(optionHeaders ?? {}) };
  if (fetchOptions.body !== undefined && !Object.keys(headers).some((key) => key.toLowerCase() === "content-type")) {
    headers["Content-Type"] = "application/json";
  }
  const response = await fetch(path, {
    cache: "no-store",
    headers,
    ...fetchOptions,
    signal: controller.signal,
  }).finally(() => window.clearTimeout(timeout));
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `HTTP ${response.status}`);
  }
  return response.json();
}
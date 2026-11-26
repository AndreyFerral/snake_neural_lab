from __future__ import annotations

import numpy as np

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except Exception:  # pragma: no cover - fallback when numba is unavailable
    NUMBA_AVAILABLE = False

    def njit(*args, **kwargs):  # type: ignore[no-redef]
        def _decorator(fn):
            return fn
        return _decorator


@njit(cache=True)
def compute_signals_and_ego_numba(
    n: int,
    snake_xy: np.ndarray,
    direction: int,
    food_x: int,
    food_y: int,
    steps_without_food: int,
    hunger_limit: float,
    tail_track_xy: np.ndarray,
    win: int,
    want_ego: bool,
    blk: np.ndarray,
    visited: np.ndarray,
    queue: np.ndarray,
    parent: np.ndarray,
    ttl_buf: np.ndarray,
    reach_buf: np.ndarray,
    out_sig: np.ndarray,
    out_area: np.ndarray,
    out_ego: np.ndarray,
) -> bool:
    """Компилируемое ядро BFS для SnakeEnv: 34 сигнала + 3 area_ratio + (опц.) эго-карта (4, win, win).

    Точно воспроизводит логику чистого Python в SnakeEnv.get_extra_signals и get_egocentric_map
    с нулевыми аллокациями внутри шага. Возвращает flag `all_empty_reachable`.
    """
    board_cells = n * n
    snake_len = snake_xy.shape[0]
    head_x = int(snake_xy[0, 0])
    head_y = int(snake_xy[0, 1])
    tail_x = int(snake_xy[snake_len - 1, 0])
    tail_y = int(snake_xy[snake_len - 1, 1])

    # Направление еды и хвоста в относительной системе координат
    dx = food_x - head_x
    dy = food_y - head_y
    tx = tail_x - head_x
    ty = tail_y - head_y
    if direction == 0:
        fwd = -dy
        rgt = dx
        t_fwd = -ty
        t_rgt = tx
    elif direction == 1:
        fwd = dx
        rgt = dy
        t_fwd = tx
        t_rgt = ty
    elif direction == 2:
        fwd = dy
        rgt = -dx
        t_fwd = ty
        t_rgt = -tx
    else:
        fwd = -dx
        rgt = -dy
        t_fwd = -tx
        t_rgt = -ty

    for i in range(board_cells):
        blk[i] = 0
        visited[i] = 0

    occupied_no_tail = 0
    for i in range(snake_len - 1):
        fidx = int(snake_xy[i, 1]) * n + int(snake_xy[i, 0])
        if blk[fidx] == 0:
            blk[fidx] = 1
            occupied_no_tail += 1

    head_idx = head_y * n + head_x
    food_idx = food_y * n + food_x
    tail_idx = tail_y * n + tail_x
    empty_no_tail = board_cells - occupied_no_tail

    # 1. Объединённый BFS от головы: food_reachable и current_comps
    saved_tail_cur = blk[tail_idx]
    tail_was_free = saved_tail_cur == 0
    blk[tail_idx] = 1
    total_empty_cur = empty_no_tail - (1 if tail_was_free else 0)

    v_stamp = 1
    current_comps = 0
    visited_empty_cur = 0

    if snake_len == 1:
        food_reachable = 1.0
        for start in range(board_cells):
            if blk[start] == 0 and visited[start] != v_stamp:
                current_comps += 1
                visited[start] = v_stamp
                q_head = 0
                q_tail = 1
                queue[0] = start
                while q_head < q_tail:
                    curr = queue[q_head]
                    q_head += 1
                    cx = curr % n
                    cy = curr // n
                    if cy > 0:
                        nxt = curr - n
                        if blk[nxt] == 0 and visited[nxt] != v_stamp:
                            visited[nxt] = v_stamp
                            queue[q_tail] = nxt
                            q_tail += 1
                    if cx + 1 < n:
                        nxt = curr + 1
                        if blk[nxt] == 0 and visited[nxt] != v_stamp:
                            visited[nxt] = v_stamp
                            queue[q_tail] = nxt
                            q_tail += 1
                    if cy + 1 < n:
                        nxt = curr + n
                        if blk[nxt] == 0 and visited[nxt] != v_stamp:
                            visited[nxt] = v_stamp
                            queue[q_tail] = nxt
                            q_tail += 1
                    if cx > 0:
                        nxt = curr - 1
                        if blk[nxt] == 0 and visited[nxt] != v_stamp:
                            visited[nxt] = v_stamp
                            queue[q_tail] = nxt
                            q_tail += 1
                visited_empty_cur += q_tail
                if visited_empty_cur >= total_empty_cur:
                    break
    else:
        # 4 соседа головы в порядке DIRECTIONS: UP, RIGHT, DOWN, LEFT
        for d_i in range(4):
            if d_i == 0:
                if head_y <= 0:
                    continue
                nb = head_idx - n
            elif d_i == 1:
                if head_x + 1 >= n:
                    continue
                nb = head_idx + 1
            elif d_i == 2:
                if head_y + 1 >= n:
                    continue
                nb = head_idx + n
            else:
                if head_x <= 0:
                    continue
                nb = head_idx - 1

            if blk[nb] == 0 and visited[nb] != v_stamp:
                current_comps += 1
                visited[nb] = v_stamp
                q_head = 0
                q_tail = 1
                queue[0] = nb
                while q_head < q_tail:
                    curr = queue[q_head]
                    q_head += 1
                    cx = curr % n
                    cy = curr // n
                    if cy > 0:
                        nxt = curr - n
                        if blk[nxt] == 0 and visited[nxt] != v_stamp:
                            visited[nxt] = v_stamp
                            queue[q_tail] = nxt
                            q_tail += 1
                    if cx + 1 < n:
                        nxt = curr + 1
                        if blk[nxt] == 0 and visited[nxt] != v_stamp:
                            visited[nxt] = v_stamp
                            queue[q_tail] = nxt
                            q_tail += 1
                    if cy + 1 < n:
                        nxt = curr + n
                        if blk[nxt] == 0 and visited[nxt] != v_stamp:
                            visited[nxt] = v_stamp
                            queue[q_tail] = nxt
                            q_tail += 1
                    if cx > 0:
                        nxt = curr - 1
                        if blk[nxt] == 0 and visited[nxt] != v_stamp:
                            visited[nxt] = v_stamp
                            queue[q_tail] = nxt
                            q_tail += 1
                visited_empty_cur += q_tail

        if head_idx == food_idx:
            food_reachable = 1.0
        elif head_x == tail_x and head_y == tail_y:
            food_reachable = 0.0
        else:
            food_reachable = 1.0 if visited[food_idx] == v_stamp else 0.0

        if visited_empty_cur < total_empty_cur:
            for start in range(board_cells):
                if blk[start] == 0 and visited[start] != v_stamp:
                    current_comps += 1
                    visited[start] = v_stamp
                    q_head = 0
                    q_tail = 1
                    queue[0] = start
                    while q_head < q_tail:
                        curr = queue[q_head]
                        q_head += 1
                        cx = curr % n
                        cy = curr // n
                        if cy > 0:
                            nxt = curr - n
                            if blk[nxt] == 0 and visited[nxt] != v_stamp:
                                visited[nxt] = v_stamp
                                queue[q_tail] = nxt
                                q_tail += 1
                        if cx + 1 < n:
                            nxt = curr + 1
                            if blk[nxt] == 0 and visited[nxt] != v_stamp:
                                visited[nxt] = v_stamp
                                queue[q_tail] = nxt
                                q_tail += 1
                        if cy + 1 < n:
                            nxt = curr + n
                            if blk[nxt] == 0 and visited[nxt] != v_stamp:
                                visited[nxt] = v_stamp
                                queue[q_tail] = nxt
                                q_tail += 1
                        if cx > 0:
                            nxt = curr - 1
                            if blk[nxt] == 0 and visited[nxt] != v_stamp:
                                visited[nxt] = v_stamp
                                queue[q_tail] = nxt
                                q_tail += 1
                    visited_empty_cur += q_tail
                    if visited_empty_cur >= total_empty_cur:
                        break

    blk[tail_idx] = saved_tail_cur
    board_split = 1.0 if current_comps > 1 else 0.0
    all_empty_reachable = (
        current_comps == 1
        and snake_len >= 2
        and total_empty_cur > 0
        and visited_empty_cur == total_empty_cur
    )

    endgame = 1.0 if (float(snake_len) / float(board_cells)) >= 0.75 else 0.0
    h_lim = hunger_limit if hunger_limit > 1.0 else 1.0
    hunger_ratio = float(steps_without_food) / h_lim
    hunger_high = 1.0 if hunger_ratio >= 0.65 else 0.0
    hunger_critical = 1.0 if hunger_ratio >= 0.85 else 0.0

    # Предвычисляем 3 относительных хода (0=fwd, 1=left, 2=right)
    nx_arr = np.empty(3, dtype=np.int32)
    ny_arr = np.empty(3, dtype=np.int32)
    nidx_arr = np.empty(3, dtype=np.int32)
    eat_arr = np.zeros(3, dtype=np.int32)
    dng_arr = np.zeros(3, dtype=np.int32)

    for act in range(3):
        if act == 0:
            abs_dir = direction
        elif act == 1:
            abs_dir = (direction - 1) % 4
        else:
            abs_dir = (direction + 1) % 4

        if abs_dir == 0:
            nx = head_x
            ny = head_y - 1
        elif abs_dir == 1:
            nx = head_x + 1
            ny = head_y
        elif abs_dir == 2:
            nx = head_x
            ny = head_y + 1
        else:
            nx = head_x - 1
            ny = head_y

        nx_arr[act] = nx
        ny_arr[act] = ny
        will_eat = (nx == food_x and ny == food_y)
        eat_arr[act] = 1 if will_eat else 0

        if nx < 0 or nx >= n or ny < 0 or ny >= n:
            dng_arr[act] = 1
            nidx_arr[act] = -1
        else:
            next_idx = ny * n + nx
            nidx_arr[act] = next_idx
            is_dng = (blk[next_idx] == 1) or (will_eat and next_idx == tail_idx)
            dng_arr[act] = 1 if is_dng else 0

    tail_blocked_0 = 0.0
    tail_blocked_1 = 0.0
    tail_blocked_2 = 0.0
    part_0 = 0.0
    part_1 = 0.0
    part_2 = 0.0
    ps_0 = 0.0
    ps_1 = 0.0
    ps_2 = 0.0
    pm_0 = 0.0
    pm_1 = 0.0
    pm_2 = 0.0
    out_area[0] = 0.0
    out_area[1] = 0.0
    out_area[2] = 0.0

    # 2. Объединённый BFS (с O(1) fast-path) на каждое безопасное действие
    for act in range(3):
        nx = int(nx_arr[act])
        ny = int(ny_arr[act])
        next_idx = int(nidx_arr[act])
        will_eat = eat_arr[act] == 1
        is_danger = dng_arr[act] == 1

        if is_danger:
            tb_v = 1.0
            pt_v = 0.0 if will_eat else 1.0
            ps_v = 1.0
            pm_v = 0.0
            ar_v = 0.0
        else:
            if will_eat:
                saved_tail = blk[tail_idx]
                blk[tail_idx] = 1
                tail_target_idx = tail_idx
                new_len = snake_len + 1
                empty_after = empty_no_tail - (1 if saved_tail == 0 else 0) - 1
            else:
                saved_tail = 0
                if snake_len >= 2:
                    tail_target_idx = int(snake_xy[snake_len - 2, 1]) * n + int(snake_xy[snake_len - 2, 0])
                else:
                    tail_target_idx = next_idx
                new_len = snake_len
                empty_after = empty_no_tail - 1

            saved_nh = blk[next_idx]
            blk[next_idx] = 1

            # O(1) fast-path для не-точки сочленения при связном поле
            fast_ok = False
            if current_comps == 1 and empty_after > 0:
                tail_connected = will_eat or (next_idx == tail_idx)
                if not tail_connected:
                    if tail_y > 0 and blk[tail_idx - n] == 0:
                        tail_connected = True
                    elif tail_x + 1 < n and blk[tail_idx + 1] == 0:
                        tail_connected = True
                    elif tail_y + 1 < n and blk[tail_idx + n] == 0:
                        tail_connected = True
                    elif tail_x > 0 and blk[tail_idx - 1] == 0:
                        tail_connected = True

                if tail_connected:
                    e_up = ny > 0 and blk[next_idx - n] == 0
                    e_dn = ny + 1 < n and blk[next_idx + n] == 0
                    e_lt = nx > 0 and blk[next_idx - 1] == 0
                    e_rt = nx + 1 < n and blk[next_idx + 1] == 0
                    k_free = (1 if e_up else 0) + (1 if e_dn else 0) + (1 if e_lt else 0) + (1 if e_rt else 0)
                    if k_free == 1:
                        fast_ok = True
                    elif k_free == 2:
                        if e_up and e_rt:
                            fast_ok = blk[next_idx - n + 1] == 0
                        elif e_rt and e_dn:
                            fast_ok = blk[next_idx + n + 1] == 0
                        elif e_dn and e_lt:
                            fast_ok = blk[next_idx + n - 1] == 0
                        elif e_lt and e_up:
                            fast_ok = blk[next_idx - n - 1] == 0
                    elif k_free == 3:
                        if not e_lt:
                            fast_ok = (blk[next_idx - n + 1] == 0) and (blk[next_idx + n + 1] == 0)
                        elif not e_rt:
                            fast_ok = (blk[next_idx - n - 1] == 0) and (blk[next_idx + n - 1] == 0)
                        elif not e_up:
                            fast_ok = (blk[next_idx + n - 1] == 0) and (blk[next_idx + n + 1] == 0)
                        else:
                            fast_ok = (blk[next_idx - n - 1] == 0) and (blk[next_idx - n + 1] == 0)
                    elif k_free == 4:
                        diags = (
                            (1 if blk[next_idx - n - 1] == 0 else 0)
                            + (1 if blk[next_idx - n + 1] == 0 else 0)
                            + (1 if blk[next_idx + n - 1] == 0 else 0)
                            + (1 if blk[next_idx + n + 1] == 0 else 0)
                        )
                        fast_ok = diags >= 3

            if fast_ok:
                free_cells_ratio = board_cells - new_len
                if free_cells_ratio < 1:
                    free_cells_ratio = 1
                ar_v = float(1 + empty_after) / float(free_cells_ratio)
                if ar_v > 1.0:
                    ar_v = 1.0
                pt_v = 0.0
                if next_idx == tail_target_idx or blk[tail_target_idx] == 0:
                    has_tail_path = True
                else:
                    tt_x = tail_target_idx % n
                    tt_y = tail_target_idx // n
                    has_tail_path = False
                    if tt_y > 0 and (tail_target_idx - n == next_idx or blk[tail_target_idx - n] == 0):
                        has_tail_path = True
                    elif tt_x + 1 < n and (tail_target_idx + 1 == next_idx or blk[tail_target_idx + 1] == 0):
                        has_tail_path = True
                    elif tt_y + 1 < n and (tail_target_idx + n == next_idx or blk[tail_target_idx + n] == 0):
                        has_tail_path = True
                    elif tt_x > 0 and (tail_target_idx - 1 == next_idx or blk[tail_target_idx - 1] == 0):
                        has_tail_path = True
                tb_v = 0.0 if has_tail_path else 1.0
                ps_v = 0.0
                pm_v = 0.0
            else:
                v_stamp += 1
                next_comps = 0
                reach_nbs = 0
                for d_i in range(4):
                    if d_i == 0:
                        if ny <= 0:
                            continue
                        nb = next_idx - n
                    elif d_i == 1:
                        if nx + 1 >= n:
                            continue
                        nb = next_idx + 1
                    elif d_i == 2:
                        if ny + 1 >= n:
                            continue
                        nb = next_idx + n
                    else:
                        if nx <= 0:
                            continue
                        nb = next_idx - 1

                    if blk[nb] == 0 and visited[nb] != v_stamp:
                        next_comps += 1
                        visited[nb] = v_stamp
                        q_head = 0
                        q_tail = 1
                        queue[0] = nb
                        while q_head < q_tail:
                            curr = queue[q_head]
                            q_head += 1
                            cx = curr % n
                            cy = curr // n
                            if cy > 0:
                                nxt = curr - n
                                if blk[nxt] == 0 and visited[nxt] != v_stamp:
                                    visited[nxt] = v_stamp
                                    queue[q_tail] = nxt
                                    q_tail += 1
                            if cx + 1 < n:
                                nxt = curr + 1
                                if blk[nxt] == 0 and visited[nxt] != v_stamp:
                                    visited[nxt] = v_stamp
                                    queue[q_tail] = nxt
                                    q_tail += 1
                            if cy + 1 < n:
                                nxt = curr + n
                                if blk[nxt] == 0 and visited[nxt] != v_stamp:
                                    visited[nxt] = v_stamp
                                    queue[q_tail] = nxt
                                    q_tail += 1
                            if cx > 0:
                                nxt = curr - 1
                                if blk[nxt] == 0 and visited[nxt] != v_stamp:
                                    visited[nxt] = v_stamp
                                    queue[q_tail] = nxt
                                    q_tail += 1
                        reach_nbs += q_tail

                free_cells_ratio = board_cells - new_len
                if free_cells_ratio < 1:
                    free_cells_ratio = 1
                ar_v = float(1 + reach_nbs) / float(free_cells_ratio)
                if ar_v > 1.0:
                    ar_v = 1.0
                if will_eat:
                    pt_v = 0.0
                else:
                    pt_v = 0.0 if visited[food_idx] == v_stamp else 1.0

                if next_idx == tail_target_idx:
                    has_tail_path = True
                elif blk[tail_target_idx] == 0:
                    has_tail_path = visited[tail_target_idx] == v_stamp
                else:
                    tt_x = tail_target_idx % n
                    tt_y = tail_target_idx // n
                    has_tail_path = False
                    if tt_y > 0 and (tail_target_idx - n == next_idx or visited[tail_target_idx - n] == v_stamp):
                        has_tail_path = True
                    elif tt_x + 1 < n and (tail_target_idx + 1 == next_idx or visited[tail_target_idx + 1] == v_stamp):
                        has_tail_path = True
                    elif tt_y + 1 < n and (tail_target_idx + n == next_idx or visited[tail_target_idx + n] == v_stamp):
                        has_tail_path = True
                    elif tt_x > 0 and (tail_target_idx - 1 == next_idx or visited[tail_target_idx - 1] == v_stamp):
                        has_tail_path = True
                tb_v = 0.0 if has_tail_path else 1.0

                if reach_nbs < empty_after:
                    visited_total = reach_nbs
                    for start in range(board_cells):
                        if blk[start] == 0 and visited[start] != v_stamp:
                            next_comps += 1
                            visited[start] = v_stamp
                            q_head = 0
                            q_tail = 1
                            queue[0] = start
                            while q_head < q_tail:
                                curr = queue[q_head]
                                q_head += 1
                                cx = curr % n
                                cy = curr // n
                                if cy > 0:
                                    nxt = curr - n
                                    if blk[nxt] == 0 and visited[nxt] != v_stamp:
                                        visited[nxt] = v_stamp
                                        queue[q_tail] = nxt
                                        q_tail += 1
                                if cx + 1 < n:
                                    nxt = curr + 1
                                    if blk[nxt] == 0 and visited[nxt] != v_stamp:
                                        visited[nxt] = v_stamp
                                        queue[q_tail] = nxt
                                        q_tail += 1
                                if cy + 1 < n:
                                    nxt = curr + n
                                    if blk[nxt] == 0 and visited[nxt] != v_stamp:
                                        visited[nxt] = v_stamp
                                        queue[q_tail] = nxt
                                        q_tail += 1
                                if cx > 0:
                                    nxt = curr - 1
                                    if blk[nxt] == 0 and visited[nxt] != v_stamp:
                                        visited[nxt] = v_stamp
                                        queue[q_tail] = nxt
                                        q_tail += 1
                            visited_total += q_tail
                            if visited_total >= empty_after:
                                break
                ps_v = 1.0 if next_comps > current_comps else 0.0
                pm_v = 1.0 if next_comps < current_comps else 0.0

            blk[next_idx] = saved_nh
            if will_eat:
                blk[tail_idx] = saved_tail

        out_area[act] = np.float32(ar_v)
        if act == 0:
            tail_blocked_0 = tb_v
            part_0 = pt_v
            ps_0 = ps_v
            pm_0 = pm_v
        elif act == 1:
            tail_blocked_1 = tb_v
            part_1 = pt_v
            ps_1 = ps_v
            pm_1 = pm_v
        else:
            tail_blocked_2 = tb_v
            part_2 = pt_v
            ps_2 = ps_v
            pm_2 = pm_v

    # 3. Ленивый расчёт tail_close (BFS кратчайшего пути к хвосту с блокировкой snake[1:-1])
    tc_0 = 0.0
    tc_1 = 0.0
    tc_2 = 0.0
    tt_len = tail_track_xy.shape[0]
    need_tail_close = False
    hit_tt = np.zeros(3, dtype=np.int32)
    for act in range(3):
        nx = int(nx_arr[act])
        ny = int(ny_arr[act])
        if nx == tail_x and ny == tail_y:
            hit_tt[act] = 1
            need_tail_close = True
        else:
            for ti in range(tt_len):
                if nx == int(tail_track_xy[ti, 0]) and ny == int(tail_track_xy[ti, 1]):
                    hit_tt[act] = 1
                    need_tail_close = True
                    break

    if need_tail_close and head_idx != tail_idx:
        # В _shortest_path_to_tail блокируется snake[1:-1], т.е. blk без головы (если голова не дублируется в [1:-1])
        # Валидный способ без модификации blk: используем отдельный флаг через v_stamp
        head_dup_in_mid = False
        for i in range(1, snake_len - 1):
            if int(snake_xy[i, 0]) == head_x and int(snake_xy[i, 1]) == head_y:
                head_dup_in_mid = True
                break
        saved_head_blk = blk[head_idx]
        if not head_dup_in_mid:
            blk[head_idx] = 0
        # Хвост тоже должен быть открыт (он уже не в snake[:-1], но на всякий случай сохраняем)
        saved_tail_blk2 = blk[tail_idx]
        tail_dup_in_mid = False
        for i in range(1, snake_len - 1):
            if int(snake_xy[i, 0]) == tail_x and int(snake_xy[i, 1]) == tail_y:
                tail_dup_in_mid = True
                break
        if not tail_dup_in_mid:
            blk[tail_idx] = 0

        v_stamp += 1
        visited[head_idx] = v_stamp
        parent[head_idx] = -1
        q_head = 0
        q_tail = 1
        queue[0] = head_idx
        found_t = False
        while q_head < q_tail and not found_t:
            curr = queue[q_head]
            q_head += 1
            cx = curr % n
            cy = curr // n
            for d_i in range(4):
                if d_i == 0:
                    if cy <= 0:
                        continue
                    nxt = curr - n
                elif d_i == 1:
                    if cx + 1 >= n:
                        continue
                    nxt = curr + 1
                elif d_i == 2:
                    if cy + 1 >= n:
                        continue
                    nxt = curr + n
                else:
                    if cx <= 0:
                        continue
                    nxt = curr - 1
                if visited[nxt] != v_stamp and blk[nxt] == 0:
                    visited[nxt] = v_stamp
                    parent[nxt] = curr
                    if nxt == tail_idx:
                        found_t = True
                        break
                    queue[q_tail] = nxt
                    q_tail += 1

        blk[head_idx] = saved_head_blk
        blk[tail_idx] = saved_tail_blk2

        if found_t:
            plen = 0
            cur = tail_idx
            first_step = -1
            while cur != head_idx and cur != -1:
                first_step = cur
                cur = parent[cur]
                plen += 1
            if 0 < plen < 3 and first_step != -1:
                p1x = first_step % n
                p1y = first_step // n
                pdx = p1x - head_x
                pdy = p1y - head_y
                if pdx == 0 and pdy == -1:
                    path_dir = 0
                elif pdx == 1 and pdy == 0:
                    path_dir = 1
                elif pdx == 0 and pdy == 1:
                    path_dir = 2
                else:
                    path_dir = 3
                for act in range(3):
                    if hit_tt[act] == 0:
                        continue
                    if act == 0:
                        abs_dir = direction
                    elif act == 1:
                        abs_dir = (direction - 1) % 4
                    else:
                        abs_dir = (direction + 1) % 4
                    if abs_dir == path_dir:
                        has_safe_alt = False
                        for alt in range(3):
                            if alt != act and dng_arr[alt] == 0:
                                has_safe_alt = True
                                break
                        if has_safe_alt:
                            if act == 0:
                                tc_0 = 1.0
                            elif act == 1:
                                tc_1 = 1.0
                            else:
                                tc_2 = 1.0

    # Заполняем 34 выходных сигнала
    out_sig[0] = 1.0 if fwd > 0 else 0.0
    out_sig[1] = 1.0 if rgt < 0 else 0.0
    out_sig[2] = 1.0 if rgt > 0 else 0.0
    out_sig[3] = 1.0 if fwd < 0 else 0.0
    out_sig[4] = float(eat_arr[0])
    out_sig[5] = float(eat_arr[1])
    out_sig[6] = float(eat_arr[2])
    out_sig[7] = food_reachable
    out_sig[8] = float(dng_arr[0])
    out_sig[9] = float(dng_arr[1])
    out_sig[10] = float(dng_arr[2])
    out_sig[11] = 1.0 if t_fwd > 0 else 0.0
    out_sig[12] = 1.0 if t_rgt < 0 else 0.0
    out_sig[13] = 1.0 if t_rgt > 0 else 0.0
    out_sig[14] = 1.0 if t_fwd < 0 else 0.0
    out_sig[15] = tail_blocked_0
    out_sig[16] = tail_blocked_1
    out_sig[17] = tail_blocked_2
    out_sig[18] = tc_0
    out_sig[19] = tc_1
    out_sig[20] = tc_2
    out_sig[21] = part_0
    out_sig[22] = part_1
    out_sig[23] = part_2
    out_sig[24] = endgame
    out_sig[25] = ps_0
    out_sig[26] = ps_1
    out_sig[27] = ps_2
    out_sig[28] = hunger_high
    out_sig[29] = hunger_critical
    out_sig[30] = board_split
    out_sig[31] = pm_0
    out_sig[32] = pm_1
    out_sig[33] = pm_2

    # 4. Опционально строим эго-карту (4, win, win) за тот же проход, переиспользуя blk!
    if want_ego:
        inv_len = 1.0 / float(snake_len if snake_len > 0 else 1)
        for i in range(board_cells):
            ttl_buf[i] = 0.0
            reach_buf[i] = 0.0
        for i in range(snake_len):
            fidx = int(snake_xy[i, 1]) * n + int(snake_xy[i, 0])
            ttl_buf[fidx] = np.float32(float(snake_len - i) * inv_len)

        if all_empty_reachable:
            for i in range(board_cells):
                if blk[i] == 0:
                    reach_buf[i] = 1.0
            # Кончик хвоста достижим, только если рядом с ним есть свободная клетка или голова
            tail_reach = False
            if tail_y > 0 and (tail_idx - n == head_idx or blk[tail_idx - n] == 0):
                tail_reach = True
            elif tail_x + 1 < n and (tail_idx + 1 == head_idx or blk[tail_idx + 1] == 0):
                tail_reach = True
            elif tail_y + 1 < n and (tail_idx + n == head_idx or blk[tail_idx + n] == 0):
                tail_reach = True
            elif tail_x > 0 and (tail_idx - 1 == head_idx or blk[tail_idx - 1] == 0):
                tail_reach = True
            if not tail_reach:
                reach_buf[tail_idx] = 0.0
        else:
            v_stamp += 1
            visited[head_idx] = v_stamp
            if blk[head_idx] == 0:
                reach_buf[head_idx] = 1.0
            q_head = 0
            q_tail = 1
            queue[0] = head_idx
            while q_head < q_tail:
                curr = queue[q_head]
                q_head += 1
                cx = curr % n
                cy = curr // n
                if cy > 0:
                    nxt = curr - n
                    if visited[nxt] != v_stamp and blk[nxt] == 0:
                        visited[nxt] = v_stamp
                        reach_buf[nxt] = 1.0
                        queue[q_tail] = nxt
                        q_tail += 1
                if cx + 1 < n:
                    nxt = curr + 1
                    if visited[nxt] != v_stamp and blk[nxt] == 0:
                        visited[nxt] = v_stamp
                        reach_buf[nxt] = 1.0
                        queue[q_tail] = nxt
                        q_tail += 1
                if cy + 1 < n:
                    nxt = curr + n
                    if visited[nxt] != v_stamp and blk[nxt] == 0:
                        visited[nxt] = v_stamp
                        reach_buf[nxt] = 1.0
                        queue[q_tail] = nxt
                        q_tail += 1
                if cx > 0:
                    nxt = curr - 1
                    if visited[nxt] != v_stamp and blk[nxt] == 0:
                        visited[nxt] = v_stamp
                        reach_buf[nxt] = 1.0
                        queue[q_tail] = nxt
                        q_tail += 1

        # Ориентация окна (DIRECTIONS: 0=up, 1=right, 2=down, 3=left)
        if direction == 0:
            fx, fy, rx, ry = 0, -1, 1, 0
        elif direction == 1:
            fx, fy, rx, ry = 1, 0, 0, 1
        elif direction == 2:
            fx, fy, rx, ry = 0, 1, -1, 0
        else:
            fx, fy, rx, ry = -1, 0, 0, -1

        radius = win // 2
        for row in range(win):
            f_off = radius - row
            for col in range(win):
                r_off = col - radius
                wx = head_x + f_off * fx + r_off * rx
                wy = head_y + f_off * fy + r_off * ry
                out_ego[2, row, col] = 0.0
                if 0 <= wx < n and 0 <= wy < n:
                    fidx = wy * n + wx
                    out_ego[0, row, col] = 0.0
                    out_ego[1, row, col] = ttl_buf[fidx]
                    out_ego[3, row, col] = reach_buf[fidx]
                else:
                    out_ego[0, row, col] = 1.0
                    out_ego[1, row, col] = 0.0
                    out_ego[3, row, col] = 0.0

        f_row = radius - (dx * fx + dy * fy)
        f_col = radius + (dx * rx + dy * ry)
        if 0 <= f_row < win and 0 <= f_col < win:
            out_ego[2, f_row, f_col] = 1.0

    return all_empty_reachable


_EMPTY_TAIL_TRACK = np.empty((0, 2), dtype=np.int32)

import json
import gzip
import argparse
import sys
from pathlib import Path
from collections import deque

# Add project root to path so we can import project files if needed
sys.path.append(str(Path(__file__).resolve().parent.parent))

def bfs_reachable_space(board_size, snake, start):
    snake_set = set((p[0], p[1]) for p in snake)
    visited = set([start])
    queue = deque([start])
    
    while queue:
        cx, cy = queue.popleft()
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < board_size and 0 <= ny < board_size:
                if (nx, ny) not in snake_set and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    return len(visited) - 1

def get_empty_connected_components(board_size, snake):
    snake_set = set((p[0], p[1]) for p in snake)
    visited = set()
    components = 0
    
    for y in range(board_size):
        for x in range(board_size):
            if (x, y) not in snake_set and (x, y) not in visited:
                components += 1
                queue = deque([(x, y)])
                visited.add((x, y))
                while queue:
                    cx, cy = queue.popleft()
                    for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < board_size and 0 <= ny < board_size:
                            if (nx, ny) not in snake_set and (nx, ny) not in visited:
                                visited.add((nx, ny))
                                queue.append((nx, ny))
    return components

def is_food_reachable_bfs(board_size, snake, food):
    if not snake:
        return False
    head = tuple(snake[0])
    target = tuple(food)
    if head == target:
        return True
    snake_set = set(tuple(p) for p in snake[1:])
    visited = {head}
    queue = deque([head])
    while queue:
        cx, cy = queue.popleft()
        if (cx, cy) == target:
            return True
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nx, ny = cx + dx, cy + dy
            if 0 <= nx < board_size and 0 <= ny < board_size:
                if (nx, ny) not in snake_set and (nx, ny) not in visited:
                    visited.add((nx, ny))
                    queue.append((nx, ny))
    return False

def analyze_replay_file(file_path):
    if file_path.suffix == ".gz":
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
    score = data.get("final_score")
    reason = data.get("final_reason")
    steps = data.get("steps", [])
    board_size = data.get("board_size", 10)
    agent_kind = data.get("agent_kind", "unknown")
    
    analysis = {
        "filename": file_path.name,
        "agent": agent_kind,
        "score": score,
        "reason": reason,
        "total_steps": len(steps),
        "avg_compactness": 0.0,
        "loops_detected": 0,
        "avg_distance_to_food": 0.0,
        "avg_final_reachable_ratio": 1.0,
        "partition_ratio": 0.0,
        "max_components": 1,
        "partition_start_score": None,
        "trapped_events": 0,
        "food_reachable_ratio": 1.0,
        "classification": "Неизвестно"
    }
    
    if not steps:
        return analysis
        
    coiling_compactness = []
    final_steps_reachable_ratio = []
    dist_sum = 0
    dist_count = 0
    state_history = []
    partition_steps = 0
    food_reachable_steps = 0
    
    for i, step_data in enumerate(steps):
        snake = step_data.get("snake", [])
        food = step_data.get("food", [0, 0])
        if not snake:
            continue
            
        head = tuple(snake[0])
        
        # Manhattan distance to food
        manhattan_dist = abs(head[0] - food[0]) + abs(head[1] - food[1])
        dist_sum += manhattan_dist
        dist_count += 1
        
        # Loop detection
        state_key = (head, tuple(food), len(snake))
        if state_key in state_history[-30:]:
            analysis["loops_detected"] += 1
        state_history.append(state_key)
        
        # Connected components (partitioning)
        components = get_empty_connected_components(board_size, snake)
        if components > 1:
            partition_steps += 1
            analysis["max_components"] = max(analysis["max_components"], components)
            if analysis["partition_start_score"] is None:
                analysis["partition_start_score"] = len(snake) - 3
                
        # Coiling compactness
        adj_segments = 0
        snake_set = set(tuple(p) for p in snake)
        for segment in snake:
            x, y = segment
            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                if (x + dx, y + dy) in snake_set:
                    adj_segments += 1
        compactness = (adj_segments / 2.0 - (len(snake) - 1)) / max(1, len(snake))
        coiling_compactness.append(compactness)
        
        # Reachable food check
        if is_food_reachable_bfs(board_size, snake, food):
            food_reachable_steps += 1
            
        # Reachable ratio in final 50 steps
        if len(steps) - i <= 50:
            reachable = bfs_reachable_space(board_size, snake, head)
            free_cells = (board_size * board_size) - len(snake)
            ratio = reachable / max(1, free_cells)
            final_steps_reachable_ratio.append(ratio)
            if ratio < 0.3 and free_cells > 5:
                analysis["trapped_events"] += 1
                
    if dist_count > 0:
        analysis["avg_distance_to_food"] = dist_sum / dist_count
    if coiling_compactness:
        analysis["avg_compactness"] = sum(coiling_compactness) / len(coiling_compactness)
    if final_steps_reachable_ratio:
        analysis["avg_final_reachable_ratio"] = sum(final_steps_reachable_ratio) / len(final_steps_reachable_ratio)
        
    analysis["partition_ratio"] = partition_steps / len(steps)
    analysis["food_reachable_ratio"] = food_reachable_steps / len(steps)
    
    # Classification logic
    if reason == "victory":
        analysis["classification"] = "Победа"
    elif reason == "wall_collision":
        analysis["classification"] = "Смерть: Стена"
    elif reason == "body_collision":
        analysis["classification"] = "Смерть: Тело (Хвост)"
    elif reason == "starvation":
        # If food was reachable in at least 40% of the steps, it starved by avoiding food (Reward Trap)
        if analysis["food_reachable_ratio"] > 0.4:
            analysis["classification"] = "Ловушка наград"
        else:
            analysis["classification"] = "Заперта в кармане"
    else:
        analysis["classification"] = reason
        
    return analysis

def main():
    parser = argparse.ArgumentParser(description="Универсальный инструмент для анализа реплеев.")
    parser.add_argument("file", nargs="?", help="Путь к конкретному файлу реплея. Если не указан, анализирует всю папку.")
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent
    replays_dir = project_root / ".runtime" / "replays"
    
    if args.file:
        file_path = Path(args.file)
        if not file_path.is_file():
            print(f"[ОШИБКА] Файл не найден: {file_path}")
            sys.exit(1)
            
        res = analyze_replay_file(file_path)
        print("=" * 60)
        print(f"АНАЛИЗ ФАЙЛА: {res['filename']}")
        print(f"Агент: {res['agent']} | Счет: {res['score']} | Результат: {res['reason']}")
        print(f"Классификация: {res['classification']}")
        print(f"Всего шагов: {res['total_steps']}")
        print("-" * 60)
        print(f"Доля шагов с доступной едой: {res['food_reachable_ratio']*100:.1f}%")
        print(f"Средняя компактность скручивания: {res['avg_compactness']:.3f}")
        print(f"Среднее расстояние до еды: {res['avg_distance_to_food']:.2f}")
        print(f"Количество циклов (зацикливаний): {res['loops_detected']}")
        print(f"Шагов с сегментированным полем: {res['partition_ratio']*100:.1f}%")
        print(f"Макс. изолированных карманов: {res['max_components']}")
        print(f"Первый разрыв поля на счете: {res['partition_start_score']}")
        print(f"Доступная доля пустого поля в конце (последние 50 шагов): {res['avg_final_reachable_ratio']*100:.1f}%")
        print("=" * 60)
        sys.exit(0)
        
    # Analyse all replays
    if not replays_dir.is_dir():
        print(f"[ОШИБКА] Папка с реплеями не найдена: {replays_dir}")
        sys.exit(1)
        
    files = []
    for ext in ["*.json.gz", "*.json"]:
        files.extend([f for f in replays_dir.glob(ext) if f.name not in {"last_replay.json", "last_replay.json.gz"}])
    if not files:
        print("Папка с реплеями пуста.")
        sys.exit(0)
        
    analyses = [analyze_replay_file(f) for f in files]
    
    # Print detailed table
    print(f"Найдено {len(analyses)} файлов реплеев в {replays_dir.name}:\n")
    print("=" * 115)
    print(f"{'Имя файла':<45} | {'Результат':<15} | {'Счет':<6} | {'Шагов':<6} | {'Циклы':<5} | {'Доступ еды':<10} | {'Классификация':<20}")
    print("-" * 115)
    for res in sorted(analyses, key=lambda x: x["score"], reverse=True):
        print(f"{res['filename'][:43]:<45} | {res['reason']:<15} | {res['score']:<6} | {res['total_steps']:<6} | {res['loops_detected']:<5} | {res['food_reachable_ratio']*100:>8.1f}% | {res['classification']:<20}")
    print("=" * 115)
    print()

    # Summary comparing Victory vs Starvation vs Collisions
    grouped = {}
    for res in analyses:
        cls = res["classification"]
        grouped.setdefault(cls, []).append(res)

    print("СВОДНАЯ СТАТИСТИКА ПО КЛАССАМ ИСХОДОВ:")
    print("=" * 80)
    print(f"{'Классификация':<25} | {'Кол-во':<6} | {'Ср. счет':<10} | {'Ср. шаги':<10} | {'Ср. компактность':<18}")
    print("-" * 80)
    for cls, runs in sorted(grouped.items(), key=lambda x: len(x[1]), reverse=True):
        avg_score = sum(r["score"] for r in runs) / len(runs)
        avg_steps = sum(r["total_steps"] for r in runs) / len(runs)
        avg_comp = sum(r["avg_compactness"] for r in runs) / len(runs)
        print(f"{cls:<25} | {len(runs):<6} | {avg_score:<10.2f} | {avg_steps:<10.1f} | {avg_comp:<18.3f}")
    print("=" * 80)

if __name__ == "__main__":
    main()

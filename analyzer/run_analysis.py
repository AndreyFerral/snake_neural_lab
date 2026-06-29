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
        "trapped_events": 0
    }
    
    if not steps:
        return analysis
        
    coiling_compactness = []
    final_steps_reachable_ratio = []
    dist_sum = 0
    dist_count = 0
    state_history = []
    partition_steps = 0
    
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
        print(f"Всего шагов: {res['total_steps']}")
        print("-" * 60)
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
        
    print(f"Найдено {len(files)} файлов для сравнения в {replays_dir.name}:\n")
    
    winning_runs = []
    starved_runs = []
    
    for f in files:
        res = analyze_replay_file(f)
        if res["reason"] == "victory":
            winning_runs.append(res)
        elif res["reason"] == "starvation":
            starved_runs.append(res)
            
    print(f"Побед: {len(winning_runs)} | Голод: {len(starved_runs)}")
    print("=" * 70)
    print(f"{'Показатель':<40} | {'Победа':<10} | {'Голод':<10} | {'Разница':<10}")
    print("-" * 70)
    
    def print_row(label, key, format_str="{:.3f}"):
        win_vals = [r[key] for r in winning_runs if r[key] is not None]
        starve_vals = [r[key] for r in starved_runs if r[key] is not None]
        
        win_avg = sum(win_vals) / len(win_vals) if win_vals else 0.0
        starve_avg = sum(starve_vals) / len(starve_vals) if starve_vals else 0.0
        diff = win_avg - starve_avg
        
        diff_str = f"+{format_str.format(diff)}" if diff > 0 else format_str.format(diff)
        print(f"{label:<40} | {format_str.format(win_avg):<10} | {format_str.format(starve_avg):<10} | {diff_str:<10}")
        
    print_row("Средний счет (яблоки)", "score", "{:.2f}")
    print_row("Среднее число шагов", "total_steps", "{:.1f}")
    print_row("Компактность скручивания (coiling)", "avg_compactness", "{:.3f}")
    print_row("Расстояние до еды (Manhattan)", "avg_distance_to_food", "{:.2f}")
    print_row("Обнаружено зацикливаний", "loops_detected", "{:.1f}")
    print_row("Доля шагов с разрывом поля", "partition_ratio", "{:.3f}")
    print_row("Макс. число изолированных карманов", "max_components", "{:.1f}")
    print_row("Доступное поле в конце (ratio)", "avg_final_reachable_ratio", "{:.3f}")
    print("=" * 70)

if __name__ == "__main__":
    main()

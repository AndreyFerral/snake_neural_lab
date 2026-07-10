import json
import gzip
import sys
import argparse
from pathlib import Path

# Add project root to path so we can import project files if needed
sys.path.append(str(Path(__file__).resolve().parent.parent))

from backend_config import GLOBAL_SIGNAL_LABELS, SIGNAL_LABELS

def get_signal_label(name: str) -> str:
    label = SIGNAL_LABELS.get(name, name)
    try:
        label.encode(sys.stdout.encoding or 'ascii')
        return label
    except Exception:
        # Fallback to plain English/ASCII
        ASCII_SIGNAL_LABELS = {
            "food_ahead": "Food Ahead",
            "food_left": "Food Left",
            "food_right": "Food Right",
            "food_behind": "Food Behind",
            "apple_step_forward": "Food Step Forward",
            "apple_step_left": "Food Step Left",
            "apple_step_right": "Food Step Right",
            "danger_forward": "Danger Forward",
            "danger_left": "Danger Left",
            "danger_right": "Danger Right",
            "tail_ahead": "Tail Ahead",
            "tail_left": "Tail Left",
            "tail_right": "Tail Right",
            "tail_behind": "Tail Behind",
            "tail_blocked_forward": "Tail Blocked Forward",
            "tail_blocked_left": "Tail Blocked Left",
            "tail_blocked_right": "Tail Blocked Right",
            "space_tight_forward": "Space Tight Forward",
            "space_tight_left": "Space Tight Left",
            "space_tight_right": "Space Tight Right",
            "tail_close_forward": "Tail Close Forward",
            "tail_close_left": "Tail Close Left",
            "tail_close_right": "Tail Close Right",
        }
        return ASCII_SIGNAL_LABELS.get(name, name)

def print_ascii_board(board_size: int, snake: list[list[int]], food: list[int]) -> None:
    def can_encode(s: str) -> bool:
        try:
            s.encode(sys.stdout.encoding or 'ascii')
            return True
        except Exception:
            return False
            
    has_emoji = all(can_encode(c) for c in ["🍎", "💀", "🟢", "🟡", "🟩", "·"])
    
    empty_char = "·" if has_emoji else "."
    food_char = "🍎" if has_emoji else "@"
    dead_char = "💀" if has_emoji else "X"
    head_char = "🟢" if has_emoji else "H"
    tail_char = "🟡" if has_emoji else "T"
    body_char = "🟩" if has_emoji else "o"

    # Initialize board grid
    grid = [[empty_char for _ in range(board_size)] for _ in range(board_size)]
    
    # Place food
    fx, fy = food
    if 0 <= fx < board_size and 0 <= fy < board_size:
        grid[fy][fx] = food_char
        
    # Place snake segments
    for idx, (x, y) in enumerate(snake):
        if not (0 <= x < board_size and 0 <= y < board_size):
            continue
        if idx == 0:
            grid[y][x] = dead_char if len(snake) == 1 else head_char
        elif idx == len(snake) - 1:
            grid[y][x] = tail_char
        else:
            grid[y][x] = body_char
            
    # Print the board with coordinates
    print("   " + " ".join(f"{x:2}" for x in range(board_size)))
    for y in range(board_size):
        row_str = " ".join(f"{cell:2}" for cell in grid[y])
        print(f"{y:2} {row_str}")

def main():
    parser = argparse.ArgumentParser(description="Пошаговый визуализатор реплеев.")
    parser.add_argument("file", nargs="?", help="Путь к файлу реплея (по умолчанию: last_replay.json.gz / last_replay.json)")
    parser.add_argument("--last", type=int, help="Показать последние N шагов")
    parser.add_argument("--step", type=int, help="Показать конкретный шаг")
    parser.add_argument("--all", action="store_true", help="Показать все шаги")
    args = parser.parse_args()
    
    project_root = Path(__file__).resolve().parent.parent
    replays_dir = project_root / ".runtime" / "replays"
    
    if args.file:
        replay_path = Path(args.file)
    else:
        gz_default = replays_dir / "last_replay.json.gz"
        if gz_default.is_file():
            replay_path = gz_default
        else:
            replay_path = replays_dir / "last_replay.json"
        
    if not replay_path.is_file():
        print(f"[ОШИБКА] Файл не найден: {replay_path}")
        sys.exit(1)
        
    if replay_path.suffix == ".gz":
        with gzip.open(replay_path, "rt", encoding="utf-8") as f:
            data = json.load(f)
    else:
        with open(replay_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
    board_size = data.get("board_size", 10)
    steps = data.get("steps", [])
    
    if not steps:
        print("В реплее нет записанных шагов.")
        sys.exit(0)
        
    # Filter steps to print
    steps_to_print = []
    if args.step is not None:
        steps_to_print = [s for s in steps if s.get("step") == args.step]
    elif args.last is not None:
        steps_to_print = steps[-args.last:]
    elif args.all:
        steps_to_print = steps
    else:
        # Default: show last step (death)
        steps_to_print = [steps[-1]]
        print("Показываем последний шаг игры. Используйте --last N или --step N для других шагов.")
        
    def can_encode(s: str) -> bool:
        try:
            s.encode(sys.stdout.encoding or 'ascii')
            return True
        except Exception:
            return False
    active_marker = "🟢" if can_encode("🟢") else "[+]"
    
    for step_data in steps_to_print:
        step_num = step_data.get("step")
        snake = step_data.get("snake", [])
        food = step_data.get("food", [0, 0])
        direction = step_data.get("direction", 0)
        action = step_data.get("action", 0)
        reward = step_data.get("reward", 0.0)
        reason = step_data.get("reason", "unknown")
        penalties = step_data.get("penalties", [])
        
        # Resolve action label
        from backend_config import MODEL_ACTION_LABELS
        action_label = step_data.get("action_label")
        if not action_label and 0 <= action < len(MODEL_ACTION_LABELS):
            action_label = MODEL_ACTION_LABELS[action]
        else:
            action_label = action_label or "-"
            
        # Reconstruct observation on the fly if needed
        observation = step_data.get("observation", [])
        if not observation and snake:
            from snake_env import SnakeEnv
            env = SnakeEnv(board_size=board_size)
            env.snake = [(p[0], p[1]) for p in snake]
            env.food = (food[0], food[1])
            env.direction = direction
            env._clear_observation_cache()
            observation = env.get_observation_grid().tolist()
            
        print(f"\n--- ШАГ {step_num} | Направление: {direction} | Действие: {action_label} (отн: {action}) ---")
        print(f"Награда за шаг: {reward} | Результат: {reason} | Штрафы: {', '.join(penalties) if penalties else 'нет'}")
        print_ascii_board(board_size, snake, food)
        
        # Print observation features
        print("\nПоказания датчиков (вход нейросети):")
        active_features = []
        for idx, val in enumerate(observation):
            if idx < len(GLOBAL_SIGNAL_LABELS):
                name = GLOBAL_SIGNAL_LABELS[idx]
                label = get_signal_label(name)
                if val > 0.0:
                    active_features.append(f"[{idx:2}] {label} ({name}): {val:.1f}")
        
        if active_features:
            for feat in active_features:
                print(f"    {active_marker} {feat}")
        else:
            print("    (нет активных)")
        print("-" * 60)

if __name__ == "__main__":
    main()

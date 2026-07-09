import sys
import argparse
import torch
import gzip
import json
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from snake_env import SnakeEnv
from dqn_agent import DQNAgent

# Helper to convert relative action to absolute direction
def relative_to_absolute(current_dir: int, action: int) -> int:
    if action == 0:  # Forward
        return current_dir
    elif action == 1:  # Left
        return (current_dir - 1) % 4
    elif action == 2:  # Right
        return (current_dir + 1) % 4
    return current_dir

def main():
    parser = argparse.ArgumentParser(description="Анализ решения модели на конкретном шаге или по всем реплеям.")
    parser.add_argument("path", nargs="?", default=None, help="Путь к файлу реплея или директории (по умолчанию .runtime/replays)")
    parser.add_argument("--step", type=int, default=None, help="Номер шага для анализа (по умолчанию - ход перед смертью)")
    parser.add_argument("--verbose", action="store_true", help="Выводить подробные входные сигналы для каждого файла")
    args = parser.parse_args()

    # Determine target path
    if args.path is None:
        target_path = Path(__file__).resolve().parent.parent / ".runtime" / "replays"
    else:
        target_path = Path(args.path)

    if not target_path.exists():
        print(f"[ОШИБКА] Путь не найден: {target_path}")
        sys.exit(1)

    # Collect files
    files = []
    if target_path.is_file():
        files.append(target_path)
    else:
        files = list(target_path.glob("replay_*.json.gz")) + list(target_path.glob("replay_*.json"))
        # Sort alphabetically
        files = sorted(files, key=lambda x: x.name)

    if not files:
        print(f"Реплеи не найдены в {target_path}")
        sys.exit(0)

    # Load agent
    device = torch.device("cpu")
    agent = DQNAgent(device=device)
    checkpoint_path = Path(__file__).resolve().parent.parent / ".runtime" / "snake_dqn.pt"

    if checkpoint_path.exists():
        try:
            payload = torch.load(checkpoint_path, map_location=device)
            agent.import_checkpoint(payload)
            print(f"Загружен чекпоинт модели из {checkpoint_path.name}")
        except Exception as e:
            print(f"[ВНИМАНИЕ] Не удалось загрузить чекпоинт ({e}), анализ на случайных весах.")
    else:
        print("[ВНИМАНИЕ] Чекпоинт не найден, анализ на случайных весах.")

    # Table headers
    headers = [
        "Имя файла", "Счет", "Шаг", "Q: Форв/Лев/Прав", 
        "Выбор (Жадный)", "Факт.ход", "Диагностика исхода"
    ]
    
    print("\n" + "=" * 125)
    print(f"{headers[0]:<40} | {headers[1]:<4} | {headers[2]:<4} | {headers[3]:<18} | {headers[4]:<15} | {headers[5]:<8} | {headers[6]}")
    print("-" * 125)

    env = SnakeEnv()

    for file_path in files:
        # Load replay
        try:
            if file_path.suffix == ".gz":
                with gzip.open(file_path, "rt", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
        except Exception as e:
            print(f"{file_path.name:<40} | ОШИБКА загрузки файла: {e}")
            continue

        steps = data.get("steps", [])
        if len(steps) < 2:
            print(f"{file_path.name:<40} | Слишком короткий реплей")
            continue

        # Determine target state step and the action taken from it
        if args.step is not None:
            # Find the step matching args.step as the state
            state_step_idx = -1
            for idx, s in enumerate(steps):
                if s.get("step") == args.step:
                    state_step_idx = idx
                    break
            
            if state_step_idx == -1:
                print(f"{file_path.name:<40} | Шаг {args.step} не найден")
                continue
            
            if state_step_idx == len(steps) - 1:
                print(f"{file_path.name:<40} | Шаг {args.step} является финальным (нет последующего действия для анализа)")
                continue
                
            state_step = steps[state_step_idx]
            next_step = steps[state_step_idx + 1]
        else:
            # Analyze the decision that led to death
            state_step = steps[-2]
            next_step = steps[-1]

        snake = state_step.get("snake", [])
        food = state_step.get("food", [0, 0])
        direction = state_step.get("direction", 1)
        
        # The action taken in this state is recorded in the next step
        actual_action = next_step.get("action", 0)
        is_random = next_step.get("action_random", False)

        # Recreate state
        env.snake = [(p[0], p[1]) for p in snake]
        env.food = (food[0], food[1])
        env.direction = direction
        env._clear_observation_cache()

        # Get observation and evaluate Q-values
        obs = env.get_observation_grid()
        state_tensor = agent._array_to_float_tensor(obs)
        with torch.inference_mode():
            q_values = agent.online_net(state_tensor).squeeze(0).cpu().numpy()

        greedy_action = int(np.argmax(q_values))
        q_str = f"{q_values[0]:.1f}/{q_values[1]:.1f}/{q_values[2]:.1f}"
        
        action_names = ["Вперед", "Влево", "Вправо"]
        greedy_name = action_names[greedy_action]
        actual_name = action_names[actual_action]
        if is_random:
            actual_name += " (Рнд)"

        # Check options
        safety_status = []
        for act in range(3):
            danger = env._danger_for_relative_action(act) >= 0.5
            tail_blocked = env._is_tail_blocked_after_relative_action(act) > 0.5
            if danger:
                safety_status.append(f"{action_names[act]}:Смерть")
            elif tail_blocked:
                safety_status.append(f"{action_names[act]}:ХвостБл")
            else:
                safety_status.append(f"{action_names[act]}:ОК")

        # Determine diagnostic message
        chosen_danger = env._danger_for_relative_action(actual_action) >= 0.5
        chosen_tail_blocked = env._is_tail_blocked_after_relative_action(actual_action) > 0.5
        
        # Was there a safer move?
        safe_moves = []
        for act in range(3):
            danger = env._danger_for_relative_action(act) >= 0.5
            tail_blocked = env._is_tail_blocked_after_relative_action(act) > 0.5
            if not danger and not tail_blocked:
                safe_moves.append(action_names[act])

        verdict = ""
        if chosen_danger:
            verdict = "ВЫБРАН СМЕРТЕЛЬНЫЙ ХОД! "
            if safe_moves:
                verdict += f"Был безопасный: {', '.join(safe_moves)}"
            else:
                verdict += "Выхода не было (Заперта)"
        elif chosen_tail_blocked:
            verdict = "Блокировка хвоста! "
            if safe_moves:
                verdict += f"Был безопасный: {', '.join(safe_moves)}"
            else:
                verdict += "Все ходы блокируют хвост"
        else:
            verdict = "Безопасный ход"

        short_name = file_path.name
        if len(short_name) > 40:
            short_name = short_name[:37] + "..."

        print(f"{short_name:<40} | {len(snake)-3:<4} | {state_step.get('step'):<4} | {q_str:<18} | {greedy_name:<15} | {actual_name:<8} | {verdict}")

        if args.verbose:
            print(f"  Датчики: {[idx for idx, val in enumerate(obs) if val > 0.0]}")
            print(f"  Опции: {', '.join(safety_status)}")
            print("-" * 125)

    print("=" * 125)

if __name__ == "__main__":
    main()

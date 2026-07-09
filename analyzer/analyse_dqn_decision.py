import sys
import torch
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parent.parent))

from snake_env import SnakeEnv
from dqn_agent import DQNAgent

# Initialize env and set state for step 209
env = SnakeEnv()
env.snake = [(8, 9), (8, 8), (8, 7), (7, 7), (7, 8), (6, 8), (5, 8), (5, 7), (4, 7), (3, 7), (2, 7), (1, 7), (0, 7), (0, 8), (0, 9), (1, 9)]
env.food = (9, 9)
env.direction = 2
env._clear_observation_cache()

# Get observation
obs = env.get_observation_grid()

# Load agent
device = torch.device("cpu")
agent = DQNAgent(device=device)
checkpoint_path = Path(__file__).resolve().parent.parent / ".runtime" / "snake_dqn.pt"

if checkpoint_path.exists():
    payload = torch.load(checkpoint_path, map_location=device)
    agent.import_checkpoint(payload)
    print("Loaded model checkpoint successfully.")
else:
    print("Checkpoint not found!")
    sys.exit(1)

# Evaluate Q-values
state_tensor = agent._array_to_float_tensor(obs)
with torch.inference_mode():
    q_values = agent.online_net(state_tensor).squeeze(0).cpu().numpy()

print("\n--- STATE ANALYSIS ---")
print(f"Head: {env.snake[0]}, Food: {env.food}, Direction: {env.direction} (DOWN)")
print("\nActive Sensors:")
from backend_config import GLOBAL_SIGNAL_LABELS
for idx, val in enumerate(obs):
    if val > 0.0:
        print(f"  [{idx:2}] {GLOBAL_SIGNAL_LABELS[idx]}: {val:.2f}")

print("\n--- ACTIONS EVALUATION ---")
# Actions: 0 = Forward, 1 = Left, 2 = Right
action_names = ["Forward (0)", "Left (1)", "Right (2)"]
for act in range(3):
    q_val = q_values[act]
    # Check what next state would look like
    danger = env._danger_for_relative_action(act)
    tail_blocked = env._is_tail_blocked_after_relative_action(act)
    space_tight = env._is_space_tight_after_relative_action(act)
    tail_close = env._is_tail_close_after_relative_action(act)
    partition = env._is_space_partitioned_after_relative_action(act)
    
    print(f"\nAction: {action_names[act]}")
    print(f"  Q-value: {q_val:.4f}")
    print(f"  Danger: {danger}")
    print(f"  Tail Blocked: {tail_blocked}")
    print(f"  Space Tight: {space_tight}")
    print(f"  Tail Close: {tail_close}")
    print(f"  Partition: {partition}")

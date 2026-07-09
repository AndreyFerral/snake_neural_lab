import sys
import gzip
import json
from pathlib import Path

replays_dir = Path(__file__).resolve().parent.parent / ".runtime" / "replays"
replay_files = sorted(list(replays_dir.glob("replay_score_5*.json.gz")))

if not replay_files:
    print("No score 5x replays found!")
    sys.exit(1)

replay_path = replay_files[0]
with gzip.open(replay_path, "rt", encoding="utf-8") as f:
    data = json.load(f)

print(f"Inspecting file: {replay_path.name}")
print(f"Final Score: {data['final_score']}")
print(f"Final Reason: {data['final_reason']}")
print(f"Total Steps: {len(data['steps'])}")

print("\nLast 25 steps:")
print("="*80)
print(f"{'Step':<5} | {'Head':<8} | {'Food':<8} | {'Action':<6} | {'Reward':<7} | {'Penalties'}")
print("-"*80)

for step in data['steps'][-25:]:
    head = step.get("snake")[0] if step.get("snake") else None
    food = step.get("food")
    act = step.get("action")
    rew = step.get("reward")
    penalties = step.get("penalties", [])
    print(f"{step['step']:<5} | {str(head):<8} | {str(food):<8} | {act:<6} | {rew:<7.2f} | {penalties}")
print("="*80)

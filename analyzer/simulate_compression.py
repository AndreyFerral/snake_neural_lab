import gzip
import json
import random

# 1. Generate 500 steps
steps_500 = []
snake = [[i % 10, i // 10] for i in range(95)]
for step_num in range(500):
    snake = [[(x + 1) % 10, y] for x, y in snake]
    steps_500.append({
        "step": 1700 + step_num,
        "snake": list(snake),
        "food": [random.randint(0, 9), random.randint(0, 9)],
        "direction": random.randint(0, 3),
        "action": random.randint(0, 2),
        "reward": -2.4 if random.random() < 0.2 else 0.0,
        "reason": "step",
        "penalties": ["tail_close"] if random.random() < 0.2 else []
    })

replay_500 = {
    "agent_kind": "dqn",
    "board_size": 10,
    "final_score": 95,
    "final_reason": "starvation",
    "steps": steps_500
}
raw_bytes_500 = json.dumps(replay_500, ensure_ascii=False).encode("utf-8")

# 2. Generate 2283 steps (Without Step Cap)
steps_2283 = []
snake = [[i % 10, i // 10] for i in range(95)]
for step_num in range(2283):
    snake = [[(x + 1) % 10, y] for x, y in snake]
    steps_2283.append({
        "step": step_num + 1,
        "snake": list(snake),
        "food": [random.randint(0, 9), random.randint(0, 9)],
        "direction": random.randint(0, 3),
        "action": random.randint(0, 2),
        "reward": -2.4 if random.random() < 0.2 else 0.0,
        "reason": "step",
        "penalties": ["tail_close"] if random.random() < 0.2 else []
    })

replay_2283 = {
    "agent_kind": "dqn",
    "board_size": 10,
    "final_score": 95,
    "final_reason": "starvation",
    "steps": steps_2283
}
raw_bytes_2283 = json.dumps(replay_2283, ensure_ascii=False).encode("utf-8")

print(f"500 Steps Raw Size: {len(raw_bytes_500)} bytes")
print("Compression sizes for 500 steps:")
for level in range(1, 10):
    comp = gzip.compress(raw_bytes_500, compresslevel=level)
    print(f"  Level {level}: {len(comp)} bytes")

print(f"\n2283 Steps Raw Size: {len(raw_bytes_2283)} bytes")
print("Compression sizes for 2283 steps:")
for level in range(1, 10):
    comp = gzip.compress(raw_bytes_2283, compresslevel=level)
    print(f"  Level {level}: {len(comp)} bytes")

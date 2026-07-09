import gzip
import json
import sys
from pathlib import Path

replays_dir = Path(__file__).resolve().parent.parent / ".runtime" / "replays"
file_large = replays_dir / "replay_score_95_ep_295_live_1783604709.json.gz"
file_small = replays_dir / "replay_score_95_ep_299_live_1783604717.json.gz"

if not file_large.exists() or not file_small.exists():
    # If the user restarted the training, the replays directory might not have these files anymore.
    # Let's see if there are other files in the replays directory to compare!
    files = sorted(list(replays_dir.glob("replay_score_*.json.gz")))
    if len(files) < 2:
        print("Not enough replay files to compare!")
        sys.exit(0)
    file_large = files[0]
    file_small = files[1]

print("Comparing Replay Files:")
print(f"File 1 (Large candidate): {file_large.name} (size: {file_large.stat().st_size} bytes)")
print(f"File 2 (Small candidate): {file_small.name} (size: {file_small.stat().st_size} bytes)")
print("="*100)

with gzip.open(file_large, "rt", encoding="utf-8") as f:
    data_large = json.load(f)
with gzip.open(file_small, "rt", encoding="utf-8") as f:
    data_small = json.load(f)

print(f"File 1: steps count = {len(data_large['steps'])}")
print(f"File 2: steps count = {len(data_small['steps'])}")

step_1 = data_large['steps'][0]
step_2 = data_small['steps'][0]

print("\nFile 1 first step keys and types:")
for k, v in step_1.items():
    print(f"  {k}: {type(v)} (len/val: {len(v) if isinstance(v, (list, dict, str)) else v})")
    
print("\nFile 2 first step keys and types:")
for k, v in step_2.items():
    print(f"  {k}: {type(v)} (len/val: {len(v) if isinstance(v, (list, dict, str)) else v})")

# Let's inspect step sizes and see where the bytes are
print("\nLet's check the size of the 'snake' array in steps:")
print(f"File 1 max snake len: {max(len(s.get('snake', [])) for s in data_large['steps'])}")
print(f"File 2 max snake len: {max(len(s.get('snake', [])) for s in data_small['steps'])}")

# Let's print a sample step from both to see the exact text
print("\nSample step from File 1 (JSON):")
print(json.dumps(data_large['steps'][100], ensure_ascii=False)[:300])

print("\nSample step from File 2 (JSON):")
print(json.dumps(data_small['steps'][100], ensure_ascii=False)[:300])

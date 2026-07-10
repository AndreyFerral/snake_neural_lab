import json
from pathlib import Path

transcript_path = Path("C:/Users/Andrey/.gemini/antigravity/brain/c993cceb-d892-4f6e-a10c-fb5b87480a26/.system_generated/logs/transcript_full.jsonl")

if not transcript_path.exists():
    print("Transcript path does not exist!")
    exit(1)

print("Searching transcript for penalty values...")
for line in open(transcript_path, "r", encoding="utf-8"):
    data = json.loads(line)
    content = data.get("content", "")
    
    # Let's search for changes to penalty variables
    if any(x in content for x in ["DEATH_PENALTY", "TAIL_LOSS_PENALTY", "PARTITION_PENALTY"]):
        print(f"Step {data.get('step_index')} ({data.get('type')}):")
        # Find lines in content containing the keywords
        for cl in content.split("\n"):
            if any(x in cl for x in ["DEATH_PENALTY", "TAIL_LOSS_PENALTY", "PARTITION_PENALTY", "gamma"]):
                print("  ", cl.strip())

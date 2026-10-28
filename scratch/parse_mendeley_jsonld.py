import requests
import re
import json

r = requests.get('https://data.mendeley.com/datasets/smzggbnkd2/3', headers={'User-Agent': 'Mozilla/5.0'})
scripts = re.findall(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', r.text, re.DOTALL)
for i, s in enumerate(scripts):
    d = json.loads(s)
    print(f"--- Script {i} ---")
    print(json.dumps(d, indent=2))

# Also search for files or data in HTML
print("\n--- Searching for data-state or window.__INITIAL_STATE__ ---")
initial_states = re.findall(r'window\.__INITIAL_STATE__\s*=\s*(\{.*?\});\s*</script>', r.text, re.DOTALL)
print("Initial states found:", len(initial_states))
if initial_states:
    state_json = json.loads(initial_states[0])
    print("State keys:", list(state_json.keys()))
    if "dataset" in state_json:
        ds = state_json["dataset"]
        print("Dataset keys:", list(ds.keys()))
        if "files" in ds:
            print("Files in state:", len(ds["files"]))
            for f in ds["files"]:
                print("  File:", f)

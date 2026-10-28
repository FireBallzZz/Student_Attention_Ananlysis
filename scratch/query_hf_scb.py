import requests
import json

hf_url = "https://huggingface.co/api/datasets/wintonYF/SCB-Dataset/tree/main"
print(f"Querying HuggingFace API: {hf_url}")
try:
    r = requests.get(hf_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
    print("Status:", r.status_code)
    if r.status_code == 200:
        items = r.json()
        print(f"Items in SCB-Dataset repo: {len(items)}")
        for it in items:
            print(f"  {it.get('path')} ({it.get('type')}) - size: {it.get('size')}")
except Exception as e:
    print("HF query error:", e)

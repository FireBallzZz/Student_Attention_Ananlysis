import requests
import json

urls = [
    "https://data.mendeley.com/api/datasets/smzggbnkd2/files",
    "https://data.mendeley.com/api/datasets/smzggbnkd2/3/files",
    "https://data.mendeley.com/public-api/datasets/smzggbnkd2/files?version=3",
    "https://data.mendeley.com/api/datasets/smzggbnkd2/files?version=3"
]

for u in urls:
    print(f"=== URL: {u} ===")
    r = requests.get(u, headers={'User-Agent': 'Mozilla/5.0'})
    print("Status:", r.status_code)
    if r.status_code == 200:
        try:
            data = r.json()
            print(json.dumps(data, indent=2)[:1000])
        except:
            print(r.text[:300])

import requests
import re
import json

url = "https://data.mendeley.com/datasets/smzggbnkd2/3"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
r = requests.get(url, headers=headers)
print("Status:", r.status_code)
html = r.text

# Look for json-ld or direct links
scripts = re.findall(r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', html, re.DOTALL)
print(f"Found {len(scripts)} json-ld scripts")
for s in scripts:
    try:
        data = json.loads(s)
        print("JSON-LD Keys:", list(data.keys()))
        if "name" in data:
            print("Name:", data.get("name"))
        if "distribution" in data:
            dist = data.get("distribution")
            print("Distribution count:", len(dist) if isinstance(dist, list) else dist)
            if isinstance(dist, list):
                for d in dist:
                    print("  Dist:", d)
    except Exception as e:
        print("Error parsing json-ld:", e)

# Also check for download links in html
links = re.findall(r'href=["\'](https?://[^"\']+)["\']', html)
for l in sorted(set(links)):
    if "download" in l or "smzggbnkd2" in l or "mendeley" in l:
        print("Link:", l)

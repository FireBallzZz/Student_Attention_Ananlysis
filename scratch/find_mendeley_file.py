import requests
import re
import json

r = requests.get('https://data.mendeley.com/datasets/smzggbnkd2/3', headers={'User-Agent': 'Mozilla/5.0'})
# Find all occurrences of "file" or "download" or UUIDs in HTML
print("HTML length:", len(r.text))

# Search for any json blobs in HTML
blobs = re.findall(r'<script[^>]*>(.*?)</script>', r.text, re.DOTALL)
print(f"Total script tags: {len(blobs)}")
for i, b in enumerate(blobs):
    if "smzggbnkd2" in b or "download" in b or "6244" in b or ".csv" in b or ".xlsx" in b:
        print(f"Match in script {i}: {len(b)} chars")
        # Print snippet
        for line in b.splitlines():
            if any(term in line.lower() for term in ["download", "file", ".csv", ".xlsx", "smzggbnkd2"]):
                print("  ", line.strip()[:150])

# Also check Mendeley public API endpoints:
api_urls = [
    "https://api.mendeley.com/datasets/smzggbnkd2/3",
    "https://data.mendeley.com/public-api/datasets/smzggbnkd2/3",
    "https://data.mendeley.com/public-api/datasets/smzggbnkd2/files",
    "https://data.mendeley.com/api/datasets/smzggbnkd2/files",
    "https://data.mendeley.com/datasets/smzggbnkd2/3/files"
]
for u in api_urls:
    try:
        res = requests.get(u, headers={'User-Agent': 'Mozilla/5.0'}, timeout=5)
        print(f"API {u} -> Status {res.status_code}")
        if res.status_code == 200:
            print("  Response:", res.text[:300])
    except Exception as e:
        print(f"API {u} -> Error {e}")

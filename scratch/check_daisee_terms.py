import requests
import re

url = "https://people.iith.ac.in/vineethnb/resources/daisee/"
r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
text = r.text

# Extract text from html
clean_text = re.sub(r'<[^>]+>', ' ', text)
lines = [l.strip() for l in clean_text.splitlines() if l.strip()]

print(f"Total lines: {len(lines)}")
for i, line in enumerate(lines):
    if any(k in line.lower() for k in ["download", "access", "form", "agreement", "eula", "terms", "license", "request"]):
        print(f"L{i}: {line}")

import requests
import base64

r = requests.get('https://api.github.com/repos/Whiffe/SCB-dataset/contents/README.md', headers={'User-Agent': 'Mozilla/5.0'})
if r.status_code == 200:
    content = base64.b64decode(r.json()['content']).decode('utf-8', errors='ignore')
    with open("scratch/scb_readme.txt", "w", encoding="utf-8") as f:
        f.write(content)
    print("Saved scratch/scb_readme.txt, total length:", len(content))
    for line in content.splitlines()[:50]:
        print(line.encode('ascii', errors='replace').decode('ascii'))

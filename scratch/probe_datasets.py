import json
import requests

def probe_mendeley():
    url = "https://data.mendeley.com/api/datasets/smzggbnkd2/3"
    print(f"Querying Mendeley API: {url}")
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        print("Status code:", r.status_code)
        if r.status_code == 200:
            data = r.json()
            print("Title:", data.get("name"))
            print("DOI:", data.get("doi", {}).get("id"))
            print("License:", data.get("licence", {}).get("name"))
            files = data.get("files", [])
            print(f"Total files: {len(files)}")
            for f in files:
                print(f"  - Name: {f.get('filename')}, Size: {f.get('size')} bytes, MIME: {f.get('content_type')}, Download: {f.get('download_url')}")
            return data
    except Exception as e:
        print("Mendeley query failed:", e)
    return None

def probe_scb():
    url = "https://api.github.com/repos/Whiffe/SCB-dataset"
    print(f"\nQuerying GitHub API for SCB-dataset: {url}")
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        print("Status code:", r.status_code)
        if r.status_code == 200:
            data = r.json()
            print("Description:", data.get("description"))
            print("License:", data.get("license"))
            print("Size KB:", data.get("size"))
            print("Default branch:", data.get("default_branch"))
            
        # Check contents
        contents_url = "https://api.github.com/repos/Whiffe/SCB-dataset/contents"
        r_c = requests.get(contents_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        if r_c.status_code == 200:
            items = r_c.json()
            print(f"Root contents count: {len(items)}")
            for it in items:
                print(f"  - {it.get('name')} ({it.get('type')})")
    except Exception as e:
        print("SCB query failed:", e)

def probe_daisee():
    url = "https://people.iith.ac.in/vineethnb/resources/daisee/"
    print(f"\nQuerying DAiSEE URL: {url}")
    try:
        r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        print("Status code:", r.status_code)
        print("Length:", len(r.text))
        # Look for keywords like license, form, agreement, download
        for line in r.text.splitlines():
            line_str = line.strip()
            if any(k in line_str.lower() for k in ["dataset", "form", "agreement", "request", "license", "download", "access"]):
                if len(line_str) < 200:
                    print("  ", line_str[:120])
    except Exception as e:
        print("DAiSEE query failed:", e)

if __name__ == "__main__":
    probe_mendeley()
    probe_scb()
    probe_daisee()

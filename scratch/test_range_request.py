import requests

url = "https://huggingface.co/datasets/wintonYF/SCB-Dataset/resolve/main/SCB5-Handrise-Read-write-2024-9-17/SCB5-Handrise-Read-write-2024-9-17.zip"
headers = {"Range": "bytes=0-1023", "User-Agent": "Mozilla/5.0"}
r = requests.get(url, headers=headers, timeout=10)
print("Status code:", r.status_code)
print("Accept-Ranges:", r.headers.get("Accept-Ranges"))
print("Content-Range:", r.headers.get("Content-Range"))
print("Content-Length:", r.headers.get("Content-Length"))

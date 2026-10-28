import re
from pathlib import Path

content_path = Path(r"C:\Users\User\.gemini\antigravity-ide\brain\16a8ddc1-c385-4cb7-9dc7-6c921cf55940\.system_generated\steps\511\content.md")
if content_path.exists():
    text = content_path.read_text(encoding="utf-8", errors="ignore")
    # Search for all URLs or file references
    urls = re.findall(r'https?://[^\s"\'<>]+', text)
    print(f"Total URLs found in Mendeley HTML: {len(urls)}")
    for u in sorted(set(urls)):
        if "data.mendeley.com" in u or "mendeley" in u:
            print("  -", u)
else:
    print("Content path not found.")

import sys, zipfile
sys.path.append(".")
from scratch.test_remote_zip import HttpRangeFile

url = "https://huggingface.co/datasets/wintonYF/SCB-Dataset/resolve/main/SCB5-Handrise-Read-write-2024-9-17/SCB5-Handrise-Read-write-2024-9-17.zip"
f = HttpRangeFile(url)
zf = zipfile.ZipFile(f)

# Find sample label
sample_label = "SCB5-Handrise-Read-write-2024-9-17/labels/train/16_000805.txt"
content = zf.read(sample_label).decode('utf-8')
print("Sample label file:", sample_label)
print("Content:\n", content)

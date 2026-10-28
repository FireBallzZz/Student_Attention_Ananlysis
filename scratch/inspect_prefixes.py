import sys, zipfile
from collections import Counter
sys.path.append(".")
from scratch.test_remote_zip import HttpRangeFile

url = "https://huggingface.co/datasets/wintonYF/SCB-Dataset/resolve/main/SCB5-Handrise-Read-write-2024-9-17/SCB5-Handrise-Read-write-2024-9-17.zip"
f = HttpRangeFile(url)
zf = zipfile.ZipFile(f)
names = zf.namelist()

train_imgs = [n.split('/')[-1] for n in names if 'images/train/' in n and n.endswith(('.jpg', '.png'))]
val_imgs = [n.split('/')[-1] for n in names if 'images/val/' in n and n.endswith(('.jpg', '.png'))]

train_prefixes = [x.split('_')[0] if '_' in x else x[:4] for x in train_imgs]
val_prefixes = [x.split('_')[0] if '_' in x else x[:4] for x in val_imgs]

print("Top train prefixes:", Counter(train_prefixes).most_common(10))
print("Top val prefixes:", Counter(val_prefixes).most_common(10))

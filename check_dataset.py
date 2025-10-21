# check_dataset.py
from pathlib import Path

p = Path("my_yolo_dataset")
for sub in ["images/train","images/val","labels/train","labels/val"]:
    pp = p/sub
    print(sub, "exists:", pp.exists(), "count:", len(list(pp.glob('*'))))

# list some images without labels (train)
missing=[]
for f in (p/"images"/"train").glob("*"):
    txt=(p/"labels"/"train"/(f.stem+".txt"))
    if not txt.exists() or txt.stat().st_size==0:
        missing.append(f.name)
print("train images missing/empty labels (sample):", missing[:20])

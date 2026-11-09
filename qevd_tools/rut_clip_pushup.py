"""Rút riêng các clip chống đẩy khỏi một Part của QEVD-FIT-300K mà không cần giải nén cả Part.

Cách dùng (chạy trong thư mục dự án):  python qevd_tools/rut_clip_pushup.py 4
Cần có trong Data/QEVD-FIT-COACH/: 3 file QEVD-FIT-300k-Part-N.z01/.z02/.zip, fine_grained_labels.json,
fine_grained_labels_with_worker_ids.json. Kết quả: short_clips/*.mp4 và pushup_clips.json (gộp mọi Part đã rút).
"""
import json, os, sys
from splitzip import SplitZip

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "Data", "QEVD-FIT-COACH")
EXERCISES = {"pushups", "pushups (on knees)"}

part = int(sys.argv[1])
prefix = f"QEVD-FIT-300k-Part-{part}"
z = SplitZip([os.path.join(D, f"{prefix}.z01"), os.path.join(D, f"{prefix}.z02"), os.path.join(D, f"{prefix}.zip")])
labels = json.load(open(os.path.join(D, "fine_grained_labels.json"), encoding="utf-8"))
worker = {os.path.basename(r["video_path"]): r["worker_id"]
          for r in json.load(open(os.path.join(D, "fine_grained_labels_with_worker_ids.json"), encoding="utf-8"))}
out_dir = os.path.join(D, "short_clips")
os.makedirs(out_dir, exist_ok=True)

rows = []
for r in labels:
    ex = {l.split(" - ")[0] for l in r["labels"]} & EXERCISES
    name = os.path.basename(r["video_path"])
    if not ex or f"{prefix}/{name}" not in z.entries:
        continue
    dst = os.path.join(out_dir, name)
    if not os.path.exists(dst):
        with open(dst, "wb") as f:
            f.write(z.read(f"{prefix}/{name}"))
    rows.append(dict(video=name, part=part, exercise=sorted(ex)[0], worker_id=worker[name], split=r["split"],
                     attrs=[l.split(" - ", 1)[1] for l in r["labels"]]))
z.close()

index_path = os.path.join(D, "pushup_clips.json")
old = json.load(open(index_path, encoding="utf-8")) if os.path.exists(index_path) else []
merged = [r for r in old if r["part"] != part] + rows
json.dump(sorted(merged, key=lambda r: r["video"]), open(index_path, "w", encoding="utf-8"), indent=1)
print(f"Part {part}: {len(rows)} clip | tổng trong pushup_clips.json: {len(merged)}")

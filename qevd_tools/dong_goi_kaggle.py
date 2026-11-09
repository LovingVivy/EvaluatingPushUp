"""Đóng gói keypoint + nhãn của clip chống đẩy QEVD thành một file zip để upload lên Kaggle (dataset PRIVATE).

Cách dùng:  python qevd_tools/dong_goi_kaggle.py
Kết quả: qevd_pushup_keypoints.zip ở thư mục dự án.
"""
import json, os, zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
D = os.path.join(ROOT, "Data", "QEVD-FIT-COACH")
clips = [r for r in json.load(open(os.path.join(D, "pushup_clips.json"), encoding="utf-8")) if r["exercise"] == "pushups"]
clips = [r for r in clips if os.path.exists(os.path.join(D, "keypoints_pushups", r["video"].replace(".mp4", ".npz")))]

out = os.path.join(ROOT, "qevd_pushup_keypoints.zip")
with zipfile.ZipFile(out, "w", zipfile.ZIP_STORED) as z:          # .npz đã nén sẵn
    z.writestr("qevd_pushup_keypoints/qevd_pushup_clips.json", json.dumps(clips, indent=1))
    for r in clips:
        name = r["video"].replace(".mp4", ".npz")
        z.write(os.path.join(D, "keypoints_pushups", name), f"qevd_pushup_keypoints/keypoints/{name}")
print(len(clips), "clip ->", out, round(os.path.getsize(out) / 1e6), "MB")

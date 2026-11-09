"""Phân tích một video chống đẩy: trích khung xương, nhận diện góc quay, đếm rep và chấm từng rep.

Quy trình giống notebook 08: đếm rep bằng cách dò đáy trên cả video (không phải bộ đếm trực tiếp của bản camera).
Mô hình FeatCNN, cách chuẩn hoá và quy tắc kết luận dùng chung với realtime/chay_camera.py, nên kết quả khớp với
lúc huấn luyện.
"""
import os
import sys
import threading
import time

import cv2
import mediapipe as mp
import numpy as np
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
from scipy.signal import find_peaks, savgol_filter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "realtime"))
import chay_camera as rt  # noqa: E402  (mô hình, chuẩn hoá, ngưỡng kết luận, cách vẽ khung xương)

TARGET_FPS = 15.0                 # dữ liệu huấn luyện đều ở 15 khung/giây
VIS_THR = 0.5
MIN_AMP = {"front": 0.10, "side": 3.0, "diagonal": 3.0}     # biên độ tối thiểu của một rep (notebook 02, 08)
L_SH, R_SH, L_EL, R_EL, L_WR, R_WR = 11, 12, 13, 14, 15, 16
L_HIP, L_ANK = 23, 27

# Tên kết luận trong kết quả trả về  ->  tên mà realtime/chay_camera.py dùng khi vẽ
VERDICT_RT = {"dung": "dung", "sai": "sai", "khong_chac": "khong chac", "khong_danh_gia": "khong danh gia duoc"}

_models, _models_lock = None, threading.Lock()


def models():
    global _models
    with _models_lock:
        if _models is None:
            _models = rt.load_models()
        return _models


@torch.no_grad()
def predict(X, view):
    x = torch.from_numpy(X)
    v = torch.full((len(X),), rt.VIEW_NAMES.index(view), dtype=torch.long)
    return torch.stack([torch.sigmoid(m(x, v)) for m in models()]).mean(0).numpy()        # (số rep, 4)


# ----------------------------------------------------------------------------------------------------------------------
# Trích khung xương
# ----------------------------------------------------------------------------------------------------------------------
def extract_keypoints(path, progress=None):
    """Đọc video, chạy MediaPipe trên khoảng 15 khung/giây. Video 30 khung/giây được lấy 1 trong 2 khung như bộ QEVD."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Không mở được video. Hãy thử file MP4, MOV, AVI hoặc WEBM khác.")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    if not 1.0 <= src_fps <= 240.0:
        src_fps = 30.0
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, round(src_fps / TARGET_FPS))
    options = vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=rt.POSE_MODEL),
        running_mode=vision.RunningMode.VIDEO, num_poses=1)
    kp2d, detected, size, i = [], [], None, 0
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if size is None:
                size = (frame.shape[1], frame.shape[0])                    # kích thước sau khi OpenCV xoay theo metadata
            if i % step == 0:
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = landmarker.detect_for_video(image, int(round(i * 1000 / src_fps)))
                if result.pose_landmarks:
                    kp2d.append([[l.x, l.y, l.z, l.visibility] for l in result.pose_landmarks[0]])
                    detected.append(True)
                else:
                    kp2d.append(np.zeros((33, 4)))
                    detected.append(False)
                if progress and n_total:
                    progress(min(1.0, i / n_total))
            i += 1
    cap.release()
    if not kp2d:
        raise ValueError("Không đọc được khung hình nào từ video này.")
    return dict(kp2d=np.asarray(kp2d, dtype=np.float32).reshape(-1, 33, 4), detected=np.asarray(detected, dtype=bool),
                fps=src_fps / step, src_fps=src_fps, step=step, n_src_frames=i, width=size[0], height=size[1])


# ----------------------------------------------------------------------------------------------------------------------
# Nhận diện góc quay, tín hiệu lên/xuống và đếm rep (giống hệt notebook 02 / 08)
# ----------------------------------------------------------------------------------------------------------------------
def detect_view(d):
    kp, det = d["kp2d"], d["detected"]
    px = kp[:, :, :2] * np.array([d["width"], d["height"]])
    ok = det & (kp[:, [11, 12, 23, 24], 3].min(1) > 0.5)
    if ok.sum() < 5:
        return "front", None
    shoulder_w = np.linalg.norm(px[ok, 11] - px[ok, 12], axis=-1)
    torso = np.linalg.norm(px[ok][:, [11, 12]].mean(1) - px[ok][:, [23, 24]].mean(1), axis=-1)
    ratio = float(np.median(shoulder_w / (torso + 1e-6)))
    return rt.classify_view(ratio), ratio


def joint_angle(a, b, c):
    ba, bc = a - b, c - b
    cos = (ba * bc).sum(-1) / (np.linalg.norm(ba, axis=-1) * np.linalg.norm(bc, axis=-1) + 1e-8)
    return np.degrees(np.arccos(np.clip(cos, -1, 1)))


def fill_gaps(x, valid):
    x = x.astype(np.float64).copy()
    idx = np.arange(len(x))
    if valid.sum() >= 2:
        x[~valid] = np.interp(idx[~valid], idx[valid], x[valid])
    return x


def near_side(vis):
    return 0 if vis[:, [L_SH, L_EL, L_WR]].mean() >= vis[:, [R_SH, R_EL, R_WR]].mean() else 1


def rep_signal(d, view):
    kp = d["kp2d"]
    px = kp[:, :, :2] * np.array([int(d["width"]), int(d["height"])])
    vis, det = kp[:, :, 3], d["detected"]

    def in_frame(j):
        return (kp[:, j, 0] > -0.1) & (kp[:, j, 0] < 1.1) & (kp[:, j, 1] > -0.1) & (kp[:, j, 1] < 1.1)

    if view == "front":
        sh_y = px[:, [L_SH, R_SH], 1].mean(1)
        wr_y = px[:, [L_WR, R_WR], 1].mean(1)
        sh_w = np.linalg.norm(px[:, L_SH] - px[:, R_SH], axis=-1)
        valid = det & (vis[:, [L_SH, R_SH]].min(1) > VIS_THR) & (vis[:, [L_WR, R_WR]].max(1) > VIS_THR) \
            & in_frame(L_SH) & in_frame(R_SH)
        scale = np.median(sh_w[valid]) if valid.any() else 1.0
        sig = (wr_y - sh_y) / (scale + 1e-6)
    else:
        side = near_side(vis)
        s, e, w = L_SH + side, L_EL + side, L_WR + side
        sig = joint_angle(px[:, s], px[:, e], px[:, w])
        valid = det & (vis[:, [s, e, w]].min(1) > VIS_THR) & in_frame(s) & in_frame(e) & in_frame(w)

    sig = fill_gaps(sig, valid)
    if len(sig) >= 7:
        sig = savgol_filter(sig, 7, 2)
    return sig, valid


def find_reps(sig, valid, fps, min_amp=0.0, min_prom=0.35, min_rep_s=0.8, max_rep_s=8.0, min_valid=0.7):
    if len(sig) < 3 or valid.mean() < 0.3:
        return []
    lo, hi = np.percentile(sig[valid], [5, 95])
    rng = hi - lo
    bottoms, _ = find_peaks(-sig, prominence=min_prom * rng, distance=max(1, int(min_rep_s * fps * 0.6)))
    half = int(max_rep_s * fps / 2)
    reps = []
    for i, b in enumerate(bottoms):
        left = bottoms[i - 1] if i > 0 else max(0, b - half)
        right = bottoms[i + 1] if i + 1 < len(bottoms) else min(len(sig) - 1, b + half)
        start = left + int(np.argmax(sig[left:b + 1]))
        end = b + int(np.argmax(sig[b:right + 1]))
        dur = (end - start) / fps
        amp = float(min(sig[start], sig[end]) - sig[b])
        deep = amp > 0.25 * rng and amp >= min_amp
        valid_frac = float(valid[start:end + 1].mean()) if end > start else 0.0
        reps.append(dict(start=int(start), bottom=int(b), end=int(end), dur=round(dur, 2),
                         valid_frac=round(valid_frac, 2), amp=round(amp, 3),
                         ok=bool(min_rep_s <= dur <= max_rep_s and deep and valid_frac >= min_valid)))
    return reps


# ----------------------------------------------------------------------------------------------------------------------
# Phân tích cả video
# ----------------------------------------------------------------------------------------------------------------------
def verdict_of(p, view):
    score = float(max(p[1], p[3])) if view == "front" else float(p[0])        # góc trước: hông bị che, không xét
    verdict = "sai" if score >= rt.T_SAI else "dung" if score <= rt.T_DUNG else "khong_chac"
    errors = []
    if verdict == "sai":
        errors = [e for j, e in enumerate(rt.ERR_NAMES[1:], 1) if p[j] >= rt.T_LOI and not (e == "hong" and view == "front")]
    return verdict, errors, score


def analyze(path, view="auto", progress=lambda stage, value: None):
    """Trả về (kết quả gửi cho giao diện, dữ liệu nội bộ dùng để xuất video)."""
    t0 = time.time()
    progress("Đang nạp mô hình", 0.02)
    models()
    progress("Đang trích khung xương", 0.05)
    d = extract_keypoints(path, lambda f: progress("Đang trích khung xương", 0.05 + 0.85 * f))
    progress("Đang đếm rep", 0.92)
    auto_view, ratio = detect_view(d)
    v = auto_view if view == "auto" else view
    fps = d["fps"]
    sig, valid = rep_signal(d, v)
    candidates = find_reps(sig, valid, fps, MIN_AMP[v])
    # Dao động quá nhỏ hoặc quá ngắn/dài không phải rep (ví dụ đứng yên ở tư thế chờ): bỏ hẳn, không đếm
    reps = [r for r in candidates if r["amp"] >= MIN_AMP[v] and 0.8 <= r["dur"] <= 8.0]

    progress("Đang chấm từng rep", 0.95)
    det = d["detected"]
    xy = d["kp2d"][:, rt.JOINTS, :2] * np.array([d["width"], d["height"]], dtype=np.float32)
    xyz = d["kp2d"][:, rt.JOINTS, :3].copy()
    vis = d["kp2d"][:, rt.JOINTS, 3].copy()
    if det.sum() >= 2:
        idx = np.arange(len(det))
        flat = xy.reshape(len(det), -1)
        for k in range(flat.shape[1]):
            flat[~det, k] = np.interp(idx[~det], idx[det], flat[det, k])
    vis[~det] = 0.0
    ok_reps = [r for r in reps if r["ok"]]
    probs = predict(np.stack([rt.make_sample(xy, xyz, vis, r["start"], r["end"])[0] for r in ok_reps]), v) if ok_reps else []

    rows, k = [], 0
    for n, r in enumerate(reps, 1):
        row = dict(number=n, start=round(r["start"] / fps, 2), bottom=round(r["bottom"] / fps, 2),
                   end=round(r["end"] / fps, 2), duration=r["dur"], amplitude=r["amp"],
                   verdict="khong_danh_gia", errors=[], score=None, p=None)
        if r["ok"]:
            p = probs[k]
            k += 1
            verdict, errors, score = verdict_of(p, v)
            row.update(verdict=verdict, errors=errors, score=round(score, 3),
                       p={name: round(float(p[j]), 3) for j, name in enumerate(rt.ERR_NAMES)})
        rows.append(row)

    frames = [[[round(float(x), 4), round(float(y), 4), round(float(c), 2)] for x, y, _, c in d["kp2d"][i, rt.JOINTS]]
              if det[i] else None for i in range(len(det))]
    counts = {key: sum(r["verdict"] == key for r in rows) for key in VERDICT_RT}
    result = dict(
        view=v, view_mode=view, view_auto=auto_view, view_ratio=None if ratio is None else round(ratio, 3),
        fps=round(fps, 4), src_fps=round(d["src_fps"], 4), width=d["width"], height=d["height"],
        duration=round(d["n_src_frames"] / d["src_fps"], 2), detect_rate=round(float(det.mean()), 3),
        frames=frames, signal=[round(float(s), 3) for s in sig], reps=rows, counts=counts,
        elapsed=round(time.time() - t0, 1), edges=rt.EDGES)
    internal = dict(kp2d=d["kp2d"], detected=det, step=d["step"], fps=fps, src_fps=d["src_fps"])
    progress("Xong", 1.0)
    return result, internal


# ----------------------------------------------------------------------------------------------------------------------
# Xuất video (WebM VP8: trình duyệt nào cũng phát được, OpenCV ghi được mà không cần thư viện ngoài)
# ----------------------------------------------------------------------------------------------------------------------
def render(path, result, internal, out_path, annotate=True, max_side=960, progress=lambda value: None):
    cap = cv2.VideoCapture(path)
    src_fps, step = internal["src_fps"], internal["step"]
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    kp2d, det = internal["kp2d"], internal["detected"]
    ends = [(round(r["end"] * internal["fps"]) * step, r) for r in result["reps"]]
    writer, i = None, 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if writer is None:
            h, w = frame.shape[:2]
            scale = min(1.0, max_side / max(w, h))
            size = (int(w * scale) // 2 * 2, int(h * scale) // 2 * 2)
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"VP80"), src_fps, size)
            if not writer.isOpened():
                raise RuntimeError("OpenCV không ghi được video WebM trên máy này.")
        frame = cv2.resize(frame, size)
        if annotate:
            k = min(len(det) - 1, int(round(i / step)))
            done = [r for e, r in ends if e <= i]
            last = None
            if done:
                r = done[-1]
                last = (r["number"], VERDICT_RT[r["verdict"]], r["errors"])
            rt.draw(frame, kp2d[k] if det[k] else None, len(done), result["view"],
                    "auto" if result["view_mode"] == "auto" else "manual", last, src_fps, "full")
        writer.write(frame)
        i += 1
        if n_total and i % 10 == 0:
            progress(min(1.0, i / n_total))
    cap.release()
    if writer is None:
        raise RuntimeError("Không đọc được khung hình nào để xuất.")
    writer.release()
    progress(1.0)

"""Chấm điểm chống đẩy trực tiếp bằng camera.

Cách chạy (trong thư mục dự án, dùng Python của .venv):
    .venv\\Scripts\\python realtime\\chay_camera.py                    # camera mặc định
    .venv\\Scripts\\python realtime\\chay_camera.py --source 1         # camera thứ hai
    .venv\\Scripts\\python realtime\\chay_camera.py --source video.mp4 # chạy trên file video như thể là camera

Phím: q = thoát, r = đếm lại từ đầu, v = đổi góc quay bằng tay (auto -> front -> side -> diagonal).

Cửa sổ hiển thị khung xương, số rep, góc quay và kết luận của rep vừa xong. Kết luận xuất hiện ngay sau khi
bạn đẩy lên hết một rep. Mô hình, cách chuẩn hoá và ngưỡng kết luận giống hệt notebook 08; phần khác duy nhất là
bộ đếm rep: notebook nhìn cả video rồi mới tìm rep, còn ở đây rep được phát hiện ngay khi nó xảy ra.

File này được sinh tự động từ code của các notebook (bước 3, 7, 8) để bảo đảm xử lý giống hệt lúc train.
"""
import argparse, collections, csv, os, signal, sys, time

import cv2
import mediapipe as mp
import numpy as np
import torch
import torch.nn as nn
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HERE = os.path.dirname(os.path.abspath(__file__))
POSE_MODEL = os.path.join(HERE, "pose_landmarker_full.task")
WEIGHTS = [os.path.join(HERE, "weights", f"featcnn_seed{s}.pt") for s in (0, 1, 2)]

# Ngưỡng kết luận (giống notebook 08)
T_SAI, T_DUNG, T_LOI = 0.60, 0.40, 0.50
VIS_THR = 0.5
VIEW_NAMES = ["front", "side", "diagonal"]
ERR_NAMES = ["sai", "nong", "hong", "sap"]
ERR_ASCII = {"nong": "xuong chua du sau", "hong": "hong vong / mong nho", "sap": "do sap / chan cham san"}

# Biên độ tối thiểu để một dao động được coi là rep. Góc trước: độ cao vai tính theo độ rộng vai.
# Góc ngang/chéo: góc khuỷu (độ). Chạy trực tiếp cần ngưỡng chặt hơn bản xử lý video (3 độ) để nhiễu không thành rep.
RT_MIN_AMP = {"front": 0.15, "side": 15.0, "diagonal": 15.0}
MIN_REP_S, MAX_REP_S = 0.6, 8.0      # thời lượng tính từ lúc rời đỉnh tới lúc lên lại đỉnh, nên ngắn hơn mốc 0,8 giây (đỉnh tới đỉnh) của bản xử lý video

L_SH, R_SH, L_EL, R_EL, L_WR, R_WR, L_HIP, R_HIP = 11, 12, 13, 14, 15, 16, 23, 24

T = 32
JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
JOINT_NAMES = ["L_shoulder", "R_shoulder", "L_elbow", "R_elbow", "L_wrist", "R_wrist",
               "L_hip", "R_hip", "L_knee", "R_knee", "L_ankle", "R_ankle"]
# Cạnh của đồ thị khung xương (chỉ số theo JOINTS), dùng cho ST-GCN và để vẽ
EDGES = [(0, 1), (0, 2), (2, 4), (1, 3), (3, 5), (0, 6), (1, 7), (6, 7), (6, 8), (8, 10), (7, 9), (9, 11)]
SH, HIP = [0, 1], [6, 7]
VIEW_ID = {"front": 0, "side": 1, "diagonal": 2}
LABEL_ID = {"good": 0, "bad": 1}


def resample(seq, n):
    pos = np.linspace(0, len(seq) - 1, n)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(seq) - 1)
    w = (pos - lo).reshape(-1, *[1] * (seq.ndim - 1))
    return seq[lo] * (1 - w) + seq[hi] * w


def normalize_2d(xy):
    mid_sh, mid_hip = xy[:, SH].mean(1), xy[:, HIP].mean(1)
    origin = np.median(mid_hip, axis=0)
    torso = np.median(np.linalg.norm(mid_sh - mid_hip, axis=-1))
    sh_width = np.median(np.linalg.norm(xy[:, SH[0]] - xy[:, SH[1]], axis=-1))
    out = (xy - origin) / (max(torso, sh_width) + 1e-6)
    return np.clip(out, -3, 3)      # chặn các khớp bị MediaPipe đoán văng ra xa (99% giá trị nằm trong ±1.8)


def make_sample(xy, xyz, vis, start, end):
    sl = slice(start, end + 1)
    v = resample(vis[sl], T)[..., None]
    x2d = np.concatenate([resample(normalize_2d(xy[sl]), T), v], axis=-1)     # (T, 12, 3): x, y, visibility
    x3d = np.concatenate([resample(xyz[sl], T), v], axis=-1)                  # (T, 12, 4): x, y, z, visibility
    return x2d.astype(np.float32), x3d.astype(np.float32)


def torch_angle(a, b, c):
    ba, bc = a - b, c - b
    cos = (ba * bc).sum(-1) / (ba.norm(dim=-1) * bc.norm(dim=-1) + 1e-6)
    return torch.acos(cos.clamp(-1 + 1e-6, 1 - 1e-6)) / np.pi           # về khoảng 0..1


def frame_features(x):
    # x: (B, T, 12, 3) -> (B, 28, T): 16 đặc trưng hình học + 12 kênh visibility cho từng khung
    P, V = x[..., :2], x[..., 2]
    f = []
    for s in (0, 1):                                                    # trái, phải
        sh, el, wr, hip, kn, an = [P[:, :, j + s] for j in (0, 2, 4, 6, 8, 10)]
        f += [torch_angle(sh, el, wr), torch_angle(sh, hip, an), torch_angle(hip, kn, an), torch_angle(el, sh, hip)]
    y = lambda joints: P[:, :, joints, 1].mean(2)
    sh_y, el_y, wr_y, hip_y = y([0, 1]), y([2, 3]), y([4, 5]), y([6, 7])
    f += [wr_y - sh_y, hip_y - sh_y, el_y - sh_y,
          sh_y - sh_y.mean(1, keepdim=True), hip_y - hip_y.mean(1, keepdim=True),
          (P[:, :, 4] - P[:, :, 5]).norm(dim=-1), (P[:, :, 2] - P[:, :, 3]).norm(dim=-1), (P[:, :, 0] - P[:, :, 1]).norm(dim=-1)]
    return torch.cat([torch.stack(f, dim=1), V.permute(0, 2, 1)], dim=1)


class FeatCNN(nn.Module):
    def __init__(self, n_out=4, c=64, dropout=0.3):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(28)
        self.net = nn.Sequential(
            nn.Conv1d(28, c, 5, padding=2), nn.BatchNorm1d(c), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(c, c, 5, padding=2), nn.BatchNorm1d(c), nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(c, c, 3, padding=1), nn.BatchNorm1d(c), nn.ReLU())
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(2 * c + 3, n_out))

    def forward(self, x, view_id):
        h = self.net(self.input_bn(frame_features(x)))
        return self.head(torch.cat([h.mean(2), h.amax(2), nn.functional.one_hot(view_id, 3).float()], dim=1))


# ----------------------------------------------------------------------------------------------------------------------
# Mô hình
# ----------------------------------------------------------------------------------------------------------------------
def load_models():
    models = []
    for path in WEIGHTS:
        model = FeatCNN()
        model.load_state_dict(torch.load(path, map_location="cpu"))
        models.append(model.eval())
    return models


@torch.no_grad()
def predict_rep(models, x2d, view):
    x = torch.from_numpy(x2d[None])
    v = torch.tensor([VIEW_NAMES.index(view)])
    return torch.stack([torch.sigmoid(m(x, v)) for m in models]).mean(0)[0].numpy()      # [sai, nong, hong, sap]


# ----------------------------------------------------------------------------------------------------------------------
# Tín hiệu lên/xuống của từng khung hình (giống notebook 02, nhưng tính dần theo thời gian)
# ----------------------------------------------------------------------------------------------------------------------
def joint_angle(a, b, c):
    ba, bc = a - b, c - b
    cos = float((ba * bc).sum() / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-8))
    return float(np.degrees(np.arccos(np.clip(cos, -1, 1))))


def in_frame(p):
    return -0.1 < p[0] < 1.1 and -0.1 < p[1] < 1.1


def view_ratio(kp, w, h):
    """Tỉ lệ độ rộng vai / chiều dài thân trên ảnh; None nếu vai hoặc hông không nhìn rõ."""
    if kp[[L_SH, R_SH, L_HIP, R_HIP], 3].min() <= VIS_THR:
        return None
    px = kp[:, :2] * np.array([w, h])
    torso = np.linalg.norm(px[[L_SH, R_SH]].mean(0) - px[[L_HIP, R_HIP]].mean(0))
    return float(np.linalg.norm(px[L_SH] - px[R_SH]) / (torso + 1e-6))


def classify_view(ratio):
    return "side" if ratio < 0.28 else "front" if ratio > 0.95 else "diagonal"


class Signal:
    """Tính tín hiệu lên/xuống cho từng khung và làm mượt bằng trung bình 3 khung gần nhất."""

    def __init__(self):
        self.recent = collections.deque(maxlen=3)
        self.shoulder_w = collections.deque(maxlen=90)     # để lấy độ rộng vai trung vị (góc trước)
        self.side_votes = collections.deque(maxlen=90)     # bên tay nào nhìn rõ hơn (góc ngang/chéo)

    def update(self, kp, w, h, view):
        """Trả về (giá trị tín hiệu đã làm mượt hoặc None, khung này có hợp lệ không)."""
        px = kp[:, :2] * np.array([w, h])
        vis = kp[:, 3]
        if view == "front":
            valid = vis[[L_SH, R_SH]].min() > VIS_THR and vis[[L_WR, R_WR]].max() > VIS_THR and in_frame(kp[L_SH]) and in_frame(kp[R_SH])
            if valid:
                self.shoulder_w.append(np.linalg.norm(px[L_SH] - px[R_SH]))
            if len(self.shoulder_w) < 10:                   # chờ đủ 10 khung để độ rộng vai trung vị ổn định
                return None, False
            raw = (px[[L_WR, R_WR], 1].mean() - px[[L_SH, R_SH], 1].mean()) / (np.median(self.shoulder_w) + 1e-6)
        else:
            self.side_votes.append(vis[[L_SH, L_EL, L_WR]].mean() >= vis[[R_SH, R_EL, R_WR]].mean())
            side = 0 if np.mean(self.side_votes) >= 0.5 else 1
            s, e, wr = L_SH + side, L_EL + side, L_WR + side
            valid = vis[[s, e, wr]].min() > VIS_THR and in_frame(kp[s]) and in_frame(kp[e]) and in_frame(kp[wr])
            raw = joint_angle(px[s], px[e], px[wr])
        if not valid:
            return (float(np.mean(self.recent)) if self.recent else None), False      # giữ giá trị cũ, không cập nhật
        self.recent.append(raw)
        return float(np.mean(self.recent)), True


# ----------------------------------------------------------------------------------------------------------------------
# Bộ đếm rep trực tiếp: máy trạng thái "ở trên -> đang xuống -> đang lên -> xong một rep"
# ----------------------------------------------------------------------------------------------------------------------
class RepCounter:
    TOP_WINDOW_S = 1.5      # "điểm cao nhất" là giá trị lớn nhất trong 1,5 giây gần đây, để không bám vào một đỉnh cũ

    def __init__(self, min_amp):
        self.min_amp = min_amp
        self.state = "top"
        self.window = collections.deque()                   # (giá trị, chỉ số khung, thời điểm) trong lúc đang ở trên
        self.top = self.bottom = self.peak = None

    def _back_to_top(self, *samples):
        self.state = "top"
        self.window = collections.deque(samples)

    def update(self, sig, idx, t):
        """Đưa vào tín hiệu của khung mới. Trả về (start_idx, bottom_idx, end_idx, biên độ, thời lượng) khi một rep vừa kết thúc."""
        if self.state == "top":
            self.window.append((sig, idx, t))
            while t - self.window[0][2] > self.TOP_WINDOW_S:
                self.window.popleft()
            top_val = max(x[0] for x in self.window)
            near_top = [x for x in self.window if top_val - x[0] < 0.25 * self.min_amp]
            self.top = (top_val, near_top[-1][1], near_top[-1][2])            # rep bắt đầu từ khung cuối cùng còn ở sát đỉnh
            if top_val - sig >= self.min_amp:
                self.state, self.bottom = "down", (sig, idx, t)
        elif self.state == "down":
            if sig < self.bottom[0]:
                self.bottom = (sig, idx, t)
            elif sig - self.bottom[0] >= self.min_amp:
                self.state, self.peak = "rising", (sig, idx, t)
            if t - self.top[2] > MAX_REP_S:                                 # xuống quá lâu không lên: huỷ
                self._back_to_top((sig, idx, t))
        elif self.state == "rising":
            if sig > self.peak[0]:
                self.peak = (sig, idx, t)
            elif sig < self.bottom[0]:                                      # chưa lên hết đã xuống sâu hơn: vẫn là rep cũ
                self.state, self.bottom = "down", (sig, idx, t)
            else:
                rise = self.peak[0] - self.bottom[0]
                going_down_again = self.peak[0] - sig >= max(0.15 * rise, 0.5 * self.min_amp)
                resting_at_top = t - self.peak[2] >= 0.4
                if going_down_again or resting_at_top:                      # đã qua đỉnh: kết thúc một lần lên
                    # Chỉ tính là rep nếu lên lại được ít nhất nửa quãng đã xuống. Nếu không (nhún nhẹ ở đáy, hoặc
                    # "đỉnh" trước đó là giá trị nhiễu quá cao) thì bỏ qua và lấy đỉnh vừa rồi làm mốc mới.
                    full = rise >= 0.5 * (self.top[0] - self.bottom[0])
                    rep = (self.top[1], self.bottom[1], self.peak[1], min(self.top[0], self.peak[0]) - self.bottom[0],
                           self.peak[2] - self.top[2]) if full else None
                    self._back_to_top(self.peak, (sig, idx, t))
                    return rep
        return None


# ----------------------------------------------------------------------------------------------------------------------
# Chấm một rep
# ----------------------------------------------------------------------------------------------------------------------
def judge_rep(models, frames, view, w, h):
    """frames: danh sách (kp (33,4) hoặc None, hợp lệ). Trả về (kết luận, danh sách lỗi, xác suất 4 đầu ra, điểm kết luận)."""
    valid_frac = float(np.mean([ok for _, ok in frames]))
    det = np.array([kp is not None for kp, _ in frames])
    if valid_frac < 0.7 or det.sum() < 2:
        return "khong danh gia duoc", [], None, None
    kp = np.stack([k if k is not None else np.zeros((33, 4), np.float32) for k, _ in frames])
    xy = kp[:, JOINTS, :2] * np.array([w, h], dtype=np.float32)
    vis = kp[:, JOINTS, 3].copy()
    idx = np.arange(len(det))
    flat = xy.reshape(len(det), -1)
    for k in range(flat.shape[1]):                                          # nội suy khung không phát hiện được người
        flat[~det, k] = np.interp(idx[~det], idx[det], flat[det, k])
    vis[~det] = 0.0
    x2d, _ = make_sample(xy, np.zeros((len(det), len(JOINTS), 3), np.float32), vis, 0, len(det) - 1)
    p = predict_rep(models, x2d, view)
    # Điểm dùng để kết luận. Ở góc trước, hông bị thân che nên đầu ra "hong" không đáng tin (AUC khoảng 0,6 ở bước 7),
    # và nó kéo theo cả đầu ra "sai" chung: người tập đúng vẫn bị báo sai mà không nêu được lỗi gì. Vì vậy ở góc trước
    # chỉ xét hai lỗi nhìn thấy được là xuống nông và đổ sập. Ở góc ngang/chéo dùng đầu ra "sai" chung như cũ.
    score = float(max(p[1], p[3])) if view == "front" else float(p[0])
    verdict = "sai" if score >= T_SAI else "dung" if score <= T_DUNG else "khong chac"
    errors = []
    if verdict == "sai":
        errors = [e for j, e in enumerate(ERR_NAMES[1:], 1) if p[j] >= T_LOI and not (e == "hong" and view == "front")]
    return verdict, errors, p, score


# ----------------------------------------------------------------------------------------------------------------------
# Vòng lặp chính
# ----------------------------------------------------------------------------------------------------------------------
COLOR = {"dung": (0, 200, 0), "sai": (0, 0, 255), "khong chac": (0, 200, 255), "khong danh gia duoc": (160, 160, 160)}

# Các xương được vẽ (chỉ số MediaPipe). Thân trên luôn vẽ khi nhìn rõ.
L_KNEE, R_KNEE, L_ANK, R_ANK = 25, 26, 27, 28
BONES_UPPER = [(L_SH, R_SH), (L_SH, L_EL), (L_EL, L_WR), (R_SH, R_EL), (R_EL, R_WR), (L_SH, L_HIP), (R_SH, R_HIP), (L_HIP, R_HIP)]
BONES_THIGH = [(L_HIP, L_KNEE), (R_HIP, R_KNEE)]
BONES_SHIN = [(L_KNEE, L_ANK), (R_KNEE, R_ANK)]


class SkeletonSmoother:
    """Làm mượt vị trí khớp CHỈ để vẽ (mô hình vẫn nhận keypoint gốc), cho khung xương bớt rung."""

    def __init__(self, alpha=0.5):
        self.alpha, self.prev = alpha, None

    def update(self, kp):
        if kp is None:
            self.prev = None
            return None
        if self.prev is None:
            self.prev = kp.copy()
        else:
            self.prev[:, :3] = self.alpha * kp[:, :3] + (1 - self.alpha) * self.prev[:, :3]
            self.prev[:, 3] = kp[:, 3]
        return self.prev


def draw_skeleton(frame, kp, view, mode="full"):
    """Khung xương xanh lá sáng có quầng, khớp là chấm trắng.

    mode="full" (mặc định): luôn vẽ đủ 12 khớp ở mọi góc quay. Xương có một đầu khớp nhìn không rõ (MediaPipe chỉ đoán
    vị trí, ví dụ chân ở góc trước bị thân che) được vẽ mảnh, màu xám, để người xem biết đó là vị trí ước lượng.
    mode="clear": chỉ vẽ xương nhìn rõ; ở góc trước không vẽ ống chân.
    """
    h, w = frame.shape[:2]
    p = (kp[:, :2] * np.array([w, h])).astype(int)
    vis = kp[:, 3]
    if mode == "clear":
        bones = BONES_UPPER + BONES_THIGH + ([] if view in (None, "front") else BONES_SHIN)
    else:
        bones = BONES_UPPER + BONES_THIGH + BONES_SHIN
    sure = [(a, b) for a, b in bones if min(vis[a], vis[b]) > VIS_THR]
    unsure = [] if mode == "clear" else [bone for bone in bones if bone not in sure]
    if not sure and not unsure:
        return
    thick = max(2, round(min(w, h) / 150))
    for a, b in unsure:                                                     # vị trí ước lượng: mảnh, xám, không quầng
        cv2.line(frame, tuple(p[a]), tuple(p[b]), (190, 190, 190), max(1, thick // 2 + 1), cv2.LINE_AA)
    if sure:
        glow = frame.copy()
        for a, b in sure:
            cv2.line(glow, tuple(p[a]), tuple(p[b]), (0, 255, 0), thick * 4, cv2.LINE_AA)
        cv2.addWeighted(glow, 0.35, frame, 0.65, 0, frame)                  # quầng sáng quanh xương
        for a, b in sure:
            cv2.line(frame, tuple(p[a]), tuple(p[b]), (110, 255, 110), thick, cv2.LINE_AA)
    sure_joints = {j for bone in sure for j in bone}
    for j in sorted({j for bone in sure + unsure for j in bone}):
        if j in sure_joints:
            cv2.circle(frame, tuple(p[j]), thick + 3, (0, 200, 0), -1, cv2.LINE_AA)
            cv2.circle(frame, tuple(p[j]), thick + 1, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            cv2.circle(frame, tuple(p[j]), thick + 1, (190, 190, 190), -1, cv2.LINE_AA)


class Recorder:
    """Ghi video ra file. Với camera, tốc độ khung hình thật được đo từ 30 khung đầu (camera thường báo sai hoặc không báo).

    Đuôi .avi dùng codec MJPG (file lớn nhưng trình phát nào của Windows cũng mở được); đuôi khác dùng mp4v.
    """

    def __init__(self, path, fps=None):
        self.path, self.fps, self.writer, self.pending, self.failed = path, fps, None, [], False

    def _open(self, size):
        fourcc = cv2.VideoWriter_fourcc(*("MJPG" if self.path.lower().endswith(".avi") else "mp4v"))
        self.writer = cv2.VideoWriter(self.path, fourcc, self.fps, size)
        if not self.writer.isOpened():
            print(f"CẢNH BÁO: không ghi được video ra {self.path}. Thử đổi đuôi file sang .avi.", flush=True)
            self.writer, self.failed = None, True
        else:
            for frame, _ in self.pending:
                self.writer.write(frame)
        self.pending = []

    def write(self, frame, t):
        if self.failed:
            return
        if self.writer is not None:
            self.writer.write(frame)
            return
        self.pending.append((frame.copy(), t))
        if self.fps is None and len(self.pending) >= 30:
            span = self.pending[-1][1] - self.pending[0][1]
            self.fps = float(np.clip((len(self.pending) - 1) / max(span, 1e-6), 5, 60))
        if self.fps is not None:
            self._open((frame.shape[1], frame.shape[0]))

    def close(self):
        if self.writer is None and self.pending and not self.failed:        # quay quá ngắn, chưa kịp đo tốc độ
            self.fps = self.fps or 30.0
            self._open((self.pending[0][0].shape[1], self.pending[0][0].shape[0]))
        if self.writer is not None:
            self.writer.release()
            print(f"Đã lưu video: {self.path} ({self.fps:.1f} khung/giây)", flush=True)


def put_text(frame, text, org, scale, color, thickness=2):
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)       # viền đen cho dễ đọc
    cv2.putText(frame, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw(frame, kp, count, view, view_mode, last, fps, skeleton="full"):
    h, w = frame.shape[:2]
    if kp is not None:
        draw_skeleton(frame, kp, view, skeleton)
    k = min(w, h) / 480                                                     # cỡ chữ theo kích thước khung hình
    bar = frame[: int(70 * k)].copy()
    cv2.addWeighted(bar, 0.35, np.zeros_like(bar), 0.65, 0, frame[: int(70 * k)])    # thanh nền mờ phía trên
    view_text = (view or "dang nhan dien...") + (" (tu dong)" if view_mode == "auto" else " (chon tay)")
    put_text(frame, f"goc quay: {view_text}   {fps:.0f} fps", (int(10 * k), int(24 * k)), 0.55 * k, (255, 255, 255), max(1, round(1.5 * k)))
    if last is not None:
        number, verdict, errors = last
        text = f"Rep {number}: {verdict.upper()}" + (f" ({', '.join(ERR_ASCII[e] for e in errors)})" if errors else "")
        put_text(frame, text, (int(10 * k), int(56 * k)), 0.7 * k, COLOR[verdict], max(2, round(2 * k)))
    number = str(count)                                                     # số rep cỡ lớn ở góc trên bên phải
    (tw, th), _ = cv2.getTextSize(number, cv2.FONT_HERSHEY_SIMPLEX, 2.6 * k, max(3, round(5 * k)))
    put_text(frame, number, (w - tw - int(14 * k), int(78 * k) + th), 2.6 * k, (255, 255, 255), max(3, round(5 * k)))
    return frame


def main():
    global T_SAI, T_DUNG
    ap = argparse.ArgumentParser(description="Chấm điểm chống đẩy trực tiếp bằng camera")
    ap.add_argument("--source", default="0", help="số thứ tự camera (0, 1, ...) hoặc đường dẫn file video")
    ap.add_argument("--view", default="auto", choices=["auto", "front", "side", "diagonal"], help="góc quay; auto = tự nhận diện")
    ap.add_argument("--backend", default="auto", choices=["auto", "dshow", "msmf"], help="cách mở camera trên Windows (auto: thử DirectShow trước)")
    ap.add_argument("--no-window", action="store_true", help="không mở cửa sổ (dùng để kiểm thử trên file video)")
    ap.add_argument("--save", default=None, help="lưu video có chú thích ra file .mp4")
    ap.add_argument("--log", default=None, help="lưu kết luận từng rep ra file .csv")
    ap.add_argument("--record", default=None, help="lưu hình camera GỐC (chưa vẽ gì) ra file .mp4, để phân tích lại sau")
    ap.add_argument("--skeleton", default="full", choices=["full", "clear"],
                    help="full: luôn vẽ đủ 12 khớp, khớp chỉ được đoán vị trí vẽ màu xám; clear: chỉ vẽ khớp nhìn rõ")
    ap.add_argument("--t-sai", type=float, default=T_SAI, help=f"xác suất từ mức này trở lên thì kết luận 'sai' (mặc định {T_SAI})")
    ap.add_argument("--t-dung", type=float, default=T_DUNG, help=f"xác suất từ mức này trở xuống thì kết luận 'đúng' (mặc định {T_DUNG})")
    args = ap.parse_args()
    T_SAI, T_DUNG = args.t_sai, args.t_dung

    is_camera = args.source.isdigit()
    if is_camera:
        # Trên Windows, backend mặc định (MSMF) có thể mất cả phút để mở camera; DirectShow mở gần như ngay.
        backends = {"dshow": [cv2.CAP_DSHOW], "msmf": [cv2.CAP_MSMF], "auto": [cv2.CAP_DSHOW, cv2.CAP_ANY] if os.name == "nt" else [cv2.CAP_ANY]}[args.backend]
        cap = None
        for backend in backends:
            print(f"Đang mở camera {args.source} ({'DirectShow' if backend == cv2.CAP_DSHOW else 'MSMF' if backend == cv2.CAP_MSMF else 'mặc định'})...", flush=True)
            cap = cv2.VideoCapture(int(args.source), backend)
            if cap.isOpened() and cap.read()[0]:
                break
            cap.release()
            cap = None
        if cap is None:
            sys.exit(f"Không mở được camera {args.source}. Kiểm tra: máy có camera không, có ứng dụng khác đang dùng camera không "
                     f"(Zoom, Teams, trình duyệt), Windows có cho phép ứng dụng dùng camera không (Settings > Privacy > Camera). "
                     f"Thử camera khác bằng --source 1.")
    else:
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            sys.exit(f"Không mở được file video: {args.source}")
    file_fps = cap.get(cv2.CAP_PROP_FPS)
    if not 1 <= file_fps <= 240:                            # camera hay báo 0 hoặc -1
        file_fps = 30.0
    print(f"Đã mở nguồn video: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}. Đang tải mô hình...", flush=True)

    models = load_models()
    options = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=POSE_MODEL),
                                           running_mode=vision.RunningMode.VIDEO, num_poses=1)
    landmarker = vision.PoseLandmarker.create_from_options(options)
    if not args.no_window:
        print("Sẵn sàng. Cửa sổ camera sẽ hiện ra; bấm vào cửa sổ rồi nhấn q để thoát.", flush=True)

    def reset():
        return dict(signal=Signal(), counter=None, buffer={}, count=0, last=None, ratios=collections.deque(maxlen=45), switch=0,
                    view=None if args.view == "auto" else args.view)

    view_mode = args.view
    st = reset()
    smoother = SkeletonSmoother()
    results, idx, t0, fps_show, t_prev = [], 0, time.perf_counter(), 0.0, None
    rec_fps = None if is_camera else file_fps               # camera: để Recorder tự đo
    raw_rec = Recorder(args.record, rec_fps) if args.record else None
    out_rec = Recorder(args.save, rec_fps) if args.save else None

    WINDOW = "Cham diem chong day (q: thoat, r: dem lai, v: doi goc quay)"
    stop = [False]
    signal.signal(signal.SIGINT, lambda *_: stop.__setitem__(0, True))      # Ctrl+C: thoát êm để file video được đóng đúng cách

    while not stop[0]:
        ok, frame = cap.read()
        if not ok:
            break
        t = (time.perf_counter() - t0) if is_camera else idx / file_fps       # file video: thời gian theo số khung
        h, w = frame.shape[:2]
        if raw_rec is not None:
            raw_rec.write(frame, t)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        res = landmarker.detect_for_video(image, int(t * 1000) + idx)       # + idx: bảo đảm timestamp luôn tăng
        kp = np.array([[l.x, l.y, l.z, l.visibility] for l in res.pose_landmarks[0]], dtype=np.float32) if res.pose_landmarks else None

        # 1) Nhận diện góc quay liên tục: trung vị tỉ lệ vai/thân của 45 khung gần nhất. Nếu kết quả khác góc đang dùng
        #    suốt 20 khung liền (ví dụ lúc đầu người tập còn đang ngồi, sau mới vào tư thế) thì đổi góc, giữ nguyên số rep.
        if view_mode == "auto" and kp is not None:
            r = view_ratio(kp, w, h)
            if r is not None:
                st["ratios"].append(r)
            if len(st["ratios"]) >= 20:
                candidate = classify_view(float(np.median(st["ratios"])))
                if st["view"] is None:
                    st["view"] = candidate
                elif candidate == st["view"]:
                    st["switch"] = 0
                else:
                    st["switch"] += 1
                    between_reps = st["counter"] is None or st["counter"].state == "top"
                    if st["switch"] >= 20 and between_reps:
                        print(f"Đổi góc quay: {st['view']} -> {candidate}", flush=True)
                        st.update(view=candidate, signal=Signal(), counter=None, buffer={}, switch=0)

        # 2) Tín hiệu lên/xuống và bộ đếm rep
        if st["view"] is not None:
            if st["counter"] is None:
                st["counter"] = RepCounter(RT_MIN_AMP[st["view"]])
            sig, valid = st["signal"].update(kp, w, h, st["view"]) if kp is not None else (None, False)
            st["buffer"][idx] = (kp, valid)
            for old in [k for k in st["buffer"] if k < idx - int(MAX_REP_S * 40)]:
                del st["buffer"][old]
            rep = st["counter"].update(sig, idx, t) if (sig is not None and valid) else None
            if rep is not None:
                start, bottom, end, amp, dur = rep
                if MIN_REP_S <= dur <= MAX_REP_S and all(k in st["buffer"] for k in (start, end)):
                    frames = [st["buffer"].get(k, (None, False)) for k in range(start, end + 1)]
                    verdict, errors, p, score = judge_rep(models, frames, st["view"], w, h)
                    st["count"] += 1
                    st["last"] = (st["count"], verdict, errors)
                    row = dict(rep=st["count"], bat_dau_s=round(start / file_fps, 2) if not is_camera else "", ket_thuc_s=round(t, 2),
                               thoi_luong_s=round(dur, 2), goc_quay=st["view"], ket_luan=verdict, loi=", ".join(errors),
                               p_sai="" if p is None else round(float(p[0]), 3), p_nong="" if p is None else round(float(p[1]), 3),
                               p_hong="" if p is None else round(float(p[2]), 3), p_sap="" if p is None else round(float(p[3]), 3))
                    results.append(row)
                    row["diem_ket_luan"] = "" if score is None else round(score, 3)
                    row["bien_do"] = round(float(amp), 3)
                    print(f"Rep {row['rep']}: {verdict}" + (f" ({', '.join(ERR_ASCII[e] for e in errors)})" if errors else "")
                          + ("" if p is None else f"  diem={score:.2f} (sai={p[0]:.2f} nong={p[1]:.2f} hong={p[2]:.2f} sap={p[3]:.2f})")
                          + f" | bien do={amp:.2f} | {dur:.1f}s | goc quay={st['view']}", flush=True)

        # 3) Hiển thị
        now = time.perf_counter()
        if t_prev is not None:
            fps_show = 0.9 * fps_show + 0.1 / max(now - t_prev, 1e-6) if fps_show else 1 / max(now - t_prev, 1e-6)
        t_prev = now
        if not args.no_window or args.save:
            shown = draw(frame, smoother.update(kp), st["count"], st["view"], view_mode, st["last"], fps_show, args.skeleton)
            if out_rec is not None:
                out_rec.write(shown, t)
            if not args.no_window:
                cv2.imshow(WINDOW, shown)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:      # nhấn q hoặc bấm X đóng cửa sổ
                    break
                if key == ord("r"):
                    st = reset()
                    if view_mode != "auto":
                        st["view"] = view_mode
                if key == ord("v"):
                    view_mode = ["auto", "front", "side", "diagonal"][(["auto", "front", "side", "diagonal"].index(view_mode) + 1) % 4]
                    st = reset()
                    st["view"] = None if view_mode == "auto" else view_mode
        idx += 1

    cap.release()
    landmarker.close()
    for recorder in (raw_rec, out_rec):
        if recorder is not None:
            recorder.close()
    if not args.no_window:
        cv2.destroyAllWindows()
    if args.log and results:
        with open(args.log, "w", newline="", encoding="utf-8-sig") as f:
            wr = csv.DictWriter(f, fieldnames=list(results[0].keys()))
            wr.writeheader(); wr.writerows(results)
    counts = collections.Counter(r["ket_luan"] for r in results)
    print(f"Tổng: {len(results)} rep | góc quay: {st['view']} | {dict(counts)}")


if __name__ == "__main__":
    main()

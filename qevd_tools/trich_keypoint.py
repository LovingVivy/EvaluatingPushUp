"""Trích keypoint MediaPipe cho các clip chống đẩy QEVD đã rút (pushup_clips.json).

Cách dùng:  python qevd_tools/trich_keypoint.py [số tiến trình]
Lấy 1 trong 2 khung hình (30 fps -> 15 fps, bằng fps của bộ Mendeley). Chạy lại sẽ bỏ qua clip đã xong.
Cần: pip install mediapipe opencv-python-headless, và file pose_landmarker_full.task đặt trong qevd_tools/.
Kết quả: Data/QEVD-FIT-COACH/keypoints_pushups/<clip>.npz, cùng định dạng với notebook 01.
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.join(os.path.dirname(HERE), "Data", "QEVD-FIT-COACH")
OUT = os.path.join(D, "keypoints_pushups")
MODEL = os.path.join(HERE, "pose_landmarker_full.task")
STEP = 2
EXERCISE = "pushups"


def work(video):
    import cv2, mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision
    out_path = os.path.join(OUT, video.replace(".mp4", ".npz"))
    options = vision.PoseLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=MODEL),
                                           running_mode=vision.RunningMode.VIDEO, num_poses=1)
    cap = cv2.VideoCapture(os.path.join(D, "short_clips", video))
    fps = (cap.get(cv2.CAP_PROP_FPS) or 30.0) / STEP
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    kp2d, kp3d, detected = [], [], []
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if i % STEP == 0:
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                result = landmarker.detect_for_video(image, int((i // STEP) * 1000 / fps))
                if result.pose_landmarks:
                    kp2d.append([[l.x, l.y, l.z, l.visibility] for l in result.pose_landmarks[0]])
                    kp3d.append([[l.x, l.y, l.z, l.visibility] for l in result.pose_world_landmarks[0]])
                    detected.append(True)
                else:
                    kp2d.append(np.zeros((33, 4))); kp3d.append(np.zeros((33, 4))); detected.append(False)
            i += 1
    cap.release()
    tmp = out_path + ".tmp.npz"
    np.savez_compressed(tmp, kp2d=np.asarray(kp2d, dtype=np.float32).reshape(-1, 33, 4),
                        kp3d=np.asarray(kp3d, dtype=np.float32).reshape(-1, 33, 4),
                        detected=np.asarray(detected, dtype=bool),
                        fps=np.float32(fps), width=np.int32(width), height=np.int32(height))
    os.replace(tmp, out_path)
    return video


if __name__ == "__main__":
    from concurrent.futures import ProcessPoolExecutor
    os.makedirs(OUT, exist_ok=True)
    clips = [r["video"] for r in json.load(open(os.path.join(D, "pushup_clips.json"), encoding="utf-8")) if r["exercise"] == EXERCISE]
    todo = [v for v in clips if not os.path.exists(os.path.join(OUT, v.replace(".mp4", ".npz")))]
    print("tổng", len(clips), "| còn lại", len(todo), flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(int(sys.argv[1]) if len(sys.argv) > 1 else 8) as pool:
        for n, _ in enumerate(pool.map(work, todo, chunksize=4), 1):
            if n % 100 == 0:
                print(n, "/", len(todo), round(time.time() - t0), "s", flush=True)
    print("xong", round(time.time() - t0), "s")

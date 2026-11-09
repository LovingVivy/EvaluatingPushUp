"""Ứng dụng web chấm điểm chống đẩy: kéo thả video vào trình duyệt để phân tích.

Cách chạy (trong thư mục dự án, dùng Python của .venv):
    .venv\\Scripts\\python app\\server.py                 # mở http://127.0.0.1:8765 trong trình duyệt
    .venv\\Scripts\\python app\\server.py --port 9000     # đổi cổng
    .venv\\Scripts\\python app\\server.py --no-browser    # không tự mở trình duyệt

Chỉ dùng thư viện chuẩn của Python cho phần máy chủ; phần phân tích nằm trong engine.py.
Video tải lên được lưu tạm và bị xoá khi tắt chương trình. Mọi thứ chạy trên máy này, không gửi đi đâu.
"""
import argparse
import atexit
import json
import mimetypes
import os
import shutil
import sys
import tempfile
import threading
import traceback
import urllib.parse
import uuid
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

for _stream in (sys.stdout, sys.stderr):              # in được tiếng Việt kể cả khi cửa sổ lệnh không dùng UTF-8
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import engine  # noqa: E402

STATIC = os.path.join(HERE, "static")
TMP = tempfile.mkdtemp(prefix="cham_diem_chong_day_")
atexit.register(shutil.rmtree, TMP, True)
MAX_UPLOAD = 2 * 1024 ** 3
VIEWS = ("auto", "front", "side", "diagonal")
RENDERS = {"export": True, "preview": False}          # loại video xuất -> có vẽ khung xương và kết luận không

JOBS, JOBS_LOCK = {}, threading.Lock()
WORK_LOCK = threading.Lock()                           # mỗi lúc chỉ chạy một việc nặng (MediaPipe, mô hình, ghi video)


class Job:
    def __init__(self, name, path, view):
        self.id, self.name, self.path, self.view = uuid.uuid4().hex[:12], name, path, view
        self.state, self.stage, self.progress = "queued", "Đang chờ", 0.0
        self.result = self.internal = self.error = None
        self.renders = {}

    def status(self):
        out = dict(id=self.id, name=self.name, view=self.view, state=self.state, stage=self.stage,
                   progress=round(self.progress, 3), error=self.error,
                   renders={k: {x: v[x] for x in ("state", "progress", "error")} for k, v in self.renders.items()})
        if self.state == "done":
            out["result"] = self.result
        return out


def run_analysis(job):
    def progress(stage, value):
        job.stage, job.progress = stage, value
    with WORK_LOCK:
        job.state = "running"
        try:
            job.result, job.internal = engine.analyze(job.path, job.view, progress)
            job.state = "done"
        except Exception as e:  # noqa: BLE001  (báo lỗi cho giao diện thay vì làm sập máy chủ)
            traceback.print_exc()
            job.state, job.error = "error", str(e) or e.__class__.__name__


def run_render(job, kind):
    r = job.renders[kind]
    with WORK_LOCK:
        r["state"] = "running"
        try:
            engine.render(job.path, job.result, job.internal, r["path"], annotate=RENDERS[kind], max_side=720,
                          progress=lambda value: r.__setitem__("progress", value))
            r["state"] = "done"
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            r["state"], r["error"] = "error", str(e) or e.__class__.__name__


class Handler(BaseHTTPRequestHandler):
    server_version = "ChamDiemChongDay/1.0"

    def log_message(self, fmt, *args):                 # chỉ in lỗi, không in mọi request
        if args and str(args[1]).startswith(("4", "5")) and str(args[1]) != "416":
            sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    # ---------------------------------------------------------------- tiện ích
    def send_json(self, obj, status=200):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, ctype=None, download_name=None):
        size = os.path.getsize(path)
        ctype = ctype or mimetypes.guess_type(path)[0] or "application/octet-stream"
        start, end, status = 0, size - 1, 200
        rng = self.headers.get("Range", "")
        if rng.startswith("bytes=") and size:
            first = rng[6:].split(",")[0].strip()
            a, _, b = first.partition("-")
            try:
                if a:
                    start, end = int(a), (int(b) if b else size - 1)
                else:
                    start, end = max(0, size - int(b)), size - 1
            except ValueError:
                start, end = 0, size - 1
            end = min(end, size - 1)
            if start > end:
                self.send_response(416)
                self.send_header("Content-Range", f"bytes */{size}")
                self.end_headers()
                return
            status = 206
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(end - start + 1))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        if download_name:
            ascii_name = download_name.encode("ascii", "ignore").decode() or "video.webm"
            self.send_header("Content-Disposition",
                             f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{urllib.parse.quote(download_name)}")
        self.end_headers()
        try:
            with open(path, "rb") as f:
                f.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = f.read(min(1 << 20, left))
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    left -= len(chunk)
        except (ConnectionResetError, BrokenPipeError, ConnectionAbortedError):
            pass                                           # trình duyệt đóng kết nối khi tua video: bình thường

    def job_from(self, parts):
        with JOBS_LOCK:
            return JOBS.get(parts[2]) if len(parts) > 2 else None

    # ---------------------------------------------------------------- GET
    def do_GET(self):
        url = urllib.parse.urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        if not parts or parts == ["index.html"]:
            return self.send_file(os.path.join(STATIC, "index.html"), "text/html; charset=utf-8")
        if parts[:2] == ["api", "job"]:
            job = self.job_from(parts)
            if job is None:
                return self.send_json({"error": "Không tìm thấy phiên phân tích này (máy chủ có thể đã khởi động lại)."}, 404)
            if len(parts) == 3:
                return self.send_json(job.status())
            if parts[3] == "video":
                return self.send_file(job.path)
            if parts[3] == "render" and len(parts) == 5 and parts[4] in RENDERS:
                r = job.renders.get(parts[4])
                if not r or r["state"] != "done":
                    return self.send_json({"error": "Video chưa được tạo xong."}, 409)
                name = os.path.splitext(job.name)[0] + "_cham_diem.webm" if parts[4] == "export" else None
                return self.send_file(r["path"], "video/webm", name)
        self.send_json({"error": "Không có trang này."}, 404)

    # ---------------------------------------------------------------- POST
    def do_POST(self):
        url = urllib.parse.urlparse(self.path)
        parts = [p for p in url.path.split("/") if p]
        if parts == ["api", "analyze"]:
            return self.start_analysis(urllib.parse.parse_qs(url.query))
        if parts[:2] == ["api", "job"] and len(parts) == 5 and parts[3] == "render" and parts[4] in RENDERS:
            job = self.job_from(parts)
            if job is None or job.state != "done":
                return self.send_json({"error": "Phiên phân tích chưa xong hoặc không tồn tại."}, 409)
            kind = parts[4]
            with JOBS_LOCK:
                r = job.renders.get(kind)
                if r is None or r["state"] == "error":
                    r = job.renders[kind] = dict(state="queued", progress=0.0, error=None,
                                                 path=os.path.join(TMP, f"{job.id}_{kind}.webm"))
                    threading.Thread(target=run_render, args=(job, kind), daemon=True).start()
            return self.send_json({k: r[k] for k in ("state", "progress", "error")})
        self.send_json({"error": "Không có trang này."}, 404)

    def start_analysis(self, query):
        view = query.get("view", ["auto"])[0]
        name = os.path.basename(query.get("name", ["video.mp4"])[0]) or "video.mp4"
        length = int(self.headers.get("Content-Length") or 0)
        if view not in VIEWS:
            return self.send_json({"error": "Góc quay không hợp lệ."}, 400)
        if length <= 0:
            return self.send_json({"error": "Không nhận được dữ liệu video."}, 400)
        if length > MAX_UPLOAD:
            return self.send_json({"error": "Video quá lớn (tối đa 2 GB)."}, 413)
        ext = os.path.splitext(name)[1].lower()
        if ext not in (".mp4", ".mov", ".m4v", ".avi", ".mkv", ".webm"):
            ext = ".mp4"
        path = os.path.join(TMP, uuid.uuid4().hex + ext)
        left = length
        with open(path, "wb") as f:
            while left > 0:
                chunk = self.rfile.read(min(1 << 20, left))
                if not chunk:
                    break
                f.write(chunk)
                left -= len(chunk)
        if left > 0:
            os.remove(path)
            return self.send_json({"error": "Tải video lên bị gián đoạn."}, 400)
        job = Job(name, path, view)
        with JOBS_LOCK:
            JOBS[job.id] = job
        threading.Thread(target=run_analysis, args=(job,), daemon=True).start()
        self.send_json({"job": job.id})


def main():
    ap = argparse.ArgumentParser(description="Ứng dụng web chấm điểm chống đẩy")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-browser", action="store_true", help="không tự mở trình duyệt")
    args = ap.parse_args()
    threading.Thread(target=engine.models, daemon=True).start()          # nạp sẵn mô hình trong lúc chờ
    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"Ứng dụng đang chạy tại {url}  (nhấn Ctrl+C để tắt)", flush=True)
    if not args.no_browser:
        threading.Timer(0.8, webbrowser.open, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()

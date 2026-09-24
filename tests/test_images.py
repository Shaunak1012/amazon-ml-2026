"""Image downloader against a local HTTP server: ok, resize, 404, corrupt, transient 503, resume."""
import io
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from PIL import Image

from src.images import download_images, safe_name


def _png(w=800, h=600) -> bytes:
    b = io.BytesIO()
    Image.new("RGB", (w, h), (200, 30, 30)).save(b, "PNG")
    return b.getvalue()


@pytest.fixture(scope="module")
def server():
    hits = {"flaky": 0}
    png = _png()

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):
            if self.path.startswith("/img"):
                body, code = png, 200
            elif self.path == "/flaky":
                hits["flaky"] += 1
                body, code = (png, 200) if hits["flaky"] > 1 else (b"busy", 503)
            elif self.path == "/corrupt":
                body, code = b"\x89PNG not really", 200
            else:
                body, code = b"nope", 404
            self.send_response(code)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", hits
    srv.shutdown()


def test_download_statuses_and_resume(server, tmp_path):
    base, hits = server
    ids = ["a1", "a2", "missing", "bad", "flaky", "nourl"]
    urls = [f"{base}/img1", f"{base}/img2", f"{base}/404", f"{base}/corrupt", f"{base}/flaky", ""]
    man = download_images(ids, urls, tmp_path, max_side=256, workers=4, retries=2, progress=False)
    st = dict(zip(man["id"], man["status"]))
    assert st == {"a1": "ok", "a2": "ok", "missing": "http_404", "bad": "bad_image", "flaky": "ok", "nourl": "no_url"}
    assert list(man["id"]) == ids  # same order as input, no rows dropped
    with Image.open(tmp_path / "a1.jpg") as im:
        assert max(im.size) == 256 and im.mode == "RGB"
    assert not list(tmp_path.glob("*.part"))  # no half-written files
    assert (tmp_path / "manifest.csv").exists()

    man2 = download_images(ids, urls, tmp_path, max_side=256, workers=4, retries=0, progress=False)
    assert set(man2.loc[man2["status"] == "cached", "id"]) == {"a1", "a2", "flaky"}


def test_safe_name():
    assert safe_name("B00XYZ_1") == "B00XYZ_1"
    assert len(safe_name("https://x.com/a b?.jpg")) == 40

import http.server
import json
import sys
import threading
import time

sys.path.insert(0, r"C:\i\gesture")

import gesture_app as ga

assert ga.APP_VERSION

captured = []


class Handler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        captured.append(json.loads(self.rfile.read(length).decode("utf-8")))
        self.send_response(200)
        self.end_headers()

    def log_message(self, *args):
        pass


server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target=server.serve_forever, daemon=True).start()
url = f"http://127.0.0.1:{server.server_address[1]}/report"

rep = ga.GestureReporter(url)
rep.min_interval = 0.0


def wait_sent(target, timeout=3.0):
    deadline = time.time() + timeout
    while rep.sent < target and time.time() < deadline:
        time.sleep(0.02)
    return rep.sent >= target


hands1 = [{"hand": "Right", "gesture": "Кулак", "x": 0.1234, "y": 0.5678}]
assert rep.update(hands1, 30.0) is True, "first update must send"
assert wait_sent(1), "first post must complete"
assert rep.update(hands1, 30.0) is False, "same state must be deduplicated"

hands2 = [{"hand": "Left", "gesture": "Открытая ладонь", "x": 0.5, "y": 0.5}]
assert rep.update(hands2, 29.0) is True, "changed state must send"
assert wait_sent(2), "second post must complete"
assert rep.update([], 28.0) is True, "empty hands must send clear event"
assert wait_sent(3), "third post must complete"

deadline = time.time() + 3
while len(captured) < 3 and time.time() < deadline:
    time.sleep(0.05)
assert len(captured) == 3, f"expected 3 posts, got {len(captured)}"

assert captured[0]["hands"][0]["side"] == "Right"
assert captured[0]["hands"][0]["gesture"] == "Кулак"
assert captured[0]["hands"][0]["x"] == 0.123
assert captured[0]["hands"][0]["y"] == 0.568
assert "time" in captured[0] and captured[0]["fps"] == 30.0
assert captured[2]["hands"] == []

rep.enabled = False
assert rep.update([{"hand": "Right", "gesture": "Кулак", "x": 0.1, "y": 0.1}], 30) is False

rep.set_url(url)
rep.enabled = True
rep._last_key = None
assert rep.update(hands1, 30.0) is True

rep2 = ga.GestureReporter("http://127.0.0.1:9/nope", timeout=0.5)
rep2.min_interval = 0.0
rep2.update(hands1, 30.0)
deadline = time.time() + 3
while rep2.failed == 0 and time.time() < deadline:
    time.sleep(0.05)
assert rep2.failed >= 1, "failed counter must grow on unreachable url"
assert rep2.last_error, "last_error must be set"

print("sent:", rep.sent, "captured:", len(captured))
print("ALL REPORTER TESTS PASSED")

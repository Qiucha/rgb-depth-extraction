# ConnectionResetError Errno 54 on snapshot-server request-line read

Research for [issue #4](https://github.com/Qiucha/rgb-depth-extraction/issues/4). No fix is proposed here.

## Answer

The stack is the server **accepting a TCP connection** from `192.168.8.143`, then blocking in `BaseHTTPRequestHandler.handle_one_request` while `rfile.readline` calls `socket.recv_into`. Darwin `ECONNRESET` (54) means the **peer aborted the TCP session with RST**, not a graceful FIN and not a Python socket timeout.

That traceback is produced only while the handler is waiting for an HTTP **request line** (first line of a new connection, or the next line on an HTTP/1.1 keep-alive socket). It is **not** produced when a client times out in the middle of a long `do_POST` (stereo extract): that path has already left `readline`.

Events that fit this log:

1. **Connect-then-abort** — `URLSession` opens TCP, then cancels the task before sending `POST …` (task cancel, app suspend, racing duplicate connection). Eleven distinct client ports in a tight ephemeral range match a burst of separate sockets, not one reused connection.
2. **Keep-alive teardown with RST** — after a completed request, this server leaves the socket open (`protocol_version = "HTTP/1.1"`). `handle()` loops back into `readline`. A later client abort (idle `timeoutInterval` default 60s, or CFNetwork closing a persistent connection) RST-s the waiter. Same stack; cannot tell first vs subsequent `readline` from the traceback alone.
3. **Link / NAT / device-sleep RST** attributed to that peer IP — indistinguishable from (1)/(2) without a packet capture.

Events that do **not** fit:

- Graceful close (FIN) → empty `raw_requestline`, silent return.
- Server-side read timeout → `TimeoutError`, logged as `"Request timed out"` (`StreamRequestHandler.timeout` is `None` here anyway).
- ATS denying cleartext **before** TCP → no `accept`, so no this stack. Issue maps that treat ATS as the explanation of **this** stderr dump are not supported by the log.
- `WebSocketStreamer`’s 10s `timeoutIntervalForRequest` → different port (8765), not this HTTP handler.

`SnapshotUploadClient` uses `URLSession.shared` and does not set `timeoutInterval`. Defaults are 60s idle (`URLRequest.timeoutInterval` / `timeoutIntervalForRequest`) and 7 days resource (`timeoutIntervalForResource`). A 60s idle timeout **during** processing would RST a socket that is already past the request line; that is a different stack.

---

## 1. What the log actually contains

Files (repo `.runtime/`, not rotated in-band):

| File | APFS birth / mtime | Content |
| --- | --- | --- |
| `.runtime/snapshot-server.error.log` | 12 Aug 2026 07:34:03 | 11 identical traces, 253 lines, no timestamps |
| `.runtime/snapshot-server.log` | 12 Aug 2026 07:34:03 | Server start banners + 27 successful `snapshot_20260811_*` runs (last `snapshot_20260811_152722`) |

There is **no timestamp on each exception**. The “12 Aug” date is the filesystem timestamp. Stdout capture names are **11 Aug**. `SnapshotRequestHandler.log_message` is a no-op, so there are no HTTP access lines to join with the RSTs. The 11 client ports are all `('192.168.8.143', N)` with `N` in 55922–56075 (monotonic, small gaps). That is eleven **different** TCP connections.

Representative traceback (all 11 match):

```
socketserver.py  process_request_thread
socketserver.py  finish_request / BaseRequestHandler.__init__
http/server.py   handle → handle_one_request
http/server.py   self.raw_requestline = self.rfile.readline(65537)
socket.py        readinto → recv_into
ConnectionResetError: [Errno 54] Connection reset by peer
```

Interpreter in the stack: Homebrew CPython **3.14.6**. Line numbers match CPython 3.14 `http.server` / `socketserver` / `socket.py` as cited below.

---

## 2. Server: why this exception is printed

Sources: CPython 3.14.6 [`Lib/http/server.py`](https://github.com/python/cpython/blob/v3.14.6/Lib/http/server.py), [`Lib/socketserver.py`](https://github.com/python/cpython/blob/v3.14.6/Lib/socketserver.py), [`Lib/socket.py`](https://github.com/python/cpython/blob/v3.14.6/Lib/socket.py); local copies under `/opt/homebrew/Cellar/python@3.14/3.14.7/...` (same control flow; log cites 3.14.6). App: `src/realworld/ios_bridge/snapshot_server.py`.

### Accept and thread

`ThreadedHTTPServer` is `ThreadingMixIn` + `HTTPServer`. `ThreadingMixIn.process_request_thread` runs `finish_request` and on **any** `Exception` calls `handle_error`, which prints `Exception occurred during processing of request from <client_address>` plus `traceback.print_exc()` to **stderr** ([`socketserver.py` around `process_request_thread` / `handle_error`](https://github.com/python/cpython/blob/v3.14.6/Lib/socketserver.py)).

So the peer address in the banner is the TCP 4-tuple of the **accepted** socket. Handshake completed.

### Where `readline` sits

[`BaseHTTPRequestHandler.handle`](https://github.com/python/cpython/blob/v3.14.6/Lib/http/server.py):

```python
self.close_connection = True
self.handle_one_request()
while not self.close_connection:
    self.handle_one_request()
```

[`handle_one_request`](https://github.com/python/cpython/blob/v3.14.6/Lib/http/server.py) (v3.14.6):

```python
try:
    self.raw_requestline = self.rfile.readline(65537)
    ...
    if not self.raw_requestline:
        self.close_connection = True
        return
    if not self.parse_request():
        return
    method = getattr(self, 'do_' + self.command)
    method()
    self.wfile.flush()
except TimeoutError as e:
    self.log_error("Request timed out: %r", e)
    self.close_connection = True
    return
```

`ConnectionResetError` is **not** caught. It propagates to `process_request_thread` → stderr dump. That is this log.

`rfile` is `connection.makefile('rb')` ([`StreamRequestHandler.setup`](https://github.com/python/cpython/blob/v3.14.6/Lib/socketserver.py)). `timeout` defaults to `None` (no socket timeout). `makefile` read uses [`socket.socket.readinto`](https://github.com/python/cpython/blob/v3.14.6/Lib/socket.py): `recv_into`; a 0-byte return means the peer **shutdown** (FIN); `ECONNRESET` raises.

On this Darwin host, `errno.ECONNRESET == 54`. POSIX meaning: the connection was reset by the peer (RST), not a local timeout.

### HTTP/1.1 keep-alive on this app

`SnapshotRequestHandler.protocol_version = "HTTP/1.1"`. [`parse_request`](https://github.com/python/cpython/blob/v3.14.6/Lib/http/server.py) sets `close_connection = False` when both sides are HTTP/1.1, unless `Connection: close`. `_send_json_response` does not send `Content-Length` or `Connection: close`. After `do_POST` returns, `handle()` waits again on `readline`. A client RST at that point is **the same stack** as a RST before the first byte.

### What a processing-time abort would look like

`do_POST` reads the body with `self.rfile.read(content_length)` only **after** the request line and headers are parsed. A client abort during `run_custom_iphone_depth` would typically raise on `wfile.write` in `_send_json_response`, caught by `do_POST`’s `except Exception`. That is **not** the recorded stack.

---

## 3. Client: `URLSession` events that can RST

Sources: [`SnapshotUploadClient.swift`](../../ios/DualCamStereoCapture/DualCamStereoCapture2/DualCamStereoCapture2/SnapshotUploadClient.swift); Apple [URLSession.shared](https://developer.apple.com/documentation/foundation/urlsession/shared), [URLRequest.timeoutInterval](https://developer.apple.com/documentation/foundation/urlrequest/timeoutinterval), [timeoutIntervalForRequest](https://developer.apple.com/documentation/foundation/urlsessionconfiguration/timeoutintervalforrequest), [timeoutIntervalForResource](https://developer.apple.com/documentation/foundation/urlsessionconfiguration/timeoutintervalforresource), [NSAppTransportSecurity](https://developer.apple.com/documentation/bundleresources/information-property-list/nsapptransportsecurity), [NSAllowsLocalNetworking](https://developer.apple.com/documentation/bundleresources/information-property-list/nsapptransportsecurity/nsallowslocalnetworking).

### What the app actually does

`SnapshotUploadClient.upload` builds `URLRequest(url:)` (default timeout), sets `POST` + multipart `httpBody`, then `URLSession.shared.dataTask(with: request).resume()`. It does not set `timeoutInterval`, does not use a custom `URLSessionConfiguration`, and does not call `cancel()` itself.

`URLSession.shared` has **no** configuration object ([Apple: shared session limitations](https://developer.apple.com/documentation/foundation/urlsession/shared)). Timeouts are the defaults:

| Knob | Default | Meaning |
| --- | --- | --- |
| `URLRequest.timeoutInterval` | 60s | Idle during the connection attempt / request |
| `URLSessionConfiguration.timeoutIntervalForRequest` | 60s | Idle between additional data; timer resets when data arrives |
| `timeoutIntervalForResource` | 7 days | Whole transfer |

`WebSocketStreamer` sets `timeoutIntervalForRequest = 10` on a **different** `URLSession` for the WebSocket path. It is not this HTTP upload.

### Client-side events vs this stack

| Client event | TCP seen by `http.server` | Fits this log? |
| --- | --- | --- |
| Task **cancel** / process freeze / app kill after SYN-ACK, before request line | `accept` then `recv` → `ECONNRESET` (abort) or 0-byte FIN | **RST variant fits** |
| Duplicate / raced TCP connect cancelled when another path wins | Same: extra accepted sockets, RST before HTTP | **Fits** 11 distinct ports in a burst |
| Idle **60s timeout while waiting for a response** after request line + body already sent | RST (or FIN) while server is in `do_POST` or `wfile.write`, **not** first `readline` | **Does not fit** this stack |
| Idle 60s timeout because response lacked `Content-Length` / close, after `do_POST` finished | Server already in keep-alive `readline` | **Fits** keep-alive case |
| Graceful `close` after a finished exchange | `readline` returns `b''`; no traceback | **Does not fit** |
| ATS block with **no** TCP | No `accept` | **Does not fit** |

Apple documents that ATS **blocks** loads that are not HTTPS ([NSAppTransportSecurity](https://developer.apple.com/documentation/bundleresources/information-property-list/nsapptransportsecurity)). On **iOS 17+**, ATS does not allow **IP-literal** connections by default; `NSAllowsLocalNetworking` re-enables unqualified / `.local` / IP access ([NSAllowsLocalNetworking](https://developer.apple.com/documentation/bundleresources/information-property-list/nsapptransportsecurity/nsallowslocalnetworking)). This target’s `project.pbxproj` has camera/local-network usage strings and **no** `NSAppTransportSecurity` / `NSAllowsLocalNetworking` keys. That is a plausible **client-side** `-1022` (`NSURLErrorAppTransportSecurityRequiresSecureConnection`) for `http://<lan-ip>:8766` on iOS 17+. It is **not** an explanation of eleven accepted-then-RST sockets: those sockets existed.

---

## 4. How the pieces line up

Most consistent with **this** error log:

- Eleven short-lived TCP connections from the iPhone LAN address.
- Each died with **RST while the server was blocked on the HTTP request line**.
- Therefore: **pre-request abort** (cancel / race / sleep) and/or **post-response keep-alive abort** (including a 60s idle timeout after a response that did not delimit the body).

Not consistent:

- “URLSession 60s timeout during depth extraction.”
- “ATS blocked HTTP, and that is what stderr printed.”
- “Python `http.server` timed out.”

A packet capture (RST vs FIN, whether any request bytes were sent) would separate (1) from (2). This log cannot.

## Sources

- CPython v3.14.6: [http/server.py](https://github.com/python/cpython/blob/v3.14.6/Lib/http/server.py), [socketserver.py](https://github.com/python/cpython/blob/v3.14.6/Lib/socketserver.py), [socket.py](https://github.com/python/cpython/blob/v3.14.6/Lib/socket.py)
- Apple: [URLSession.shared](https://developer.apple.com/documentation/foundation/urlsession/shared), [URLRequest.timeoutInterval](https://developer.apple.com/documentation/foundation/urlrequest/timeoutinterval), [timeoutIntervalForRequest](https://developer.apple.com/documentation/foundation/urlsessionconfiguration/timeoutintervalforrequest), [timeoutIntervalForResource](https://developer.apple.com/documentation/foundation/urlsessionconfiguration/timeoutintervalforresource), [NSAppTransportSecurity](https://developer.apple.com/documentation/bundleresources/information-property-list/nsapptransportsecurity), [NSAllowsLocalNetworking](https://developer.apple.com/documentation/bundleresources/information-property-list/nsapptransportsecurity/nsallowslocalnetworking)
- Repo: `.runtime/snapshot-server.error.log`, `.runtime/snapshot-server.log`, `src/realworld/ios_bridge/snapshot_server.py`, `ios/.../SnapshotUploadClient.swift`, `ios/.../project.pbxproj`

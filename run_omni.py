#!/usr/bin/env python
"""Gemini Omni 1.2 Flash video runner (stdlib only).

Reproduces the "multi-round motion transfer" demo from
x.com/FinanceYF5/status/2118537565884252564: round 1 moves the motion of a
reference clip onto a character still, round 2+ edit that result in place
through `previous_interaction_id`, round N extends it.

    python run_omni.py generate --prompt-file P [--image IMG]... [--video VID]...
                       [--duration 10] [--resolution 720p] [--aspect 16:9]
                       [--task reference_to_video] [--out-dir out] [--execute]
    python run_omni.py continue --previous <interaction id or out/<id>.json>
                       --prompt-file P [--image IMG]... [--execute]
    python run_omni.py resume out/<id>.json          # re-poll / re-download, no new spend

Without --execute every command prints the plan and the price and sends nothing.
The key is read from GOOGLE_API_KEY / GEMINI_API_KEY, then
%APPDATA%\\agent-sprite-forge\\config.json.
It is never printed.
"""
import argparse, base64, json, mimetypes, os, re, sys, time, urllib.error, urllib.request

API = "https://generativelanguage.googleapis.com"
MODEL = "gemini-omni-1.2-flash"
USD_PER_S = {"360p": 0.023, "720p": 0.07, "1080p": 0.104, "4k": 0.209}
HERE = os.path.dirname(os.path.abspath(__file__))


# ------------------------------------------------------------------ key
def load_key():
    for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
        v = os.environ.get(name, "").strip()
        if v:
            return v, "env:" + name
    cfg = os.path.join(os.environ.get("APPDATA", ""), "agent-sprite-forge", "config.json")
    if os.path.isfile(cfg):
        try:
            d = json.load(open(cfg, encoding="utf-8"))
            for name in ("GOOGLE_API_KEY", "GEMINI_API_KEY"):
                if d.get(name):
                    return d[name].strip(), "config.json:" + name
        except Exception:
            pass
    return None, None


# ------------------------------------------------------------------ http
def call(method, path, key, body=None, headers=None, raw=None, timeout=600):
    url = path if path.startswith("http") else API + path
    h = {"x-goog-api-key": key}
    if body is not None:
        raw = json.dumps(body).encode("utf-8")
        h["Content-Type"] = "application/json"
    if headers:
        h.update(headers)
    req = urllib.request.Request(url, data=raw, method=method, headers=h)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = r.read()
            return r.status, dict(r.headers), data
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), e.read()


def jload(data):
    try:
        return json.loads(data.decode("utf-8"))
    except Exception:
        return {"_raw": data[:2000].decode("utf-8", "replace")}


def die(msg, code=1):
    print("error: " + msg, file=sys.stderr)
    sys.exit(code)


# ------------------------------------------------------------------ files api
def upload_file(path, key):
    """Resumable upload to the File API; returns the file resource (uri, name)."""
    size = os.path.getsize(path)
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    st, hd, data = call("POST", "/upload/v1beta/files", key,
                        body={"file": {"display_name": os.path.basename(path)}},
                        headers={"X-Goog-Upload-Protocol": "resumable",
                                 "X-Goog-Upload-Command": "start",
                                 "X-Goog-Upload-Header-Content-Length": str(size),
                                 "X-Goog-Upload-Header-Content-Type": mime})
    if st != 200:
        die("upload start %s: %s" % (st, jload(data)))
    up = {k.lower(): v for k, v in hd.items()}.get("x-goog-upload-url")
    if not up:
        die("upload start returned no X-Goog-Upload-URL")
    with open(path, "rb") as f:
        blob = f.read()
    st, hd, data = call("POST", up, key, raw=blob,
                        headers={"Content-Length": str(size),
                                 "X-Goog-Upload-Offset": "0",
                                 "X-Goog-Upload-Command": "upload, finalize"})
    if st != 200:
        die("upload finalize %s: %s" % (st, jload(data)))
    info = jload(data).get("file", {})
    name = info.get("name")
    for _ in range(120):
        if info.get("state", "ACTIVE") == "ACTIVE":
            return info
        time.sleep(2)
        st, _, data = call("GET", "/v1beta/" + name, key)
        info = jload(data)
    die("file %s never became ACTIVE" % name)


def inline_part(kind, path):
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    b = open(path, "rb").read()
    return {"type": kind, "mime_type": mime, "data": base64.b64encode(b).decode("ascii")}


# ------------------------------------------------------------------ result handling
def find_videos(obj, acc=None):
    acc = [] if acc is None else acc
    if isinstance(obj, dict):
        if obj.get("type") == "video" and ("data" in obj or "uri" in obj):
            acc.append(obj)
        for v in obj.values():
            find_videos(v, acc)
    elif isinstance(obj, list):
        for v in obj:
            find_videos(v, acc)
    return acc


def download_uri(uri, key, dest):
    name = re.search(r"(files/[^:?/]+)", uri)
    if name:
        for _ in range(300):
            st, _, data = call("GET", "/v1beta/" + name.group(1), key)
            info = jload(data)
            if info.get("state", "ACTIVE") == "ACTIVE":
                break
            time.sleep(3)
    st, hd, data = call("GET", uri if ":download" in uri else uri + ":download?alt=media", key)
    if st != 200:
        die("download %s: %s" % (st, data[:300]))
    open(dest, "wb").write(data)
    return len(data)


def save_result(resp, out_dir, key, stem):
    os.makedirs(out_dir, exist_ok=True)
    iid = resp.get("id") or stem
    meta = os.path.join(out_dir, "%s.json" % iid.replace("/", "_"))
    json.dump(resp, open(meta, "w", encoding="utf-8"), indent=1)
    vids = find_videos(resp)
    outs = []
    for i, v in enumerate(vids):
        dest = os.path.join(out_dir, "%s%s.mp4" % (iid.replace("/", "_"), "" if i == 0 else "_%d" % i))
        if "data" in v:
            open(dest, "wb").write(base64.b64decode(v["data"]))
            n = os.path.getsize(dest)
            v["data"] = "<written to %s>" % os.path.basename(dest)
        else:
            n = download_uri(v["uri"], key, dest)
        outs.append((dest, n))
    json.dump(resp, open(meta, "w", encoding="utf-8"), indent=1)   # without the base64 blob
    return iid, meta, outs


def poll_interaction(resp, key):
    """Some interactions return before the video exists; poll by id until a video appears
    or the status says it is done."""
    iid = resp.get("id")
    for _ in range(200):
        if find_videos(resp) or resp.get("status") in ("completed", "failed", "cancelled"):
            return resp
        if not iid:
            return resp
        time.sleep(5)
        st, _, data = call("GET", "/v1beta/interactions/" + iid, key)
        resp = jload(data)
    return resp


# ------------------------------------------------------------------ commands
def build_input(args, key, execute):
    parts = []
    plan = []
    for p in args.image or []:
        if not os.path.isfile(p):
            die("image not found: " + p)
        parts.append(inline_part("image", p) if execute else {"type": "image", "file": p})
        plan.append("image  %s (%d bytes)" % (p, os.path.getsize(p)))
    for p in args.video or []:
        if not os.path.isfile(p):
            die("video not found: " + p)
        if execute:
            info = upload_file(p, key)
            parts.append({"type": "video", "uri": info["uri"]})
        else:
            parts.append({"type": "video", "file": p})
        plan.append("video  %s (%d bytes, uploaded through the File API)" % (p, os.path.getsize(p)))
    text = open(args.prompt_file, encoding="utf-8").read().strip()
    parts.append({"type": "text", "text": text})
    plan.append("text   %s (%d chars)" % (args.prompt_file, len(text)))
    return parts, plan, text


def run_request(body, args, key, src):
    secs = int(re.sub(r"\D", "", body.get("response_format", {}).get("duration", "10s")) or 10)
    res = body.get("response_format", {}).get("resolution", "720p")
    est = USD_PER_S.get(res, 0.07) * secs
    print("model       %s" % MODEL)
    print("key         %s" % (src or "NONE"))
    print("duration    %ds   resolution %s   aspect %s" % (secs, res, body.get("response_format", {}).get("aspect_ratio", "16:9")))
    print("estimate    USD %.2f (output video only; inputs are priced as tokens and are small)" % est)
    if not args.execute:
        print("plan (not sent):")
        print(json.dumps(body, indent=1, ensure_ascii=False)[:4000])
        print("add --execute to send it.")
        return
    if not key:
        die("no API key (GOOGLE_API_KEY / GEMINI_API_KEY / config.json)")
    t0 = time.time()
    st, _, data = call("POST", "/v1beta/interactions", key, body=body)
    resp = jload(data)
    if isinstance(resp, list) and resp:
        resp = resp[0]
    if st != 200:
        err = resp.get("error", resp)
        if st == 402:
            die("402 billing: %s" % err.get("message", err))
        die("%s: %s" % (st, json.dumps(err)[:1500]))
    resp = poll_interaction(resp, key)
    iid, meta, outs = save_result(resp, args.out_dir, key, "omni-%d" % int(t0))
    print("interaction %s   (%.0fs)" % (iid, time.time() - t0))
    print("meta        %s" % meta)
    for dest, n in outs:
        print("video       %s (%d bytes)" % (dest, n))
    if not outs:
        print("no video in the response; status=%s; see the meta file" % resp.get("status"))
        sys.exit(2)


def cmd_generate(args):
    key, src = load_key()
    parts, plan, _ = build_input(args, key, args.execute)
    body = {"model": MODEL, "input": parts,
            "response_format": {"type": "video", "aspect_ratio": args.aspect,
                                "resolution": args.resolution, "duration": "%ds" % args.duration,
                                "delivery": "uri"}}
    if args.task:
        body["generation_config"] = {"video_config": {"task": args.task}}
    for p in plan:
        print("input       " + p)
    run_request(body, args, key, src)


def cmd_continue(args):
    key, src = load_key()
    prev = args.previous
    if os.path.isfile(prev):
        prev = json.load(open(prev, encoding="utf-8")).get("id") or die("no id in " + prev)
    parts, plan, text = build_input(args, key, args.execute)
    body = {"model": MODEL, "previous_interaction_id": prev,
            "input": parts if len(parts) > 1 else text}
    rf = {"type": "video", "delivery": "uri"}
    if args.duration:
        rf["duration"] = "%ds" % args.duration
    if args.resolution:
        rf["resolution"] = args.resolution
    body["response_format"] = rf
    if args.task:
        body["generation_config"] = {"video_config": {"task": args.task}}
    print("previous    %s" % prev)
    for p in plan:
        print("input       " + p)
    run_request(body, args, key, src)


def cmd_resume(args):
    key, src = load_key()
    meta = json.load(open(args.meta, encoding="utf-8"))
    iid = meta.get("id") or die("no id in " + args.meta)
    st, _, data = call("GET", "/v1beta/interactions/" + iid, key)
    resp = jload(data)
    if st != 200:
        die("%s: %s" % (st, json.dumps(resp)[:800]))
    resp = poll_interaction(resp, key)
    iid, meta_path, outs = save_result(resp, os.path.dirname(args.meta) or ".", key, iid)
    print("meta        %s" % meta_path)
    for dest, n in outs:
        print("video       %s (%d bytes)" % (dest, n))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, duration_default):
        p.add_argument("--prompt-file", required=True)
        p.add_argument("--image", action="append", help="character / frame reference (inline, repeatable, max 6)")
        p.add_argument("--video", action="append", help="motion / edit reference clip (File API, repeatable)")
        p.add_argument("--duration", type=int, default=duration_default, help="3..15 seconds")
        p.add_argument("--resolution", default=None if duration_default is None else "720p",
                       choices=[None, "360p", "720p", "1080p", "4k"])
        p.add_argument("--task", default=None,
                       choices=[None, "text_to_video", "image_to_video", "reference_to_video", "edit", "extend"])
        p.add_argument("--out-dir", default=os.path.join(HERE, "out"))
        p.add_argument("--execute", action="store_true", help="actually send the paid request")

    g = sub.add_parser("generate", help="round 1: a new clip from text / images / a motion reference video")
    common(g, 10)
    g.add_argument("--aspect", default="16:9", choices=["16:9", "9:16"])
    g.set_defaults(fn=cmd_generate)

    c = sub.add_parser("continue", help="round 2+: edit or extend the previous result in place")
    common(c, None)
    c.add_argument("--previous", required=True, help="interaction id, or the out/<id>.json of the previous round")
    c.set_defaults(fn=cmd_continue)

    r = sub.add_parser("resume", help="re-poll and re-download a finished interaction; no new spend")
    r.add_argument("meta")
    r.set_defaults(fn=cmd_resume)

    args = ap.parse_args()
    if getattr(args, "duration", None) is not None and not 3 <= args.duration <= 15:
        die("--duration must be 3..15")
    args.fn(args)


if __name__ == "__main__":
    main()

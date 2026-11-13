#!/usr/bin/env python
"""Zero-key multi-round motion transfer: pose -> 2D cut-out puppet.

No model API. The motion comes from a real clip (MediaPipe Pose on the CPU) or from a
built-in motion script; the puppet is drawn with Pillow from the kitten still's head crop
plus simple limbs, and ffmpeg packs the frames. "Multi-round" is the same pose track
rendered again with a different performer and place (round 2), then extended with a
second motion (round 3) - the structure of the Gemini Omni demo, with the generation
replaced by code.

    python puppet.py extract --video refs/motion.mp4 --out out/pose.json
    python puppet.py synth   --motion wave_step_bow --duration 10 --out out/pose.json
    python puppet.py render  --pose out/pose.json --skin kitten --scene library --out out/round1.mp4
    python puppet.py extend  --pose out/pose.json --motion nod --duration 5 --out out/pose_ext.json
    python puppet.py demo    [--video refs/motion.mp4] --out-dir out      # all three rounds + joined mp4
"""
import argparse, json, math, os, shutil, subprocess, sys, tempfile

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1280, 720, 24
JOINTS = ["nose", "l_sho", "r_sho", "l_elb", "r_elb", "l_wri", "r_wri",
          "l_hip", "r_hip", "l_kne", "r_kne", "l_ank", "r_ank"]
MP_INDEX = {"nose": 0, "l_sho": 11, "r_sho": 12, "l_elb": 13, "r_elb": 14, "l_wri": 15, "r_wri": 16,
            "l_hip": 23, "r_hip": 24, "l_kne": 25, "r_kne": 26, "l_ank": 27, "r_ank": 28}

REST = {"nose": (0.50, 0.30), "l_sho": (0.56, 0.42), "r_sho": (0.44, 0.42),
        "l_elb": (0.60, 0.52), "r_elb": (0.40, 0.52), "l_wri": (0.59, 0.62), "r_wri": (0.41, 0.62),
        "l_hip": (0.54, 0.62), "r_hip": (0.46, 0.62), "l_kne": (0.54, 0.76), "r_kne": (0.46, 0.76),
        "l_ank": (0.54, 0.90), "r_ank": (0.46, 0.90)}
# viewer's right = performer's left ("l_*"); the wave uses the arm on the viewer's right.


# ---------------------------------------------------------------- motion scripts
def smooth(a):
    a = max(0.0, min(1.0, a))
    return a * a * (3 - 2 * a)


def lerp_pose(p, q, a):
    return {j: (p[j][0] + (q[j][0] - p[j][0]) * a, p[j][1] + (q[j][1] - p[j][1]) * a) for j in JOINTS}


def shift(p, dx=0.0, dy=0.0, only=None):
    return {j: ((x + dx, y + dy) if (only is None or j in only) else (x, y)) for j, (x, y) in p.items()}


def pose_wave_step_bow(t, dur):
    """idle (0-2 s) -> wave (2-5) -> two steps right (5-7) -> bow (7-9) -> return."""
    p = dict(REST)
    bob = 0.004 * math.sin(t * 2 * math.pi * 0.5)
    p = shift(p, dy=bob)
    # wave with the left arm (viewer's right)
    a = smooth((t - 2.0) / 0.6) * (1 - smooth((t - 4.6) / 0.5))
    if a > 0:
        up = dict(p); up["l_elb"] = (0.66, 0.40); up["l_wri"] = (0.70, 0.28)
        p = lerp_pose(p, up, a)
        swing = 0.05 * math.sin((t - 2.6) * 2 * math.pi * 1.8) * a
        p["l_wri"] = (p["l_wri"][0] + swing, p["l_wri"][1] + abs(swing) * 0.3)
    # steps: body drifts right, knees lift alternately
    s = smooth((t - 5.0) / 2.0)
    p = shift(p, dx=0.10 * s)
    if 5.0 <= t <= 7.0:
        ph = (t - 5.0) * 2 * math.pi * 1.0
        lift_l = max(0.0, math.sin(ph)) * 0.05
        lift_r = max(0.0, -math.sin(ph)) * 0.05
        p["l_kne"] = (p["l_kne"][0] + lift_l * 0.6, p["l_kne"][1] - lift_l)
        p["l_ank"] = (p["l_ank"][0] + lift_l * 0.8, p["l_ank"][1] - lift_l * 1.2)
        p["r_kne"] = (p["r_kne"][0] + lift_r * 0.6, p["r_kne"][1] - lift_r)
        p["r_ank"] = (p["r_ank"][0] + lift_r * 0.8, p["r_ank"][1] - lift_r * 1.2)
    # bow
    b = smooth((t - 7.0) / 0.7) * (1 - smooth((t - 8.6) / 0.6))
    if b > 0:
        p = shift(p, dy=0.08 * b, only=["nose"])
        p = shift(p, dy=0.05 * b, dx=0.0, only=["l_sho", "r_sho", "l_elb", "r_elb", "l_wri", "r_wri"])
    return p


def pose_nod(t, dur):
    p = dict(REST)
    p = shift(p, dy=0.004 * math.sin(t * 2 * math.pi * 0.5))
    if 0.8 <= t <= 2.4:
        n = math.sin((t - 0.8) / 1.6 * math.pi)
        p = shift(p, dy=0.04 * n, only=["nose"])
    if t > 3.0:  # paws together at the end
        a = smooth((t - 3.0) / 0.8)
        tgt = dict(p); tgt["l_wri"] = (0.52, 0.56); tgt["r_wri"] = (0.48, 0.56)
        tgt["l_elb"] = (0.60, 0.55); tgt["r_elb"] = (0.40, 0.55)
        p = lerp_pose(p, tgt, a)
    return p


MOTIONS = {"wave_step_bow": pose_wave_step_bow, "nod": pose_nod}


def synth(motion, duration, fps=FPS):
    fn = MOTIONS[motion]
    n = int(round(duration * fps))
    return {"fps": fps, "source": "synth:" + motion,
            "frames": [{j: list(fn(i / fps, duration)[j]) for j in JOINTS} for i in range(n)]}


# ---------------------------------------------------------------- extraction (MediaPipe)
def extract(video, fps=FPS, max_seconds=15.0):
    import cv2  # noqa  (opencv-python comes with mediapipe)
    import mediapipe as mp
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        sys.exit("error: cannot open " + video)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames, last = [], None
    step = src_fps / fps
    next_t, idx = 0.0, 0
    with mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1,
                                min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while True:
            ok, img = cap.read()
            if not ok or idx / src_fps > max_seconds:
                break
            if idx >= next_t:
                next_t += step
                res = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
                if res.pose_landmarks:
                    lm = res.pose_landmarks.landmark
                    last = {j: [lm[MP_INDEX[j]].x, lm[MP_INDEX[j]].y] for j in JOINTS}
                frames.append(dict(last) if last else {j: list(REST[j]) for j in JOINTS})
            idx += 1
    cap.release()
    if not frames:
        sys.exit("error: no frames read from " + video)
    # normalise: centre the hips at x=0.5 on the first frame, keep the performer's scale
    hx = (frames[0]["l_hip"][0] + frames[0]["r_hip"][0]) / 2
    dx = 0.5 - hx
    for f in frames:
        for j in JOINTS:
            f[j][0] += dx
    # light temporal smoothing
    for i in range(1, len(frames) - 1):
        for j in JOINTS:
            for k in (0, 1):
                frames[i][j][k] = (frames[i - 1][j][k] + 2 * frames[i][j][k] + frames[i + 1][j][k]) / 4
    return {"fps": fps, "source": "video:" + os.path.basename(video), "frames": frames}


def extend(pose, motion, duration, blend=0.5):
    """Append a synthesized motion, blending from the track's last frame over `blend` seconds."""
    fps = pose["fps"]
    tail = synth(motion, duration, fps)["frames"]
    last = pose["frames"][-1]
    nb = int(blend * fps)
    for i in range(min(nb, len(tail))):
        a = smooth((i + 1) / nb)
        tail[i] = {j: [last[j][0] + (tail[i][j][0] - last[j][0]) * a,
                       last[j][1] + (tail[i][j][1] - last[j][1]) * a] for j in JOINTS}
    return {"fps": fps, "source": pose["source"] + " + synth:" + motion, "frames": pose["frames"] + tail}


# ---------------------------------------------------------------- drawing
SKINS = {
    "kitten": dict(fur=(252, 236, 214), fur_dark=(236, 196, 150), dress=(243, 178, 196), dress_dark=(224, 146, 170),
                   scarf=(250, 214, 222), paw=(255, 248, 240), head="refs/kitten_head.png", tint=None, glasses=False),
    "tabby": dict(fur=(176, 178, 188), fur_dark=(120, 122, 136), dress=(78, 118, 190), dress_dark=(54, 88, 150),
                  scarf=(230, 230, 236), paw=(236, 236, 240), head="refs/kitten_head.png", tint=(150, 152, 165), glasses=True),
}


def head_image(skin):
    path = os.path.join(HERE, skin["head"])
    im = Image.open(path).convert("RGBA")
    if skin["tint"]:
        g = ImageOps.grayscale(im)
        col = ImageOps.colorize(g, (40, 40, 50), (235, 235, 240), mid=skin["tint"])
        col.putalpha(im.getchannel("A"))
        im = col
    if skin["glasses"]:
        d = ImageDraw.Draw(im)
        w, h = im.size
        for cx in (w * 0.36, w * 0.64):
            d.ellipse((cx - w * 0.13, h * 0.42, cx + w * 0.13, h * 0.68), outline=(40, 40, 50, 255), width=max(2, w // 60))
        d.line((w * 0.49, h * 0.55, w * 0.51, h * 0.55), fill=(40, 40, 50, 255), width=max(2, w // 60))
    return im


def scene_library(canvas, t):
    d = ImageDraw.Draw(canvas)
    for y in range(H):
        a = y / H
        d.line((0, y, W, y), fill=(int(214 - 60 * a), int(178 - 70 * a), int(132 - 70 * a)))
    d.rectangle((0, H * 0.78, W, H), fill=(150, 108, 72))
    for i in range(0, W, 90):
        d.line((i, H * 0.78, i - 60, H), fill=(128, 90, 58), width=2)
    for x0 in (60, 300, 940):
        d.rectangle((x0, 60, x0 + 200, H * 0.78), fill=(88, 56, 34))
        for s in range(5):
            y = 80 + s * 95
            d.rectangle((x0 + 8, y, x0 + 192, y + 6), fill=(60, 38, 22))
            bx = x0 + 12
            k = 0
            while bx < x0 + 185:
                bw = 14 + ((s * 7 + k * 5 + x0 // 7) % 12)
                col = [(170, 60, 60), (60, 90, 150), (190, 150, 60), (70, 130, 90), (120, 80, 140)][(k + s + x0 // 60) % 5]
                d.rectangle((bx, y + 8, bx + bw, y + 86), fill=col)
                bx += bw + 3
                k += 1
    d.rectangle((560, 70, 860, 420), fill=(236, 240, 246))
    d.rectangle((560, 70, 860, 420), outline=(110, 80, 50), width=10)
    d.line((710, 70, 710, 420), fill=(110, 80, 50), width=8)
    d.line((560, 245, 860, 245), fill=(110, 80, 50), width=8)
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).polygon([(560, 420), (860, 420), (1000, H * 0.78), (420, H * 0.78)], fill=(255, 240, 200, 60))
    canvas.alpha_composite(glow.filter(ImageFilter.GaussianBlur(18)))


def scene_rooftop(canvas, t):
    d = ImageDraw.Draw(canvas)
    for y in range(H):
        a = y / H
        d.line((0, y, W, y), fill=(int(40 + 215 * a * a), int(50 + 110 * a), int(110 + 20 * (1 - a))))
    d.ellipse((880, 120, 960, 200), fill=(255, 230, 190))
    hx = 0
    k = 0
    while hx < W:
        bw = 40 + (k * 37) % 70
        bh = 120 + (k * 53) % 200
        d.rectangle((hx, H * 0.70 - bh, hx + bw, H * 0.72), fill=(30, 28, 50))
        for wy in range(int(H * 0.70 - bh) + 10, int(H * 0.70), 22):
            for wx in range(hx + 6, hx + bw - 8, 14):
                if (wx * 7 + wy * 3 + k) % 5 < 2:
                    d.rectangle((wx, wy, wx + 6, wy + 10), fill=(255, 220, 140))
        hx += bw + 8
        k += 1
    d.rectangle((0, H * 0.72, W, H), fill=(92, 70, 56))
    for i in range(0, W, 70):
        d.line((i, H * 0.72, i, H), fill=(76, 56, 44), width=3)
    d.line((0, H * 0.60, W, H * 0.60), fill=(60, 60, 70), width=6)
    for x in range(0, W, 48):
        d.line((x, H * 0.60, x, H * 0.72), fill=(60, 60, 70), width=4)
    for i in range(0, W, 60):
        y = H * 0.56 + 10 * math.sin(i / 60 * 1.3)
        blink = 0.75 + 0.25 * math.sin(t * 3 + i)
        d.ellipse((i - 5, y - 5, i + 5, y + 5), fill=(int(255 * blink), int(225 * blink), int(140 * blink)))


SCENES = {"library": scene_library, "rooftop": scene_rooftop}


def px(p):
    return (p[0] * W, p[1] * H)


def thick(d, a, b, w, col):
    d.line((a[0], a[1], b[0], b[1]), fill=col, width=int(w))
    for c in (a, b):
        d.ellipse((c[0] - w / 2, c[1] - w / 2, c[0] + w / 2, c[1] + w / 2), fill=col)


def draw_puppet(canvas, pose, skin, head, scale=1.0):
    d = ImageDraw.Draw(canvas)
    P = {j: px(pose[j]) for j in JOINTS}
    sw = math.hypot(P["l_sho"][0] - P["r_sho"][0], P["l_sho"][1] - P["r_sho"][1]) * scale
    limb = max(10, sw * 0.26)
    # legs
    for s in ("r", "l"):
        thick(d, P[s + "_hip"], P[s + "_kne"], limb, skin["fur"])
        thick(d, P[s + "_kne"], P[s + "_ank"], limb, skin["fur"])
        a = P[s + "_ank"]
        d.ellipse((a[0] - limb * 0.8, a[1] - limb * 0.4, a[0] + limb * 0.8, a[1] + limb * 0.5), fill=skin["paw"])
    # tail
    hip = ((P["l_hip"][0] + P["r_hip"][0]) / 2, (P["l_hip"][1] + P["r_hip"][1]) / 2)
    pts = [(hip[0] + sw * 0.5 + sw * 0.6 * k / 10, hip[1] + sw * 0.2 - sw * 0.8 * math.sin(k / 10 * 2.2)) for k in range(11)]
    for k in range(10):
        thick(d, pts[k], pts[k + 1], limb * (0.9 - 0.04 * k), skin["fur_dark"] if k % 2 else skin["fur"])
    # dress: shoulders to a flared hem
    hem_y = hip[1] + sw * 0.55
    poly = [(P["r_sho"][0] + sw * 0.05, P["r_sho"][1] + sw * 0.1), (P["l_sho"][0] - sw * 0.05, P["l_sho"][1] + sw * 0.1),
            (P["l_hip"][0] + sw * 0.55, hem_y), (P["r_hip"][0] - sw * 0.55, hem_y)]
    d.polygon(poly, fill=skin["dress"])
    d.line(poly[2:] + poly[:1], fill=skin["dress_dark"], width=int(limb * 0.3))
    # arms
    for s in ("r", "l"):
        thick(d, P[s + "_sho"], P[s + "_elb"], limb, skin["fur"])
        thick(d, P[s + "_elb"], P[s + "_wri"], limb, skin["fur"])
        w_ = P[s + "_wri"]
        d.ellipse((w_[0] - limb * 0.7, w_[1] - limb * 0.7, w_[0] + limb * 0.7, w_[1] + limb * 0.7), fill=skin["paw"])
    # scarf / collar
    cx, cy = (P["l_sho"][0] + P["r_sho"][0]) / 2, (P["l_sho"][1] + P["r_sho"][1]) / 2
    d.ellipse((cx - sw * 0.42, cy - sw * 0.05, cx + sw * 0.42, cy + sw * 0.35), fill=skin["scarf"])
    # head
    hw = int(sw * 2.4)
    ang = math.degrees(math.atan2(P["l_sho"][1] - P["r_sho"][1], P["l_sho"][0] - P["r_sho"][0]))
    hh = int(hw * head.size[1] / head.size[0])
    hd = head.resize((hw, hh), Image.LANCZOS).rotate(-ang * 0.6, resample=Image.BICUBIC, expand=False)
    nx, ny = P["nose"]
    canvas.alpha_composite(hd, (int(nx - hw / 2), int(ny - hh * 0.58)))


def font(size):
    for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"):
        if os.path.isfile(cand):
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default(size)


def render(pose, skin_name, scene_name, out, label=None, scale=1.0):
    skin = SKINS[skin_name]
    head = head_image(skin)
    scene = SCENES[scene_name]
    tmp = tempfile.mkdtemp(prefix="puppet_")
    fps = pose["fps"]
    f_big, f_small = font(34), font(22)
    for i, fr in enumerate(pose["frames"]):
        t = i / fps
        canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        scene(canvas, t)
        draw_puppet(canvas, fr, skin, head, scale)
        if label:
            d = ImageDraw.Draw(canvas)
            d.rectangle((24, 24, 24 + 620, 112), fill=(0, 0, 0, 140))
            d.text((40, 32), label[0], font=f_big, fill=(255, 255, 255, 255))
            d.text((40, 76), label[1], font=f_small, fill=(230, 230, 230, 255))
        canvas.convert("RGB").save(os.path.join(tmp, "%05d.png" % i))
    os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(fps), "-i", os.path.join(tmp, "%05d.png"),
                    "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out], check=True)
    shutil.rmtree(tmp, ignore_errors=True)
    return out


def concat(files, out):
    lst = out + ".txt"
    with open(lst, "w", encoding="utf-8") as f:
        for p in files:
            f.write("file '%s'\n" % os.path.abspath(p).replace("\\", "/"))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", out], check=True)
    os.remove(lst)


# ---------------------------------------------------------------- commands
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--video", required=True); e.add_argument("--out", required=True)
    e.add_argument("--max-seconds", type=float, default=15.0)
    s = sub.add_parser("synth"); s.add_argument("--motion", default="wave_step_bow", choices=list(MOTIONS))
    s.add_argument("--duration", type=float, default=10); s.add_argument("--out", required=True)
    r = sub.add_parser("render"); r.add_argument("--pose", required=True); r.add_argument("--skin", default="kitten", choices=list(SKINS))
    r.add_argument("--scene", default="library", choices=list(SCENES)); r.add_argument("--out", required=True)
    r.add_argument("--scale", type=float, default=1.0)
    x = sub.add_parser("extend"); x.add_argument("--pose", required=True); x.add_argument("--motion", default="nod", choices=list(MOTIONS))
    x.add_argument("--duration", type=float, default=5); x.add_argument("--out", required=True)
    dm = sub.add_parser("demo"); dm.add_argument("--video", default=None); dm.add_argument("--out-dir", default="out")
    dm.add_argument("--duration", type=float, default=10)
    a = ap.parse_args()

    if a.cmd == "extract":
        json.dump(extract(a.video, max_seconds=a.max_seconds), open(a.out, "w"), separators=(",", ":"))
        print("pose", a.out)
    elif a.cmd == "synth":
        json.dump(synth(a.motion, a.duration), open(a.out, "w"), separators=(",", ":"))
        print("pose", a.out)
    elif a.cmd == "render":
        print("video", render(json.load(open(a.pose)), a.skin, a.scene, a.out, scale=a.scale))
    elif a.cmd == "extend":
        json.dump(extend(json.load(open(a.pose)), a.motion, a.duration), open(a.out, "w"), separators=(",", ":"))
        print("pose", a.out)
    elif a.cmd == "demo":
        os.makedirs(a.out_dir, exist_ok=True)
        if a.video and os.path.isfile(a.video):
            pose = extract(a.video)
        else:
            pose = synth("wave_step_bow", a.duration)
        json.dump(pose, open(os.path.join(a.out_dir, "pose.json"), "w"), separators=(",", ":"))
        src = pose["source"]
        r1 = render(pose, "kitten", "library", os.path.join(a.out_dir, "round1.mp4"),
                    label=("Round 1 - motion transfer", "source: %s  ->  kitten, library" % src))
        r2 = render(pose, "tabby", "rooftop", os.path.join(a.out_dir, "round2.mp4"),
                    label=("Round 2 - same motion, new performer + place", "same pose track  ->  tabby, rooftop at dusk"))
        ext = extend(pose, "nod", 5)
        tail = {"fps": ext["fps"], "source": ext["source"], "frames": ext["frames"][len(pose["frames"]):]}
        r3 = render(tail, "tabby", "rooftop", os.path.join(a.out_dir, "round3.mp4"),
                    label=("Round 3 - extend", "+5 s synth:nod, blended from the last frame"))
        concat([r1, r2, r3], os.path.join(a.out_dir, "all_rounds.mp4"))
        for p in (r1, r2, r3, os.path.join(a.out_dir, "all_rounds.mp4")):
            print("video", p, os.path.getsize(p), "bytes")


if __name__ == "__main__":
    main()

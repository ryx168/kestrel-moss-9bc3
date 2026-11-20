#!/usr/bin/env python
"""Zero-key multi-round motion transfer: pose -> 2D cut-out puppet.

No model API. The motion comes from a real clip (MediaPipe Pose on the CPU) or from a
built-in motion script; the puppet is drawn with Pillow from the kitten still's head crop
plus simple limbs, and ffmpeg packs the frames. "Multi-round" is the same pose track
rendered again with a different performer and place (round 2), then extended with a
second motion (round 3) - the structure of the Gemini Omni demo, with the generation
replaced by code.

    python puppet.py extract --video refs/motion.mp4 --out out/pose.json
    python puppet.py synth   --motion dance --duration 10 --out out/pose.json
    python puppet.py render  --pose out/pose.json --skin ginger --scene beach --out out/x.mp4
    python puppet.py extend  --pose out/pose.json --motion nod --duration 5 --out out/pose_ext.json
    python puppet.py demo    [--video refs/motion.mp4] --out-dir out      # three rounds + joined mp4
    python puppet.py samples --out-dir out/samples                       # 12-clip gallery, grid, reel, sheet
    python puppet.py preview --out out/preview.jpg                       # one still per sample, no ffmpeg
    python puppet.py list                                                # motions, skins, scenes
"""
import argparse, json, math, os, shutil, subprocess, sys, tempfile

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

HERE = os.path.dirname(os.path.abspath(__file__))
W, H, FPS = 1280, 720, 24
JOINTS = ["nose", "l_sho", "r_sho", "l_elb", "r_elb", "l_wri", "r_wri",
          "l_hip", "r_hip", "l_kne", "r_kne", "l_ank", "r_ank"]
UPPER = ["nose", "l_sho", "r_sho", "l_elb", "r_elb", "l_wri", "r_wri"]
MP_INDEX = {"nose": 0, "l_sho": 11, "r_sho": 12, "l_elb": 13, "r_elb": 14, "l_wri": 15, "r_wri": 16,
            "l_hip": 23, "r_hip": 24, "l_kne": 25, "r_kne": 26, "l_ank": 27, "r_ank": 28}

REST = {"nose": (0.50, 0.30), "l_sho": (0.56, 0.42), "r_sho": (0.44, 0.42),
        "l_elb": (0.60, 0.52), "r_elb": (0.40, 0.52), "l_wri": (0.59, 0.62), "r_wri": (0.41, 0.62),
        "l_hip": (0.54, 0.62), "r_hip": (0.46, 0.62), "l_kne": (0.54, 0.76), "r_kne": (0.46, 0.76),
        "l_ank": (0.54, 0.90), "r_ank": (0.46, 0.90)}
# viewer's right = performer's left ("l_*").


# ---------------------------------------------------------------- pose helpers
def smooth(a):
    a = max(0.0, min(1.0, a))
    return a * a * (3 - 2 * a)


def L(a, b, t):
    return (a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t)


def lerp_pose(p, q, a):
    return {j: L(p[j], q[j], a) for j in JOINTS}


def shift(p, dx=0.0, dy=0.0, only=None):
    return {j: ((x + dx, y + dy) if (only is None or j in only) else (x, y)) for j, (x, y) in p.items()}


def rot_about(p, joints, cx, cy, deg):
    """Rotate joints about (cx, cy) in pixel space; positive leans to the viewer's right."""
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    out = dict(p)
    for j in joints:
        dx, dy = (p[j][0] - cx) * W, (p[j][1] - cy) * H
        out[j] = (cx + (dx * c - dy * s) / W, cy + (dx * s + dy * c) / H)
    return out


def arms(p, l_elb, l_wri, a=1.0, mirror=True, r_elb=None, r_wri=None):
    p["l_elb"] = L(p["l_elb"], l_elb, a)
    p["l_wri"] = L(p["l_wri"], l_wri, a)
    if mirror:
        r_elb, r_wri = (1 - l_elb[0], l_elb[1]), (1 - l_wri[0], l_wri[1])
    if r_elb:
        p["r_elb"] = L(p["r_elb"], r_elb, a)
        p["r_wri"] = L(p["r_wri"], r_wri, a)
    return p


# ---------------------------------------------------------------- motion scripts
def pose_wave_step_bow(t, dur):
    """idle (0-2 s) -> wave (2-5) -> two steps right (5-7) -> bow (7-9) -> return."""
    p = shift(dict(REST), dy=0.004 * math.sin(t * 2 * math.pi * 0.5))
    a = smooth((t - 2.0) / 0.6) * (1 - smooth((t - 4.6) / 0.5))
    if a > 0:
        up = dict(p); up["l_elb"] = (0.66, 0.40); up["l_wri"] = (0.70, 0.28)
        p = lerp_pose(p, up, a)
        swing = 0.05 * math.sin((t - 2.6) * 2 * math.pi * 1.8) * a
        p["l_wri"] = (p["l_wri"][0] + swing, p["l_wri"][1] + abs(swing) * 0.3)
    s = smooth((t - 5.0) / 2.0)
    p = shift(p, dx=0.10 * s)
    if 5.0 <= t <= 7.0:
        ph = (t - 5.0) * 2 * math.pi
        for side, lift in (("l", max(0.0, math.sin(ph)) * 0.05), ("r", max(0.0, -math.sin(ph)) * 0.05)):
            p[side + "_kne"] = (p[side + "_kne"][0] + lift * 0.6, p[side + "_kne"][1] - lift)
            p[side + "_ank"] = (p[side + "_ank"][0] + lift * 0.8, p[side + "_ank"][1] - lift * 1.2)
    b = smooth((t - 7.0) / 0.7) * (1 - smooth((t - 8.6) / 0.6))
    if b > 0:
        p = shift(p, dy=0.08 * b, only=["nose"])
        p = shift(p, dy=0.05 * b, only=UPPER[1:])
    return p


def pose_nod(t, dur):
    p = shift(dict(REST), dy=0.004 * math.sin(t * 2 * math.pi * 0.5))
    if 0.8 <= t <= 2.4:
        p = shift(p, dy=0.04 * math.sin((t - 0.8) / 1.6 * math.pi), only=["nose"])
    if t > 3.0:
        p = arms(p, (0.60, 0.55), (0.52, 0.56), smooth((t - 3.0) / 0.8))
    return p


def pose_jumping_jacks(t, dur):
    p = dict(REST)
    s = (1 - math.cos(2 * math.pi * t / 0.9)) / 2
    p = arms(p, (0.64, 0.32), (0.63, 0.20), s)
    p["l_kne"] = L(p["l_kne"], (0.58, 0.76), s); p["l_ank"] = L(p["l_ank"], (0.62, 0.90), s)
    p["r_kne"] = L(p["r_kne"], (0.42, 0.76), s); p["r_ank"] = L(p["r_ank"], (0.38, 0.90), s)
    return shift(p, dy=-0.025 * abs(math.sin(2 * math.pi * t / 0.9)))


def pose_march(t, dur):
    p = dict(REST)
    ph = 2 * math.pi * 1.6 * t
    up_l, up_r = max(0.0, math.sin(ph)), max(0.0, -math.sin(ph))
    p["l_kne"] = (0.55, 0.76 - 0.07 * up_l); p["l_ank"] = (0.555, 0.90 - 0.10 * up_l)
    p["r_kne"] = (0.45, 0.76 - 0.07 * up_r); p["r_ank"] = (0.445, 0.90 - 0.10 * up_r)
    sw = math.sin(ph)
    p["l_elb"] = (0.60, 0.52 - 0.02 * sw); p["l_wri"] = (0.60 + 0.02 * sw, 0.61 - 0.05 * max(0, -sw))
    p["r_elb"] = (0.40, 0.52 + 0.02 * sw); p["r_wri"] = (0.40 - 0.02 * sw, 0.61 - 0.05 * max(0, sw))
    u = t / dur
    x = -0.22 + 0.44 * smooth(u * 2) if u < 0.5 else 0.22 - 0.44 * smooth((u - 0.5) * 2)
    return shift(p, dx=x, dy=-0.008 * abs(math.sin(ph)))


def pose_dance(t, dur):
    p = dict(REST)
    a = (math.sin(math.pi * t) + 1) / 2          # which arm points up, one side per second
    p["l_elb"] = L((0.62, 0.53), (0.63, 0.34), a); p["l_wri"] = L((0.555, 0.60), (0.68, 0.22), a)
    p["r_elb"] = L((0.38, 0.53), (0.37, 0.34), 1 - a); p["r_wri"] = L((0.445, 0.60), (0.32, 0.22), 1 - a)
    bounce = 0.015 * abs(math.sin(2 * math.pi * t))
    p = shift(p, dy=bounce, only=UPPER + ["l_hip", "r_hip"])
    p["l_kne"] = (p["l_kne"][0] + bounce * 0.8, p["l_kne"][1] + bounce * 0.5)
    p["r_kne"] = (p["r_kne"][0] - bounce * 0.8, p["r_kne"][1] + bounce * 0.5)
    p = shift(p, dx=0.025 * math.sin(math.pi * t), only=UPPER + ["l_hip", "r_hip", "l_kne", "r_kne"])
    return rot_about(p, UPPER, 0.5, 0.62, 5 * math.sin(math.pi * t))


def pose_jump(t, dur):
    p = dict(REST)
    c = t % 2.0
    crouch = smooth(c / 0.4) * (1 - smooth((c - 0.4) / 0.15)) + smooth((c - 1.1) / 0.15) * (1 - smooth((c - 1.4) / 0.4))
    air = math.sin(math.pi * (c - 0.55) / 0.55) if 0.55 <= c <= 1.1 else 0.0
    up = smooth((c - 0.45) / 0.15) * (1 - smooth((c - 1.0) / 0.3))
    down = 0.06 * crouch
    p = shift(p, dy=down, only=UPPER + ["l_hip", "r_hip"])
    p["l_kne"] = (p["l_kne"][0] + 0.03 * crouch, p["l_kne"][1] + down * 0.5 - 0.02 * air)
    p["r_kne"] = (p["r_kne"][0] - 0.03 * crouch, p["r_kne"][1] + down * 0.5 - 0.02 * air)
    p = arms(p, (0.64, 0.58), (0.66, 0.64), crouch * 0.7)
    p = arms(p, (0.64, 0.33), (0.66, 0.21), up)
    return shift(p, dy=-0.16 * air)


def pose_clap(t, dur):
    p = dict(REST)
    a = (1 - math.cos(2 * math.pi * t * 1.6)) / 2
    p["l_elb"] = L((0.63, 0.52), (0.60, 0.53), a); p["l_wri"] = L((0.62, 0.46), (0.506, 0.47), a)
    p["r_elb"] = L((0.37, 0.52), (0.40, 0.53), a); p["r_wri"] = L((0.38, 0.46), (0.494, 0.47), a)
    p = shift(p, dy=0.012 * a, only=["nose"])
    return shift(p, dy=-0.006 * abs(math.sin(2 * math.pi * t * 0.8)))


def pose_kick(t, dur):
    p = dict(REST)
    c = t % 1.6
    sgn = 1 if int(t // 1.6) % 2 == 0 else -1
    side = "l" if sgn > 0 else "r"
    k = math.sin(math.pi * c / 0.8) if c < 0.8 else 0.0
    p[side + "_kne"] = (0.5 + sgn * (0.05 + 0.06 * k), 0.76 - 0.08 * k)
    p[side + "_ank"] = (0.5 + sgn * (0.05 + 0.16 * k), 0.90 - 0.18 * k)
    p = arms(p, (0.64, 0.44), (0.70, 0.42), 0.6 + 0.4 * k)
    return rot_about(p, UPPER, 0.5, 0.62, -sgn * 7 * k)


def pose_stretch(t, dur):
    p = dict(REST)
    up = smooth(t / 1.2) * (1 - smooth((t - dur + 1.0) / 1.0))
    p = arms(p, (0.58, 0.30), (0.53, 0.16), up)
    bend = math.sin(2 * math.pi * (t - 1.2) / 3.0) if t > 1.2 else 0.0
    p = rot_about(p, UPPER, 0.5, 0.62, 14 * bend * up)
    return shift(p, dy=0.003 * math.sin(t * 2))


def pose_cheer(t, dur):
    p = dict(REST)
    pump = (1 - math.cos(2 * math.pi * t * 1.5)) / 2
    p = arms(p, L((0.62, 0.40), (0.62, 0.32), pump), L((0.64, 0.30), (0.66, 0.18), pump))
    return shift(p, dy=-0.03 * max(0.0, math.sin(2 * math.pi * t * 1.5)))


def pose_model(t, dur):
    """Hands on hips, a slow weight shift and a quarter-beat head tilt - the try-on pose."""
    p = dict(REST)
    p = arms(p, (0.63, 0.52), (0.555, 0.61))
    sway = math.sin(2 * math.pi * t / 2.5)
    p = shift(p, dx=0.015 * sway, only=UPPER + ["l_hip", "r_hip", "l_kne", "r_kne"])
    p = rot_about(p, ["nose"], 0.5, 0.42, 6 * sway)
    return p


MOTIONS = {"wave_step_bow": pose_wave_step_bow, "nod": pose_nod, "jumping_jacks": pose_jumping_jacks,
           "march": pose_march, "dance": pose_dance, "jump": pose_jump, "clap": pose_clap,
           "kick": pose_kick, "stretch": pose_stretch, "cheer": pose_cheer, "model": pose_model}


def synth(motion, duration, fps=FPS, speed=1.0):
    fn = MOTIONS[motion]
    n = int(round(duration * fps))
    return {"fps": fps, "source": "synth:" + motion,
            "frames": [{j: list(fn(i / fps * speed, duration * speed)[j]) for j in JOINTS} for i in range(n)]}


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
                frames.append({j: list(v) for j, v in last.items()} if last else {j: list(REST[j]) for j in JOINTS})
            idx += 1
    cap.release()
    if not frames:
        sys.exit("error: no frames read from " + video)
    dx = 0.5 - (frames[0]["l_hip"][0] + frames[0]["r_hip"][0]) / 2
    for f in frames:
        for j in JOINTS:
            f[j][0] += dx
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


# ---------------------------------------------------------------- characters
def _skin(fur, fur_dark, dress, dress_dark, scarf, paw, tint=None, acc=()):
    return dict(fur=fur, fur_dark=fur_dark, dress=dress, dress_dark=dress_dark, scarf=scarf, paw=paw,
                head="refs/kitten_head.png", tint=tint, acc=tuple(acc))


KITTEN = dict(fur=(252, 236, 214), fur_dark=(236, 196, 150), paw=(255, 248, 240))
SKINS = {
    "kitten": _skin(dress=(243, 178, 196), dress_dark=(224, 146, 170), scarf=(250, 214, 222), **KITTEN),
    "rose": _skin(dress=(200, 60, 96), dress_dark=(150, 36, 70), scarf=(255, 255, 255), acc=["bow"], **KITTEN),
    "denim": _skin(dress=(70, 104, 162), dress_dark=(48, 74, 120), scarf=(246, 206, 92), **KITTEN),
    "gala": _skin(dress=(232, 192, 84), dress_dark=(180, 140, 50), scarf=(255, 246, 220), acc=["crown"], **KITTEN),
    "tabby": _skin((176, 178, 188), (120, 122, 136), (78, 118, 190), (54, 88, 150), (230, 230, 236), (236, 236, 240),
                   tint=((40, 40, 50), (150, 152, 165), (235, 235, 240)), acc=["glasses"]),
    "ginger": _skin((236, 160, 90), (200, 110, 50), (90, 140, 90), (60, 104, 60), (240, 220, 120), (255, 236, 210),
                    tint=((60, 30, 10), (225, 140, 60), (255, 240, 220)), acc=["cap"]),
    "snow": _skin((245, 246, 250), (210, 214, 226), (150, 190, 230), (110, 150, 200), (220, 90, 90), (255, 255, 255),
                  tint=((70, 70, 90), (215, 220, 235), (255, 255, 255)), acc=["crown"]),
    "midnight": _skin((70, 62, 96), (45, 40, 66), (230, 200, 80), (190, 160, 50), (120, 200, 220), (200, 190, 230),
                      tint=((15, 12, 25), (80, 70, 110), (210, 200, 240)), acc=["headphones"]),
}
_HEADS = {}


def head_image(skin_name):
    if skin_name in _HEADS:
        return _HEADS[skin_name]
    skin = SKINS[skin_name]
    im = Image.open(os.path.join(HERE, skin["head"])).convert("RGBA")
    if skin["tint"]:
        lo, mid, hi = skin["tint"]
        col = ImageOps.colorize(ImageOps.grayscale(im), lo, hi, mid=mid).convert("RGBA")
        col.putalpha(im.getchannel("A"))
        im = col
    d = ImageDraw.Draw(im)
    w, h = im.size
    lw = max(2, w // 60)
    for acc in skin["acc"]:
        if acc == "glasses":
            for cx in (w * 0.36, w * 0.64):
                d.ellipse((cx - w * 0.13, h * 0.42, cx + w * 0.13, h * 0.68), outline=(40, 40, 50, 255), width=lw)
            d.line((w * 0.49, h * 0.55, w * 0.51, h * 0.55), fill=(40, 40, 50, 255), width=lw)
        elif acc == "bow":
            cx, cy, s = w * 0.66, h * 0.16, w * 0.11
            d.polygon([(cx, cy), (cx - s, cy - s * 0.7), (cx - s, cy + s * 0.7)], fill=(220, 40, 70, 255))
            d.polygon([(cx, cy), (cx + s, cy - s * 0.7), (cx + s, cy + s * 0.7)], fill=(220, 40, 70, 255))
            d.ellipse((cx - s * 0.3, cy - s * 0.3, cx + s * 0.3, cy + s * 0.3), fill=(250, 90, 120, 255))
        elif acc == "crown":
            x0, x1, y0, y1 = w * 0.34, w * 0.66, h * 0.02, h * 0.20
            pts = [(x0, y1), (x0, y0 + (y1 - y0) * 0.3), (x0 + (x1 - x0) * 0.25, y1 - (y1 - y0) * 0.45),
                   (w * 0.5, y0), (x1 - (x1 - x0) * 0.25, y1 - (y1 - y0) * 0.45), (x1, y0 + (y1 - y0) * 0.3), (x1, y1)]
            d.polygon(pts, fill=(240, 196, 60, 255), outline=(170, 120, 20, 255))
            for jx, col in ((0.42, (220, 40, 60)), (0.5, (60, 120, 220)), (0.58, (40, 170, 90))):
                d.ellipse((w * jx - w * 0.02, y1 - h * 0.07, w * jx + w * 0.02, y1 - h * 0.03), fill=col + (255,))
        elif acc == "cap":
            d.chord((w * 0.24, h * 0.00, w * 0.76, h * 0.40), 180, 360, fill=(40, 90, 200, 255))
            d.ellipse((w * 0.50, h * 0.15, w * 0.95, h * 0.25), fill=(30, 70, 170, 255))
            d.ellipse((w * 0.48, h * 0.00, w * 0.52, h * 0.04), fill=(240, 240, 240, 255))
        elif acc == "headphones":
            d.arc((w * 0.06, -h * 0.02, w * 0.94, h * 0.98), 190, 350, fill=(30, 30, 36, 255), width=max(4, w // 20))
            for cx in (w * 0.08, w * 0.92):
                d.ellipse((cx - w * 0.08, h * 0.38, cx + w * 0.08, h * 0.62), fill=(230, 60, 90, 255), outline=(30, 30, 36, 255), width=lw)
    _HEADS[skin_name] = im
    return im


# ---------------------------------------------------------------- scenes
def rnd(i, k=0):
    """Deterministic hash noise in [0, 1) (no Math.random, same frames every run)."""
    v = (i * 2654435761 + k * 40503 + 12345) & 0xFFFFFFFF
    v ^= v >> 15
    v = (v * 2246822519) & 0xFFFFFFFF
    v ^= v >> 13
    return v / 4294967296.0


def vgrad(d, y0, y1, c0, c1, x0=0, x1=W):
    y0, y1 = int(y0), int(y1)
    for y in range(y0, y1):
        a = (y - y0) / max(1, y1 - y0 - 1)
        d.line((x0, y, x1, y), fill=tuple(int(c0[k] + (c1[k] - c0[k]) * a) for k in range(3)))


_BG = {}


def cached(name, fn):
    if name not in _BG:
        im = Image.new("RGBA", (W, H), (0, 0, 0, 255))
        fn(im, ImageDraw.Draw(im))
        _BG[name] = im
    return _BG[name]


def bg_library(im, d):
    vgrad(d, 0, H, (214, 178, 132), (154, 108, 62))
    d.rectangle((0, H * 0.78, W, H), fill=(150, 108, 72))
    for i in range(0, W + 60, 90):
        d.line((i, H * 0.78, i - 60, H), fill=(128, 90, 58), width=2)
    for x0 in (60, 300, 940):
        d.rectangle((x0, 60, x0 + 200, H * 0.78), fill=(88, 56, 34))
        for s in range(5):
            y = 80 + s * 95
            d.rectangle((x0 + 8, y, x0 + 192, y + 6), fill=(60, 38, 22))
            bx, k = x0 + 12, 0
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
    im.alpha_composite(glow.filter(ImageFilter.GaussianBlur(18)))


def bg_rooftop(im, d):
    vgrad(d, 0, H, (40, 50, 130), (255, 160, 110))
    d.ellipse((880, 120, 960, 200), fill=(255, 230, 190))
    hx, k = 0, 0
    while hx < W:
        bw, bh = 40 + (k * 37) % 70, 120 + (k * 53) % 200
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


def fx_rooftop(canvas, d, t):
    for i in range(0, W, 60):
        y = H * 0.56 + 10 * math.sin(i / 60 * 1.3)
        b = 0.75 + 0.25 * math.sin(t * 3 + i)
        d.ellipse((i - 5, y - 5, i + 5, y + 5), fill=(int(255 * b), int(225 * b), int(140 * b)))


def pine(d, x, base, h, col, snow=None):
    d.rectangle((x - h * 0.04, base - h * 0.15, x + h * 0.04, base), fill=(90, 60, 40))
    for k in range(3):
        top = base - h * (0.45 + 0.27 * k)
        half = h * (0.32 - 0.07 * k)
        bot = base - h * (0.12 + 0.24 * k)
        d.polygon([(x, top), (x - half, bot), (x + half, bot)], fill=col)
        if snow:
            d.polygon([(x, top), (x - half * 0.35, top + (bot - top) * 0.35), (x + half * 0.35, top + (bot - top) * 0.35)], fill=snow)


def bg_forest(im, d):
    vgrad(d, 0, H * 0.62, (140, 196, 236), (214, 236, 236))
    for k in range(6):
        cx = rnd(k, 7) * W
        d.ellipse((cx - 300, H * 0.48, cx + 300, H * 0.80), fill=(104, 150, 104))
    for k in range(14):
        x = 40 + k * 92 + rnd(k, 3) * 40
        if 420 < x < 860:
            continue
        pine(d, x, H * 0.80, 230 + rnd(k, 4) * 140, (40 + int(rnd(k, 5) * 30), 100, 60))
    vgrad(d, H * 0.78, H, (96, 156, 72), (70, 120, 52))
    d.ellipse((W * 0.30, H * 0.84, W * 0.70, H * 1.10), fill=(176, 146, 100))


def fx_forest(canvas, d, t):
    for i in range(14):
        x = (rnd(i) * W + 40 * math.sin(t * 1.3 + i)) % W
        y = (rnd(i, 1) * H + t * (30 + 30 * rnd(i, 2))) % H
        c = [(214, 120, 40), (230, 170, 50), (170, 80, 40)][i % 3]
        d.ellipse((x - 6, y - 3, x + 6, y + 3), fill=c)


def bg_beach(im, d):
    vgrad(d, 0, H * 0.56, (110, 180, 236), (250, 220, 180))
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse((960, 70, 1160, 270), fill=(255, 230, 160, 120))
    im.alpha_composite(glow.filter(ImageFilter.GaussianBlur(30)))
    d.ellipse((1020, 130, 1100, 210), fill=(255, 240, 190))
    vgrad(d, H * 0.56, H * 0.76, (40, 120, 180), (90, 180, 200))
    vgrad(d, H * 0.76, H, (240, 216, 160), (220, 190, 130))
    pts = [(110 + 60 * (k / 10) ** 2, H * 0.92 - k * 36) for k in range(11)]
    for k in range(10):
        d.line((pts[k], pts[k + 1]), fill=(130, 90, 50), width=22 - k)
    tx, ty = pts[-1]
    for a in range(0, 360, 45):
        r = math.radians(a)
        d.polygon([(tx, ty), (tx + 150 * math.cos(r) - 20 * math.sin(r), ty + 70 * math.sin(r) + 40),
                   (tx + 150 * math.cos(r) + 20 * math.sin(r), ty + 70 * math.sin(r) + 60)], fill=(50, 140, 70))


def fx_beach(canvas, d, t):
    for k in range(5):
        y0 = H * 0.59 + k * H * 0.035
        pts = [(x, y0 + 3 * math.sin(x / 50 + t * 1.6 + k)) for x in range(0, W + 20, 20)]
        d.line(pts, fill=(200, 236, 246), width=2)
    pts = [(x, H * 0.76 + 6 * math.sin(x / 80 + t * 2.0)) for x in range(0, W + 20, 20)]
    d.line(pts, fill=(250, 250, 250), width=8)
    for k in range(3):
        cx = (rnd(k, 9) * W + 18 * t * (1 + k * 0.3)) % (W + 300) - 150
        cy = 60 + k * 50
        for j in range(4):
            d.ellipse((cx + j * 40 - 50, cy - 20 - (j % 2) * 12, cx + j * 40 + 30, cy + 20), fill=(255, 255, 255))


def bg_snow(im, d):
    vgrad(d, 0, H * 0.80, (20, 30, 70), (80, 100, 150))
    d.ellipse((150, 80, 240, 170), fill=(240, 240, 220))
    for k in range(5):
        cx, top = 100 + k * 280 + rnd(k, 2) * 80, H * (0.30 + rnd(k, 3) * 0.12)
        d.polygon([(cx, top), (cx - 260, H * 0.80), (cx + 260, H * 0.80)], fill=(70, 80, 120))
        d.polygon([(cx, top), (cx - 60, top + 50), (cx - 20, top + 40), (cx + 10, top + 58), (cx + 60, top + 50)], fill=(236, 240, 250))
    for k in range(12):
        x = 30 + k * 110 + rnd(k, 5) * 40
        if 430 < x < 850:
            continue
        pine(d, x, H * 0.82, 200 + rnd(k, 6) * 120, (30, 70, 60), snow=(240, 244, 250))
    vgrad(d, H * 0.80, H, (230, 236, 246), (200, 210, 230))


def fx_snow(canvas, d, t):
    for i in range(140):
        r = 1.5 + 2.5 * rnd(i, 3)
        x = (rnd(i) * W + 25 * math.sin(t * 0.8 + i)) % W
        y = (rnd(i, 1) * H + t * (40 + 70 * rnd(i, 2))) % H
        d.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255))


def bg_stage(im, d):
    vgrad(d, 0, H * 0.80, (30, 12, 40), (60, 20, 50))
    for x in range(0, 230, 2):
        v = 0.6 + 0.4 * math.sin(x / 9.0)
        col = (int(150 * v), int(20 * v), int(36 * v))
        d.line((x, 0, x, H * 0.82), fill=col)
        d.line((W - x, 0, W - x, H * 0.82), fill=col)
    d.rectangle((0, 0, W, 40), fill=(120, 16, 30))
    vgrad(d, H * 0.80, H, (90, 56, 36), (60, 36, 24))
    for i in range(-6, 8):
        d.line((W / 2 + i * 40, H * 0.80, W / 2 + i * 160, H), fill=(70, 44, 28), width=2)


def fx_stage(canvas, d, t):
    ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    o = ImageDraw.Draw(ov)
    beams = [(320, (255, 120, 200)), (640, (255, 250, 220)), (960, (120, 200, 255))]
    for k, (sx, col) in enumerate(beams):
        tx = W / 2 + (0 if k == 1 else 260 * math.sin(t * 1.2 + k * 2.1))
        o.polygon([(sx - 20, 0), (sx + 20, 0), (tx + 130, H * 0.92), (tx - 130, H * 0.92)], fill=col + (46,))
        o.ellipse((tx - 140, H * 0.86, tx + 140, H * 0.97), fill=col + (70,))
    canvas.alpha_composite(ov.filter(ImageFilter.GaussianBlur(10)))
    for i in range(16):
        a = t * 1.5 + i * 0.7
        x, y = W / 2 + 26 * math.cos(a), 80 + 26 * math.sin(a) * 0.5
        b = 0.5 + 0.5 * math.sin(t * 6 + i)
        d.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(int(255 * b), int(255 * b), int(255 * b)))


def bg_space(im, d):
    vgrad(d, 0, H, (4, 6, 20), (20, 16, 50))
    for i in range(320):
        x, y, b = rnd(i) * W, rnd(i, 1) * H * 0.82, 120 + int(rnd(i, 2) * 135)
        r = 0.6 + rnd(i, 3) * 1.4
        d.ellipse((x - r, y - r, x + r, y + r), fill=(b, b, min(255, b + 20)))
    d.ellipse((930, 70, 1130, 270), fill=(210, 140, 90))
    for k in range(5):
        d.arc((930, 70 + k * 8, 1130, 270 - k * 8), 200, 340, fill=(180, 110, 70), width=6)
    d.arc((880, 140, 1180, 200), 0, 180, fill=(236, 210, 160), width=6)
    vgrad(d, H * 0.80, H, (150, 150, 160), (100, 100, 112))
    for i in range(12):
        x, y, r = rnd(i, 5) * W, H * (0.83 + rnd(i, 6) * 0.14), 14 + rnd(i, 7) * 40
        d.ellipse((x - r, y - r * 0.3, x + r, y + r * 0.3), fill=(120, 120, 132), outline=(176, 176, 186), width=2)


def fx_space(canvas, d, t):
    for i in range(24):
        if math.sin(t * 4 + i * 1.7) > 0.6:
            x, y = rnd(i + 900) * W, rnd(i + 900, 1) * H * 0.7
            d.line((x - 6, y, x + 6, y), fill=(255, 255, 255)); d.line((x, y - 6, x, y + 6), fill=(255, 255, 255))
    c = (t % 3.0) / 0.8
    if c < 1:
        x0, y0 = 100 + 900 * c, 60 + 160 * c
        d.line((x0 - 120, y0 - 22, x0, y0), fill=(255, 250, 220), width=3)


def bg_park(im, d):
    vgrad(d, 0, H * 0.62, (130, 196, 250), (220, 240, 255))
    for k, (cx, col) in enumerate(((200, (120, 180, 110)), (700, (110, 170, 100)), (1150, (126, 186, 116)))):
        d.ellipse((cx - 420, H * 0.50, cx + 420, H * 0.90), fill=col)
    for k, x in enumerate((90, 250, 1050, 1190)):
        base = H * 0.78
        d.rectangle((x - 9, base - 120, x + 9, base), fill=(110, 76, 50))
        for j in range(4):
            r = 52 - j * 4
            d.ellipse((x - r + (j - 1.5) * 26, base - 200 - (j % 2) * 30 - r, x + r + (j - 1.5) * 26, base - 200 - (j % 2) * 30 + r),
                      fill=(60 + j * 8, 140 + j * 6, 70))
    vgrad(d, H * 0.76, H, (110, 176, 80), (84, 146, 60))
    d.rectangle((900, H * 0.76, 1080, H * 0.79), fill=(140, 96, 60))
    for x in (912, 1060):
        d.rectangle((x, H * 0.79, x + 8, H * 0.86), fill=(70, 70, 76))
    d.polygon([(W * 0.40, H * 0.76), (W * 0.60, H * 0.76), (W * 0.78, H), (W * 0.22, H)], fill=(214, 196, 160))


def fx_park(canvas, d, t):
    for k in range(4):
        cx = (rnd(k, 11) * W + 14 * t * (1 + k * 0.4)) % (W + 300) - 150
        cy = 50 + k * 40
        for j in range(4):
            d.ellipse((cx + j * 34 - 40, cy - 16 - (j % 2) * 10, cx + j * 34 + 26, cy + 16), fill=(255, 255, 255))
    for k in range(3):
        a = t * 3 + k * 2
        x, y = 300 + k * 330 + 60 * math.sin(a * 0.5), 300 + 30 * math.sin(a)
        wing = 6 + 4 * math.sin(t * 18 + k)
        col = [(250, 200, 60), (230, 90, 140), (120, 160, 250)][k]
        d.ellipse((x - wing, y - 6, x, y + 4), fill=col); d.ellipse((x, y - 6, x + wing, y + 4), fill=col)


def bg_studio(im, d):
    vgrad(d, 0, H * 0.75, (226, 228, 232), (200, 202, 208))
    vgrad(d, H * 0.75, H, (200, 202, 208), (176, 178, 186))
    glow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(glow).ellipse((W * 0.30, H * 0.20, W * 0.70, H * 0.85), fill=(255, 255, 255, 90))
    im.alpha_composite(glow.filter(ImageFilter.GaussianBlur(60)))
    shadow = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(shadow).ellipse((W * 0.40, H * 0.88, W * 0.60, H * 0.95), fill=(0, 0, 0, 60))
    im.alpha_composite(shadow.filter(ImageFilter.GaussianBlur(8)))


SCENES = {
    "library": (bg_library, None), "rooftop": (bg_rooftop, fx_rooftop), "forest": (bg_forest, fx_forest),
    "beach": (bg_beach, fx_beach), "snow": (bg_snow, fx_snow), "stage": (bg_stage, fx_stage),
    "space": (bg_space, fx_space), "park": (bg_park, fx_park), "studio": (bg_studio, None),
}


def paint_scene(canvas, name, t):
    bg, fx = SCENES[name]
    canvas.paste(cached(name, bg), (0, 0))
    if fx:
        fx(canvas, ImageDraw.Draw(canvas), t)


# ---------------------------------------------------------------- puppet
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
    for s in ("r", "l"):
        thick(d, P[s + "_hip"], P[s + "_kne"], limb, skin["fur"])
        thick(d, P[s + "_kne"], P[s + "_ank"], limb, skin["fur"])
        a = P[s + "_ank"]
        d.ellipse((a[0] - limb * 0.8, a[1] - limb * 0.4, a[0] + limb * 0.8, a[1] + limb * 0.5), fill=skin["paw"])
    hip = ((P["l_hip"][0] + P["r_hip"][0]) / 2, (P["l_hip"][1] + P["r_hip"][1]) / 2)
    pts = [(hip[0] + sw * 0.5 + sw * 0.6 * k / 10, hip[1] + sw * 0.2 - sw * 0.8 * math.sin(k / 10 * 2.2)) for k in range(11)]
    for k in range(10):
        thick(d, pts[k], pts[k + 1], limb * (0.9 - 0.04 * k), skin["fur_dark"] if k % 2 else skin["fur"])
    hem_y = hip[1] + sw * 0.55
    poly = [(P["r_sho"][0] + sw * 0.05, P["r_sho"][1] + sw * 0.1), (P["l_sho"][0] - sw * 0.05, P["l_sho"][1] + sw * 0.1),
            (P["l_hip"][0] + sw * 0.55, hem_y), (P["r_hip"][0] - sw * 0.55, hem_y)]
    d.polygon(poly, fill=skin["dress"])
    d.line(poly[2:] + poly[:1], fill=skin["dress_dark"], width=int(limb * 0.3))
    for s in ("r", "l"):
        thick(d, P[s + "_sho"], P[s + "_elb"], limb, skin["fur"])
        thick(d, P[s + "_elb"], P[s + "_wri"], limb, skin["fur"])
        w_ = P[s + "_wri"]
        d.ellipse((w_[0] - limb * 0.7, w_[1] - limb * 0.7, w_[0] + limb * 0.7, w_[1] + limb * 0.7), fill=skin["paw"])
    cx, cy = (P["l_sho"][0] + P["r_sho"][0]) / 2, (P["l_sho"][1] + P["r_sho"][1]) / 2
    d.ellipse((cx - sw * 0.42, cy - sw * 0.05, cx + sw * 0.42, cy + sw * 0.35), fill=skin["scarf"])
    hw = int(sw * 2.4)
    tilt = math.degrees(math.atan2(P["l_sho"][1] - P["r_sho"][1], P["l_sho"][0] - P["r_sho"][0]))
    nx, ny = P["nose"]
    neck = math.degrees(math.atan2(nx - cx, cy - ny))          # head follows a sideways neck lean
    hh = int(hw * head.size[1] / head.size[0])
    hd = head.resize((hw, hh), Image.LANCZOS).rotate(-(tilt * 0.6 + neck * 0.5), resample=Image.BICUBIC, expand=False)
    canvas.alpha_composite(hd, (int(nx - hw / 2), int(ny - hh * 0.58)))


def font(size):
    for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "C:/Windows/Fonts/arialbd.ttf"):
        if os.path.isfile(cand):
            return ImageFont.truetype(cand, size)
    return ImageFont.load_default(size)


_FONTS = {}


def draw_label(canvas, label):
    if not label:
        return
    if not _FONTS:
        _FONTS["big"], _FONTS["small"] = font(34), font(22)
    d = ImageDraw.Draw(canvas)
    w = int(max(d.textlength(label[0], font=_FONTS["big"]), d.textlength(label[1], font=_FONTS["small"]))) + 34
    box = Image.new("RGBA", (w, 90), (0, 0, 0, 150))
    canvas.alpha_composite(box, (24, 24))
    d.text((40, 32), label[0], font=_FONTS["big"], fill=(255, 255, 255, 255))
    d.text((40, 76), label[1], font=_FONTS["small"], fill=(230, 230, 230, 255))


def frame_at(fr, t, skin_name, scene_name, label=None, scale=1.0, flash=0.0):
    canvas = Image.new("RGBA", (W, H), (0, 0, 0, 255))
    paint_scene(canvas, scene_name, t)
    draw_puppet(canvas, fr, SKINS[skin_name], head_image(skin_name), scale)
    if flash > 0:
        canvas = Image.blend(canvas, Image.new("RGBA", (W, H), (255, 255, 255, 255)), min(0.85, flash))
    draw_label(canvas, label)
    return canvas.convert("RGB")


def render(pose, skin_name, scene_name, out, label=None, scale=1.0, schedule=None, thumb=None):
    """schedule: [(t0, skin, scene, label), ...] switches performer / place mid-clip (hard cut + flash)."""
    tmp = tempfile.mkdtemp(prefix="puppet_")
    fps = pose["fps"]
    n = len(pose["frames"])
    for i, fr in enumerate(pose["frames"]):
        t = i / fps
        sk, sc, lb, fl = skin_name, scene_name, label, 0.0
        if schedule:
            for t0, s1, s2, l1 in schedule:
                if t >= t0:
                    sk, sc, lb = s1, s2, (l1 or label)
                    fl = (1 - (t - t0) / 0.18) if (t0 > 0 and t - t0 < 0.18) else 0.0
        im = frame_at(fr, t, sk, sc, lb, scale, fl)
        im.save(os.path.join(tmp, "%05d.png" % i))
        if thumb and i == n // 2:
            im.resize((W // 4, H // 4), Image.LANCZOS).save(thumb)
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


# ---------------------------------------------------------------- samples
# name, motion, duration (s), speed, skin, scene, title, schedule
WORLDS = ["library", "forest", "beach", "snow", "stage"]
SAMPLES = [
    ("01_jumping_jacks", "jumping_jacks", 6, 1.0, "ginger", "park", "Jumping jacks", None),
    ("02_dance", "dance", 6, 1.0, "rose", "stage", "Dance", None),
    ("03_march", "march", 6, 1.0, "tabby", "forest", "March across", None),
    ("04_jump", "jump", 6, 1.0, "snow", "snow", "Jump", None),
    ("05_clap", "clap", 6, 1.0, "midnight", "space", "Clap", None),
    ("06_kick", "kick", 6, 1.0, "ginger", "beach", "Side kicks", None),
    ("07_stretch", "stretch", 6, 1.0, "kitten", "library", "Stretch", None),
    ("08_cheer", "cheer", 6, 1.0, "gala", "rooftop", "Cheer", None),
    ("09_wave_step_bow", "wave_step_bow", 6, 1.6, "denim", "park", "Wave, step, bow", None),
    ("10_five_worlds", "dance", 10, 1.0, "kitten", "library", "Same performance, five worlds",
     [(k * 2.0, "kitten", w, ("Same performance, five worlds", "dance  ->  kitten  ->  " + w)) for k, w in enumerate(WORLDS)]),
    ("11_outfit_swap", "model", 10, 1.0, "kitten", "studio", "Virtual try-on",
     [(k * 2.5, s, "studio", ("Virtual try-on", "same model, outfit %d of 4: %s" % (k + 1, s)))
      for k, s in enumerate(["kitten", "rose", "denim", "gala"])]),
    ("12_performer_swap", "jumping_jacks", 9, 1.0, "kitten", "park", "Same motion, six performers",
     [(k * 1.5, s, "park", ("Same motion, six performers", "jumping_jacks  ->  " + s))
      for k, s in enumerate(["kitten", "tabby", "ginger", "snow", "midnight", "gala"])]),
]


def sample_label(s):
    name, motion, dur, speed, skin, scene, title, sched = s
    return (title, "synth:%s  ->  %s, %s" % (motion, skin, scene))


def make_sheet(thumbs, out, cols=4):
    tw, th = W // 4, H // 4
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * 8, rows * th + (rows + 1) * 8), (24, 24, 28))
    for k, p in enumerate(thumbs):
        im = Image.open(p) if isinstance(p, str) else p
        sheet.paste(im, (8 + (k % cols) * (tw + 8), 8 + (k // cols) * (th + 8)))
    sheet.save(out, quality=90)
    return out


def samples(out_dir, only=None):
    os.makedirs(out_dir, exist_ok=True)
    clips, thumbs = [], []
    for s in SAMPLES:
        name, motion, dur, speed, skin, scene, title, sched = s
        if only and name not in only:
            continue
        pose = synth(motion, dur, speed=speed)
        out = os.path.join(out_dir, name + ".mp4")
        thumb = os.path.join(out_dir, name + ".jpg")
        render(pose, skin, scene, out, label=sample_label(s), schedule=sched, thumb=thumb)
        clips.append(out)
        thumbs.append(thumb)
        print("video", out, os.path.getsize(out), "bytes")
    make_sheet(thumbs, os.path.join(out_dir, "samples_sheet.jpg"))
    concat(clips, os.path.join(out_dir, "samples_reel.mp4"))
    grid = [c for c in clips if os.path.basename(c)[:2] in ("01", "02", "03", "04", "05", "06", "07", "08", "09")]
    if len(grid) == 9:
        ins = sum((["-i", c] for c in grid), [])
        f = ";".join("[%d:v]scale=426:240,setsar=1[v%d]" % (k, k) for k in range(9))
        lay = "|".join("%d_%d" % ((k % 3) * 426, (k // 3) * 240) for k in range(9))
        f += ";" + "".join("[v%d]" % k for k in range(9)) + "xstack=inputs=9:layout=" + lay + "[g]"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error"] + ins + ["-filter_complex", f, "-map", "[g]", "-t", "6",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                        os.path.join(out_dir, "samples_grid.mp4")], check=True)
    for p in ("samples_reel.mp4", "samples_grid.mp4", "samples_sheet.jpg"):
        q = os.path.join(out_dir, p)
        if os.path.isfile(q):
            print("video" if q.endswith(".mp4") else "image", q, os.path.getsize(q), "bytes")


def preview(out):
    """One still per sample (at 40 % and, for scheduled samples, mid-switch), no ffmpeg."""
    stills = []
    for s in SAMPLES:
        name, motion, dur, speed, skin, scene, title, sched = s
        t = dur * 0.4
        fr = {j: MOTIONS[motion](t * speed, dur * speed)[j] for j in JOINTS}
        sk, sc, lb = skin, scene, sample_label(s)
        if sched:
            for t0, s1, s2, l1 in sched:
                if t >= t0:
                    sk, sc, lb = s1, s2, l1
        stills.append(frame_at(fr, t, sk, sc, lb).resize((W // 4, H // 4), Image.LANCZOS))
    return make_sheet(stills, out)


# ---------------------------------------------------------------- commands
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--video", required=True); e.add_argument("--out", required=True)
    e.add_argument("--max-seconds", type=float, default=15.0)
    s = sub.add_parser("synth"); s.add_argument("--motion", default="wave_step_bow", choices=list(MOTIONS))
    s.add_argument("--duration", type=float, default=10); s.add_argument("--speed", type=float, default=1.0)
    s.add_argument("--out", required=True)
    r = sub.add_parser("render"); r.add_argument("--pose", required=True); r.add_argument("--skin", default="kitten", choices=list(SKINS))
    r.add_argument("--scene", default="library", choices=list(SCENES)); r.add_argument("--out", required=True)
    r.add_argument("--scale", type=float, default=1.0)
    x = sub.add_parser("extend"); x.add_argument("--pose", required=True); x.add_argument("--motion", default="nod", choices=list(MOTIONS))
    x.add_argument("--duration", type=float, default=5); x.add_argument("--out", required=True)
    dm = sub.add_parser("demo"); dm.add_argument("--video", default=None); dm.add_argument("--out-dir", default="out")
    dm.add_argument("--duration", type=float, default=10)
    sm = sub.add_parser("samples"); sm.add_argument("--out-dir", default="out/samples")
    sm.add_argument("--only", nargs="*", help="sample names, e.g. 02_dance 10_five_worlds")
    pv = sub.add_parser("preview"); pv.add_argument("--out", default="out/preview.jpg")
    sub.add_parser("list")
    a = ap.parse_args()

    if a.cmd == "extract":
        json.dump(extract(a.video, max_seconds=a.max_seconds), open(a.out, "w"), separators=(",", ":"))
        print("pose", a.out)
    elif a.cmd == "synth":
        json.dump(synth(a.motion, a.duration, speed=a.speed), open(a.out, "w"), separators=(",", ":"))
        print("pose", a.out)
    elif a.cmd == "render":
        print("video", render(json.load(open(a.pose)), a.skin, a.scene, a.out, scale=a.scale))
    elif a.cmd == "extend":
        json.dump(extend(json.load(open(a.pose)), a.motion, a.duration), open(a.out, "w"), separators=(",", ":"))
        print("pose", a.out)
    elif a.cmd == "samples":
        samples(a.out_dir, a.only)
    elif a.cmd == "preview":
        os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
        print("image", preview(a.out))
    elif a.cmd == "list":
        print("motions:", " ".join(MOTIONS)); print("skins:  ", " ".join(SKINS)); print("scenes: ", " ".join(SCENES))
    elif a.cmd == "demo":
        os.makedirs(a.out_dir, exist_ok=True)
        pose = extract(a.video) if a.video and os.path.isfile(a.video) else synth("wave_step_bow", a.duration)
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

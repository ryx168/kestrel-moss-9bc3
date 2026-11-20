# Multi-round motion transfer on GitHub Actions

Rebuilds demo 1 of the Gemini Omni 1.2 Flash thread (x.com/FinanceYF5/status/2118537565884252564,
多轮动作迁移): a motion is moved onto a character, the result is re-rendered with a new
performer and place, then extended. Nothing renders on a PC; everything runs in Actions.

Two paths, both in `.github/workflows/motion.yml`:

| path | needs | what it does |
|---|---|---|
| **puppet** (default, zero-key) | nothing | MediaPipe Pose reads the motion from a clip (or a built-in wave / step / bow script), Pillow draws a 2D cut-out puppet from the kitten still, ffmpeg packs it. Round 1 kitten in a library, round 2 the same pose track on a grey tabby on a rooftop, round 3 a 5 s nod blended from the last frame. ~1 min. |
| **ai** (opt-in) | one repo secret: `XAI_API_KEY` (Grok Imagine), `ARK_API_KEY` (BytePlus Seedance) or `FAL_KEY` (Kling / Veo / Luma ...) | clones the MIT Agent Sprite Forge route layer and animates the kitten still with the first provider whose key exists; round 2 starts from round 1's last frame. |

Gemini is not used. The Gemini-only runner (`run_omni.py`, `omni.yml`) is kept for reference
and needs prepaid credit that this estate does not have.

## Run
```
# zero-key, built-in motion (also runs on every push that touches puppet.py or refs/)
gh workflow run motion.yml -R ryx168/kestrel-moss-9bc3

# zero-key, motion from a real clip you committed to refs/
gh workflow run motion.yml -R ryx168/kestrel-moss-9bc3 -f video=refs/motion.mp4

# add the AI path (after setting one of the three secrets)
gh workflow run motion.yml -R ryx168/kestrel-moss-9bc3 -f ai=true -f ai_resolution=720p
```
Outputs: artifact `puppet-run-<n>` and Release `puppet-<n>` with `all_rounds.mp4`, the three
round clips, `review.jpg` (18 sampled frames) and `pose.json` (the motion track, reusable).
The AI path publishes `ai-<n>` with its clips and prints the Forge spend ledger.

## Sample gallery (zero-key)
Every push and every dispatch (unless `-f samples=false`) also renders 12 clips into the
same Release:

| clip | motion | performer | place |
|---|---|---|---|
| 01 | jumping jacks | ginger, cap | park |
| 02 | dance | rose, bow | stage with sweeping spotlights |
| 03 | march across and back | tabby, glasses | forest, falling leaves |
| 04 | jump | snow, crown | snowfall at night |
| 05 | clap | midnight, headphones | moon surface, shooting star |
| 06 | side kicks | ginger | beach, moving surf |
| 07 | stretch | kitten | library |
| 08 | cheer | gala, crown | rooftop at dusk |
| 09 | wave, step, bow | denim | park |
| 10 | dance, **five worlds** (thread demo 4) | kitten | library, forest, beach, snow, stage, a cut every 2 s |
| 11 | **virtual try-on** (thread demo 3) | same kitten, four outfits | studio, a swap every 2.5 s |
| 12 | jumping jacks, **six performers** (round 2 idea) | kitten, tabby, ginger, snow, midnight, gala | park |

Plus `samples_grid.mp4` (clips 1-9 as a 3x3 wall), `samples_reel.mp4` (all 12 back to back)
and `samples_sheet.jpg` (one frame from each).

```
python puppet.py list                               # motions, skins, scenes
python puppet.py preview --out out/preview.jpg      # one still per sample in about a second
python puppet.py samples --only 02_dance 10_five_worlds
```

## Setting a secret (owner, PowerShell)
```
gh secret set XAI_API_KEY -R ryx168/kestrel-moss-9bc3
```
Prices per 10 s clip at 720p (Forge price table, 2026-10-06): Grok Imagine about $0.50-1.00,
Seedance mini $0.80, Kling via fal.ai about $1-3. BytePlus needs a $30 minimum balance.

## Local
```
pip install pillow mediapipe==0.10.21     # mediapipe only for --video
python puppet.py demo --out-dir out        # three rounds + all_rounds.mp4 (needs ffmpeg)
python puppet.py extract --video refs/motion.mp4 --out out/pose.json
python puppet.py render --pose out/pose.json --skin tabby --scene rooftop --out out/x.mp4
```
Run `python puppet.py list` for every motion, skin and scene.
Add a skin or scene by extending the `SKINS` / `SCENES` tables in `puppet.py`.

## The other eight demos (prompts for the AI path)
`prompts/demo2..demo9` cover the rest of the thread: dialogue lip-sync, virtual try-on,
one performance in five worlds, stable on-screen text, an outside-to-inside one-shot, a
style-reference explainer, a two-photo Pantheon flythrough, a coastline grown into an island.
They are written for a reference-driven video model; pass one as `ai_prompt` once a key is set.

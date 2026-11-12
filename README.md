# Gemini Omni 1.2 Flash on GitHub Actions

Multi-round motion transfer, the first of the nine demos in
x.com/FinanceYF5/status/2118537565884252564, rebuilt as a workflow: round 1 moves the
motion of a reference clip (or a written motion) onto a character still, round 2 swaps
the performer and the place in the same clip, round 3 extends it with a spoken line.
No video is rendered on a PC; every call runs in the `omni` workflow.

## Setup (owner, once)
1. Prepaid credit on the AI Studio project of the key: https://ai.studio/projects
   (video has no free tier; the estate key answered 402 "credits depleted" on 2026-11-06).
2. Repo secret `GEMINI_API_KEY` (PowerShell):
   ```
   gh secret set GEMINI_API_KEY -R ryx168/kestrel-moss-9bc3
   ```
   (paste the key at the prompt; it is never echoed).
3. Optional: `refs/motion.mp4`, a short phone clip of a person doing one clear movement
   (<= 15 s for a 10 s output, <= 3 s for an 11-15 s output), and `refs/character2.png`.

## Run
Every run is a plan + price only unless `execute=true`.
```
# round 1b, written motion, kitten still (default inputs), paid
gh workflow run omni.yml -R ryx168/kestrel-moss-9bc3 -f execute=true

# round 1a, motion transfer from refs/motion.mp4
gh workflow run omni.yml -R ryx168/kestrel-moss-9bc3 -f prompt=prompts/round1_motion_transfer.txt -f video=refs/motion.mp4 -f task=reference_to_video -f execute=true

# round 2: same motion, new performer and place (id from the round 1 meta json)
gh workflow run omni.yml -R ryx168/kestrel-moss-9bc3 -f mode=continue -f previous=<id> -f prompt=prompts/round2_new_performer.txt -f image=refs/character2.png -f task=edit -f execute=true

# round 3: extend 5 s with a Mandarin line
gh workflow run omni.yml -R ryx168/kestrel-moss-9bc3 -f mode=continue -f previous=<id> -f prompt=prompts/round3_extend_dialogue.txt -f duration=5 -f task=extend -f execute=true
```
Each run uploads `out/` as the artifact `omni-run-<n>` (meta json + mp4 + log) and, when a
clip was made, publishes Release `run-<n>` with the mp4. The interaction id for the next
round is `id` in the meta json and in the run log.

## Cost (list price, 2026-11-06)
| resolution | per second | 10 s | 15 s |
|---|---|---|---|
| 360p | $0.023 | $0.23 | $0.35 |
| 720p | $0.07 | $0.70 | $1.04 |
| 1080p | $0.104 | $1.04 | $1.56 |
| 4k | $0.209 | $2.09 | $3.14 |

## Local use
`python run_omni.py generate|continue|resume ...` with `GEMINI_API_KEY` in the environment;
`--help` lists the flags. Endpoint: `POST /v1beta/interactions` (aistudio.google.com/docs/omni).

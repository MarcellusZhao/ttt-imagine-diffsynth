# VisionReward evaluation

[VisionReward](https://github.com/THUDM/VisionReward) (THUDM, [arXiv:2412.21059](https://arxiv.org/abs/2412.21059))
is a fine-grained reward model for generated video. It asks a CogVLM2-video backbone a
fixed checklist of 29 yes/no questions about a clip — three of which interpolate the
generation prompt — maps each answer to ±1, and returns the weighted mean under a
learned weight vector. Unlike VBench it produces **one scalar per clip**, which makes it
cheap to compare arms and, because the checklist is interpretable, easy to see *where*
they differ.

This folder is a self-contained port for scoring this repo's Wan2.2-TI2V-5B arms. It
does not depend on a VisionReward checkout: the checklist and weights are vendored under
`assets/`.

## Layout

| file | what it is |
| --- | --- |
| `setup.sh` | creates the `visionreward` conda env and downloads the 25 GB checkpoint |
| `requirements.txt` | pinned deps for that env |
| `visionreward_video.py` | the scorer — model wrapper, frame sampling, checklist reduction |
| `score_videos.py` | CLI: score a directory of clips → `.jsonl` |
| `aggregate.py` | CLI: merge `.jsonl` files → per-arm summary + paired arm comparison |
| `sample-demos.sh` / `.sbatch` | generate the 100-prompt demo suite for one arm |
| `score-demos.sh` / `.sbatch` | score one arm's demo clips |
| `demos-launcher.sh` | Slurm fan-out over (arm × shard) for either stage |
| `prompts/causal_forcing_demos.txt` | the 100 [Causal-Forcing](https://github.com/thu-ml/Causal-Forcing) demo prompts |
| `assets/` | checklist + weights, vendored verbatim from upstream |

## Setup

```bash
bash eval/visionreward/setup.sh
```

Creates the `visionreward` conda env and downloads `THUDM/VisionReward-Video` to
`/work/nlp/hzhao/checkpoints/visionreward/VisionReward-Video`. Idempotent; override with
`ENV_NAME`, `MODEL_PATH`, `SKIP_ENV=1`, `SKIP_WEIGHTS=1`.

**Why a separate env.** VisionReward-Video loads via `trust_remote_code`, and its remote
modeling code passes `past_key_values` as legacy tuples and builds its own 4-D attention
masks — both removed in transformers 4.47+. The `diffsynth` env runs transformers 5.9.0,
so the model cannot load there. Sampling uses `diffsynth`; scoring uses `visionreward`.
The two `.sbatch` files activate the right one each.

## The 100-prompt evaluation

`prompts/causal_forcing_demos.txt` holds the 100 demo prompts from
[thu-ml/Causal-Forcing](https://github.com/thu-ml/Causal-Forcing/blob/main/prompts/demos.txt).
They are long (up to 775 characters) two-beat narratives — "First she … Then she …" —
which makes them a better probe for long-video drift than VBench's short prompts, and
also means they **cannot be filenames**. Hence the `custom` layout below rather than
VBench's `<prompt>-<index>.mp4`.

### 1. Sample

```bash
NUM_SHARDS=5 bash eval/visionreward/demos-launcher.sh sample
```

Four arms × 5 shards = 20 jobs, one GPU each. Arms match
`examples/wanvideo/model_inference/vbench/vbench-launcher.sh`: `base`,
`chunk-by-chunk`, `chunk-by-chunk-anchored`, `e2e-ttt-fomaml`. Geometry is pinned to the
currently-active k3/480p experiment — 480×832, 24 chunks, `fpc=53` with k=3 + first-frame
sink for the anchored arms — so the only difference between the anchored baseline and the
TTT arm is the LoRA scratchpad. Clips land in
`$SAMPLE_ROOT/<arm>/<prompt[:30]>/<name>.mp4`.

This is the expensive half: measured ~17 s/chunk at 480p on an h100 (50 steps at ~2.9
it/s) = ~7 min/video for the anchored arm, so ~12 GPU-hours per arm; the TTT arm adds
its inner loop on top. `DRY_RUN=1` prints the `sbatch` lines instead of submitting.

Point the TTT arm at a specific φ₀ with `LORA=…`. The launcher **errors** if that path is
missing rather than letting the driver fall back to a zero-init identity adapter, which
would silently turn the TTT arm into the anchored baseline. Its inner-loop flags
(`--optimizer adamw --num_gradient_steps 2 --lora_rank 128 --num_anchor_latent_frames 3
--inner_lr_init 1e-5`) mirror
[`Wan2.2-TI2V-5B-e2e-ttt-fomaml-k3-480p.yaml`](../../examples/wanvideo/model_training/configs/Wan2.2-TI2V-5B-e2e-ttt-fomaml-k3-480p.yaml)
and must be changed together with it.

### 2. Score

```bash
NUM_SHARDS=2 bash eval/visionreward/demos-launcher.sh score
```

### 3. Aggregate

```bash
SCORES=/work/nlp/hzhao/evaluations/visionreward/causal-forcing-demos/scores
python eval/visionreward/aggregate.py \
    base=$SCORES/base \
    chunk-by-chunk=$SCORES/chunk-by-chunk \
    chunk-by-chunk-anchored=$SCORES/chunk-by-chunk-anchored \
    e2e-ttt-fomaml=$SCORES/e2e-ttt-fomaml \
    --per_question --csv $SCORES/summary.csv
```

Prints per-arm mean ± sem, a **paired** comparison against the first arm (same prompt,
which removes the prompt-difficulty variance that dominates a 100-clip mean), and
optionally per-checklist-item yes-rates — the last is where arms actually separate, since
the consistency and dynamics items behave differently from the appearance ones.

## Scoring an arbitrary directory

```bash
conda activate visionreward
python eval/visionreward/score_videos.py \
    --videos_path /work/nlp/hzhao/evaluations/vbench/Wan2.2-TI2V-5B-480h-832w-60s \
    --layout vbench \
    --frame_sampling uniform \
    --output /tmp/base.jsonl --skip_existing
```

Three layouts recover the per-clip prompt:

- `vbench` — `<dir>/<prompt>-<index>.mp4`, prompt from the filename. Works on anything
  under `examples/wanvideo/model_inference/vbench/`.
- `custom` — `<dir>/<prompt[:30]>/<name>.mp4` plus `--prompt_file`, which supplies the
  full prompt. What the `custom-prompts/*` drivers write.
- `map` — `--prompt_map` JSON `{relpath: prompt}`.

## Two things that change the numbers

**Frame sampling.** Upstream's `chat` strategy takes the frame nearest each whole second
and stops at 24, so on a 60 s clip it scores **only the first 24 seconds** — precisely the
window before long-video drift shows up. `score-demos.sh` therefore defaults to
`--frame_sampling uniform`, which spreads the same 24 frames over the whole clip. The
`chat` default is kept in `score_videos.py` for comparability with published VisionReward
numbers, and warns per-run how many clips it truncated. **Scores from the two settings are
not comparable — keep the choice fixed across arms.** 24 frames is not freely raisable
either: each costs 66 tokens against the backbone's 2048-token window.

**Question set.** `assets/weight.json` has 29 weights and pairs positionally with
`VisionReward_video_qa_select.txt` (29 questions). The 64-question
`VisionReward_video_qa.txt` is vendored for reference but has no published video weight
vector, so it cannot be scored; `load_checklist` raises on a length mismatch rather than
silently misaligning questions and weights.

## Fixes relative to upstream `inference-video.py`

The upstream script is a demo, not an evaluation harness. `visionreward_video.py` keeps
its semantics and fixes:

- **Answer parsing.** Upstream compares the decoded token against lowercase `'yes'`, but
  the model emits `'Yes'`. Every answer therefore scored −1 and every video got the same
  constant `mean(-weight)`. This is the one that invalidates results, not just speed.
- **Decode once per clip**, not once per question (29 full mp4 decodes → 1).
- **Encode once per clip.** All 29 questions share one video, so the 63-layer EVA2-CLIP
  vision tower ran 29 times on identical pixels; it is now cached per clip. Largest
  single speedup here.
- **One decoding step.** Upstream calls `generate(max_new_tokens=2048)` and keeps only
  the first new token.
- **Robust fallback.** If greedy decoding leaves the yes/no vocabulary, fall back to
  comparing the Yes and No logits instead of scoring the answer as "no".
- **Resumability and sharding**, which a per-clip demo has no notion of.

`--batch_size > 1` batches the checklist into one padded forward pass — measured 10.6 s
vs. 15.5 s per clip at 8. `score_videos.py` defaults to 1; `score-demos.sh` uses 8 with
`--verify_batching`, which re-scores the job's first clip at batch 1 and aborts if the
answers differ. That guard already earned its keep: the batched path first shipped with
`expand` where `repeat` was needed, because `encode_images` returns
`(frames, tokens_per_frame, d)` rather than a batch-major tensor.

`pytorchvideo` is not a dependency: it is unmaintained and fails to import against modern
torchvision, so `install_video_transform_shims()` provides the one transform the remote
code needs (`ShortSideScale`, a faithful reimplementation) plus `NormalizeVideo` /
`CenterCropVideo` if `torchvision.transforms._transforms_video` has been removed.

# autoresearch

This is an experiment to have the LLM do deep learning chess engine research.

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar5`). The branch `autoresearch/<tag>` must not already exist — this is a fresh run.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from current main.
3. **Read the in-scope files**: The repo is small. Read these files for full context:
   - `README.md` — repository context.
   - `train.py` — training loop (modifiable). Time-budgeted to 5 minutes.
   - `model.py` — neural network architecture (modifiable). ResNet with policy + value heads.
   - `mcts.py` — Monte Carlo Tree Search (modifiable). Used during play.
   - `dataset.py` — PGN to training data conversion + PyTorch Dataset. Do not modify.
   - `run.py` — UCI protocol engine. Do not modify.
   - `elo.py` — Elo evaluation harness. Do not modify.
4. **Verify data exists**: Check that `data/report.json` exists and has the needed chunks. If not, tell the human to prepare the dataset.
5. **Verify stockfish**: Check that `stockfish` is on PATH (`which stockfish`). If not, tell the human to install it.
6. **Initialize results.tsv**: Create `results.tsv` with just the header row. The baseline will be recorded after the first run.
7. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## Experimentation

Each experiment runs on a single GPU. The training script runs for a **fixed time budget of 5 minutes** (wall clock training time, excluding startup/compilation). You launch it as: `uv run train.py`. After training, the Elo evaluation runs for around 8-12 minutes via `uv run elo.py`.

**What you CAN do:**
- Modify `train.py`, `model.py`, and `mcts.py`. Everything is fair game: model architecture, search algorithm, optimizer, hyperparameters, training loop, batch size, model size, etc.

**What you CANNOT do:**
- Modify `dataset.py`. It contains the fixed data pipeline and PyTorch Dataset.
- Modify `run.py`. It is the UCI engine interface used by the Elo evaluator.
- Modify `elo.py`. It is the ground truth evaluation harness.
- Install new packages or add dependencies. You can only use what's already in `pyproject.toml`.

**The goal is simple: get the highest Elo rating.** Since the training time budget is fixed, you don't need to worry about training time — it's always 5 minutes. Everything is fair game: change the architecture, the search algorithm, the optimizer, the hyperparameters, the batch size, the model size. The only constraint is that the code runs without crashing and finishes within the time budget.

**Elo measurement is noisy.** 30 games gives roughly +/-73 Elo uncertainty. A real +50 improvement can easily measure as -40 on a bad run. Keep this in mind — don't over-interpret small differences. The Elo evaluation takes ~5 minutes, so you can skip it for experiments where `val_loss` clearly got worse (see quick-reject below).

**VRAM** is a soft constraint. Some increase is acceptable for meaningful Elo gains, but it should not blow up dramatically.

**Simplicity criterion**: All else being equal, simpler is better. A small improvement that adds ugly complexity is not worth it. Conversely, removing something and getting equal or better results is a great outcome — that's a simplification win. When evaluating whether to keep a change, weigh the complexity cost against the improvement magnitude. A +10 Elo improvement that adds 20 lines of hacky code? Probably not worth it. A +0 Elo improvement from deleting code and simplifying? Definitely keep.

**The first run**: Your very first run should always be to establish the baseline, so you will run the training and evaluation scripts as is.

## Output format

Once `train.py` finishes, it prints a summary like this:

```
---
Epoch 2 [300s/300s]  loss=2.5646 pacc=0.301  val_loss=2.7708 val_pacc=0.259
Training finished: 2 epochs in 306.6s
peak_vram_mb: 2048.3
Saved weights to model.pt
```

Then you run `uv run elo.py` which outputs:

```
Student: minidinora (0.5s/move)
Stockfish: 0.1s/move
Games: 30
Game 1/30 [W]: Loss  vs SF 1500  Rating: 1325 (+/-248)
Game 2/30 [B]: Win   vs SF 1324  Rating: 1390 (+/-203)
...
Game 30/30 [B]: Win  vs SF 1400  Rating: 1350 (+/-73)

Final rating: 1350 (+/-110)
Score: +16 =2 -12
Games saved to elo_runs/2026-03-10-23-15.pgn
```

The final Elo rating is the main metric. You can extract it from the log file:

```
grep "^Final rating:" elo.log
```

Note that the training script is configured to always stop after 5 minutes, so depending on the computing platform, the numbers might look different. The Elo evaluation takes around 5 minutes.

## Logging results

When an experiment is done, log it to `results.tsv` (tab-separated, NOT comma-separated — commas break in descriptions).

The TSV has a header row and 5 columns:

```
commit	elo	peak_vram_mb	status	description
```

1. git commit hash (short, 7 chars)
2. Elo rating achieved (e.g. 1350) — use 0 for crashes
3. peak VRAM in MB, round to .0f (e.g. 2048) — use 0 for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what this experiment tried

Example:

```
commit	elo	peak_vram_mb	status	description
a1b2c3d	1206	2048	keep	baseline
b2c3d4e	1350	2100	keep	increase filters to 256
c3d4e5f	1180	2048	discard	switch to GeLU activation
d4e5f6g	0	0	crash	double model width (OOM)
```

## The experiment loop

The experiment runs on a dedicated branch (e.g. `autoresearch/mar5`).

LOOP FOREVER:

1. Look at the git state: the current branch/commit we're on
2. Modify `train.py`, `model.py`, and/or `mcts.py` with an experimental idea by directly hacking the code.
3. git commit
4. Run training: `uv run train.py > run.log 2>&1` (redirect everything — do NOT use tee or let output flood your context)
5. Check training succeeded: `grep "^peak_vram_mb:" run.log`. If the grep output is empty, the training crashed. Run `tail -n 50 run.log` to read the stack trace and attempt a fix.
6. **Quick-reject check**: Compare `val_loss` and `val_pacc` from the last epoch line against the baseline. If val_loss is clearly worse (e.g. significantly higher than baseline), skip the Elo evaluation — discard immediately and save ~10 minutes. Only proceed to Elo eval if training metrics look promising or ambiguous.
7. Run Elo evaluation: `uv run elo.py > elo.log 2>&1`
8. Read the results: `grep "^Final rating:" elo.log`
9. Record the results in the TSV (NOTE: do not commit the results.tsv file, leave it untracked by git)
10. If Elo improved (higher), you "advance" the branch, keeping the git commit
11. If Elo is equal or worse, you git reset back to where you started

The idea is that you are a completely autonomous researcher trying things out. If they work, keep. If they don't, discard. And you're advancing the branch so that you can iterate. If you feel like you're getting stuck in some way, you can rewind, but you should probably do this very sparingly (if ever).

**Timeout**: Each experiment takes ~5 minutes for training + ~5 minutes for Elo evaluation, so roughly 10 minutes total. If training exceeds 10 minutes, kill it and treat it as a failure (discard and revert).

**Crashes**: If a run crashes (OOM, or a bug, etc.), use your judgment: If it's something easy to fix (e.g. a typo, a missing import), fix it and re-run. If the idea itself is fundamentally broken, just skip it, log "crash" as the status in the TSV, and move on.

**NEVER STOP**: Once the experiment loop has begun (after the initial setup), do NOT pause to ask the human if you should continue. Do NOT ask "should I keep going?" or "is this a good stopping point?". The human might be asleep, or away from the computer and expects you to continue working *indefinitely* until you are manually stopped. You are autonomous. If you run out of ideas, think harder — read deep learning and chess engine papers for inspiration, re-read the in-scope files for new angles, try combining previous near-misses, try more radical architectural changes. The loop runs until the human interrupts you, period.

As an example use case, a user might leave you running while they sleep. If each experiment takes ~10 minutes then you can run roughly 6 per hour, for a total of about 48 over the duration of the average human sleep. The user then wakes up to experimental results, all completed by you while they slept!

# minidinora

Minimalistic AlphaZero chess engine. Stripped-down version of [dinora](https://github.com/Saegl/dinora).

## Structure

```
minidinora/
  model.py    - AlphaNet (residual CNN with policy + value heads)
  mcts.py     - Monte Carlo Tree Search
  dataset.py  - PGN to training data conversion + PyTorch Dataset
  train.py    - Training loop
  run.py      - UCI protocol engine
  elo.py      - Elo evaluation
```

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA
- [Stockfish](https://stockfishchess.org/download/) binary on PATH (used by `elo.py`)

## Usage

Install dependencies with `uv`

```bash
uv sync
```

### Download PGN dataset

Put `.pgn` files into `pgns/`  
For example [Leela Standard Dataset](https://lczero.org/blog/2018/09/a-standard-dataset/)
Download [ccrl-pgn.tar.bz2](http://storage.lczero.org/files/ccrl-pgn.tar.bz2) in your browser, then extract:

```bash
tar -xjf ccrl-pgn.tar.bz2 -C pgns/
```

Or [Lichess Elite Database](https://database.nikonoel.fr/)

### Create dataset from PGN files

```bash
python dataset.py
```

### Train

```bash
python train.py
```

### Calculate Elo rating

```bash
python elo.py
```

### Run UCI engine

```bash
python run.py
```

Then use standard UCI commands (`uci`, `isready`, `position`, `go`, `quit`).
Or install UCI-compatible GUI like [cutechess](https://github.com/cutechess/cutechess)

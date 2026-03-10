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
```

## Usage

### Create dataset from PGN files

Put `.pgn` files into `pgns/`, then:

```bash
python dataset.py
```

### Train

```bash
python train.py
```

### Run UCI engine

```bash
python run.py
```

Then use standard UCI commands (`uci`, `isready`, `position`, `go`, `quit`).

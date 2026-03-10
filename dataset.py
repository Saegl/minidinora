"""
Convert PGN files to training data (npz) and provide a PyTorch Dataset.

Usage:
    python dataset.py
"""

import json
import pathlib

import chess
import chess.pgn
import numpy as np
from torch.utils.data import Dataset

from model import MAX_HALFMOVES, policy_index

# ---------------------------------------------------------------------------
# Compact board state (11 uint64 values) for efficient storage
# ---------------------------------------------------------------------------


def _board_to_compact(board, flip):
    pieces = np.array(
        [
            chess.flip_vertical(board.pawns) if flip else board.pawns,
            chess.flip_vertical(board.knights) if flip else board.knights,
            chess.flip_vertical(board.bishops) if flip else board.bishops,
            chess.flip_vertical(board.rooks) if flip else board.rooks,
            chess.flip_vertical(board.queens) if flip else board.queens,
            chess.flip_vertical(board.kings) if flip else board.kings,
            chess.flip_vertical(board.occupied_co[chess.BLACK])
            if flip
            else board.occupied_co[chess.WHITE],
            chess.flip_vertical(board.occupied_co[chess.WHITE])
            if flip
            else board.occupied_co[chess.BLACK],
        ],
        dtype=np.uint64,
    )
    castling = chess.flip_vertical(board.castling_rights) if flip else board.castling_rights
    if board.has_legal_en_passant():
        file, rank = divmod(board.ep_square, 8)
        if flip:
            file = 7 - file
        ep = file * 8 + rank
    else:
        ep = 64
    config = np.array([castling, ep, board.halfmove_clock], dtype=np.uint64)
    return np.concatenate((pieces, config))


def _compact_to_tensor(array):
    tensor = np.zeros((18, 8, 8), dtype=np.float32)
    pawns, knights, bishops, rooks, queens, kings = array[0:6]
    p1, p2 = array[6], array[7]

    bitboards = np.array(
        [
            kings & p1, queens & p1, rooks & p1, bishops & p1, knights & p1, pawns & p1,
            kings & p2, queens & p2, rooks & p2, bishops & p2, knights & p2, pawns & p2,
        ],
        dtype=np.uint64,
    )
    tensor[0:12] = np.unpackbits(bitboards.view(np.uint8), bitorder="little").reshape(12, 8, 8)

    castling, ep, halfmove = int(array[-3]), int(array[-2]), array[-1]
    if castling & chess.BB_H1:
        tensor[12].fill(1.0)
    if castling & chess.BB_A1:
        tensor[13].fill(1.0)
    if castling & chess.BB_H8:
        tensor[14].fill(1.0)
    if castling & chess.BB_A8:
        tensor[15].fill(1.0)
    tensor[16].fill(halfmove / MAX_HALFMOVES)
    if ep != 64:
        tensor[17][ep // 8][ep % 8] = 1.0
    return tensor


def _z_value(game, flip):
    result = game.headers["Result"]
    if result == "1/2-1/2":
        return 0.0
    elif result == "1-0":
        return 1.0 if not flip else -1.0
    elif result == "0-1":
        return -1.0 if not flip else 1.0
    raise ValueError(f"Unknown result: {result}")


# ---------------------------------------------------------------------------
# PGN conversion
# ---------------------------------------------------------------------------


def convert_pgn(pgn_path, save_path):
    boards, policies, z_values = [], [], []

    with open(pgn_path, encoding="utf8", errors="ignore") as f:
        game = chess.pgn.read_game(f)
        while game:
            if game.headers.get("Variant", "Standard") != "Standard":
                game = chess.pgn.read_game(f)
                continue

            board = game.board()
            for move in game.mainline_moves():
                flip = not board.turn
                boards.append(_board_to_compact(board, flip))
                policies.append(policy_index(move, flip))
                z_values.append(_z_value(game, flip))
                board.push(move)

            game = chess.pgn.read_game(f)

    np.savez_compressed(
        save_path,
        boards=np.array(boards, dtype=np.uint64),
        policies=np.array(policies, dtype=np.int64),
        z_values=np.array(z_values, dtype=np.float32).reshape(-1, 1),
    )
    return len(boards)


def convert_dir(pgn_dir="pgns", save_dir="data"):
    pgn_dir = pathlib.Path(pgn_dir)
    save_dir = pathlib.Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    chunks = {}
    for path in sorted(pgn_dir.rglob("*.pgn")):
        save_path = save_dir / (path.stem + ".npz")
        count = convert_pgn(path, save_path)
        print(f"{path.name}: {count} positions")
        chunks[save_path.name] = count

    total = sum(chunks.values())
    train, val, test = {}, {}, {}
    train_n, val_n = 0, 0
    train_target = total * 0.8
    val_target = total * 0.1

    for name, n in chunks.items():
        if train_n < train_target:
            train[name] = n
            train_n += n
        elif val_n < val_target:
            val[name] = n
            val_n += n
        else:
            test[name] = n

    with open(save_dir / "report.json", "w") as f:
        json.dump({"train": train, "val": val, "test": test}, f)

    print(f"Total: {total} positions (train={train_n}, val={val_n}, test={total - train_n - val_n})")


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class ChessDataset(Dataset):
    """Lazy-loading dataset that reads npz chunks on demand."""

    def __init__(self, dataset_dir="data", split="train"):
        self.dataset_dir = pathlib.Path(dataset_dir)
        with open(self.dataset_dir / "report.json") as f:
            report = json.load(f)
        self.data = report[split]

        self.chunks_bounds = []
        self.length = 0
        for name, size in self.data.items():
            self.chunks_bounds.append((self.length, self.length + size, name))
            self.length += size

        self._left = 0
        self._right = 0

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if not (self._left <= idx < self._right):
            for left, right, name in self.chunks_bounds:
                if left <= idx < right:
                    data = np.load(self.dataset_dir / name)
                    self._left, self._right = left, right
                    n = right - left
                    perm = np.random.permutation(n)
                    self._boards = data["boards"][perm]
                    self._policies = data["policies"][perm]
                    self._z_values = data["z_values"][perm]
                    break
            else:
                raise IndexError(idx)

        i = idx - self._left
        tensor = _compact_to_tensor(self._boards[i])
        return tensor, (self._policies[i], self._z_values[i].astype(np.float32))


if __name__ == "__main__":
    convert_dir()

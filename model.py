"""
AlphaZero-style network: shared residual tower + policy head + value head.

Input:  18x8x8 board tensor (12 piece planes, 4 castling, 1 fifty-move, 1 en passant)
Policy: 1880 logits (all possible UCI moves)
Value:  scalar in [-1, 1]
"""

from itertools import chain, product

import chess
import numpy as np
import torch
import torch.nn as nn


FILTERS = 128
RES_BLOCKS = 5
VALUE_CHANNELS = 8
VALUE_LIN_CHANNELS = 32
POLICY_CHANNELS = 8

# ---------------------------------------------------------------------------
# Board encoding
# ---------------------------------------------------------------------------

MAX_HALFMOVES = 100


def _flip_vertical(bb):
    bb = ((bb >> 8) & 0x00FF_00FF_00FF_00FF) | ((bb & 0x00FF_00FF_00FF_00FF) << 8)
    bb = ((bb >> 16) & 0x0000_FFFF_0000_FFFF) | ((bb & 0x0000_FFFF_0000_FFFF) << 16)
    bb = (bb >> 32) | ((bb & 0x0000_0000_FFFF_FFFF) << 32)
    return bb


def board_to_tensor(board, flip):
    tensor = np.zeros((18, 8, 8), np.float32)

    if flip:
        p1_occ, p2_occ = board.occupied_co[chess.BLACK], board.occupied_co[chess.WHITE]
    else:
        p1_occ, p2_occ = board.occupied_co[chess.WHITE], board.occupied_co[chess.BLACK]

    bitboards = np.array(
        [
            board.kings & p1_occ,
            board.queens & p1_occ,
            board.rooks & p1_occ,
            board.bishops & p1_occ,
            board.knights & p1_occ,
            board.pawns & p1_occ,
            board.kings & p2_occ,
            board.queens & p2_occ,
            board.rooks & p2_occ,
            board.bishops & p2_occ,
            board.knights & p2_occ,
            board.pawns & p2_occ,
        ],
        dtype=np.uint64,
    )
    if flip:
        bitboards = _flip_vertical(bitboards)

    tensor[0:12] = np.unpackbits(bitboards.view(np.uint8), bitorder="little").reshape(
        12, 8, 8
    )

    if board.castling_rights & (chess.BB_H8 if flip else chess.BB_H1):
        tensor[12].fill(1.0)
    if board.castling_rights & (chess.BB_A8 if flip else chess.BB_A1):
        tensor[13].fill(1.0)
    if board.castling_rights & (chess.BB_H1 if flip else chess.BB_H8):
        tensor[14].fill(1.0)
    if board.castling_rights & (chess.BB_A1 if flip else chess.BB_A8):
        tensor[15].fill(1.0)

    tensor[16].fill(board.halfmove_clock / MAX_HALFMOVES)

    if board.has_legal_en_passant():
        file, rank = divmod(board.ep_square, 8)
        if flip:
            file = 7 - file
        tensor[17, file, rank] = 1.0

    return tensor


def boards_to_tensor(boards):
    return np.array([board_to_tensor(b, not b.turn) for b in boards])


# ---------------------------------------------------------------------------
# Policy encoding (1880 moves)
# ---------------------------------------------------------------------------

LETTERS = list("abcdefgh")
NUMBERS = list(map(str, range(1, 9)))


def _generate_uci_moves():
    moves = []
    vt = [(t, 0) for t in range(-8, 8)]
    ht = [(0, t) for t in range(-8, 8)]
    da = [(t, t) for t in range(-8, 8)]
    dd = [(t, -t) for t in range(-8, 8)]
    kt = [(-2, -1), (-1, -2), (-2, 1), (1, -2), (2, -1), (-1, 2), (2, 1), (1, 2)]

    for sr, sf in product(range(8), range(8)):
        start = LETTERS[sr] + NUMBERS[sf]
        for dr, df in chain(vt, ht, da, dd, kt):
            er, ef = sr + dr, sf + df
            if (dr, df) == (0, 0) or not (0 <= er <= 7 and 0 <= ef <= 7):
                continue
            moves.append(start + LETTERS[er] + NUMBERS[ef])

    for sr in range(8):
        start = LETTERS[sr] + "7"
        for dr, df in [(-1, 1), (0, 1), (1, 1)]:
            er, ef = sr + dr, 7
            if not (0 <= er <= 7):
                continue
            end = LETTERS[er] + NUMBERS[ef]
            for p in "qrbn":
                moves.append(start + end + p)

    return moves


def _flip_move(move):
    return "".join(str(9 - int(c)) if c.isdigit() else c for c in move)


INDEX_TO_MOVE = _generate_uci_moves()
INDEX_TO_FLIPPED_MOVE = [_flip_move(m) for m in INDEX_TO_MOVE]
MOVE_TO_INDEX = {chess.Move.from_uci(m): i for i, m in enumerate(INDEX_TO_MOVE)}
FLIPPED_MOVE_TO_INDEX = {
    chess.Move.from_uci(m): i for i, m in enumerate(INDEX_TO_FLIPPED_MOVE)
}
assert len(INDEX_TO_MOVE) == 1880


def policy_index(move, flip):
    return FLIPPED_MOVE_TO_INDEX[move] if flip else MOVE_TO_INDEX[move]


def legal_policy(raw_policy, board):
    moves = list(board.legal_moves)
    lookup = FLIPPED_MOVE_TO_INDEX if not board.turn else MOVE_TO_INDEX
    logits = np.array([float(raw_policy[lookup[m]]) for m in moves])
    exp = np.exp(logits - logits.max())
    probs = exp / exp.sum()
    return {m: float(p) for m, p in zip(moves, probs)}


# ---------------------------------------------------------------------------
# Neural network
# ---------------------------------------------------------------------------


class ResBlock(nn.Module):
    def __init__(self, filters):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(filters, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
            nn.Conv2d(filters, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.body(x) + x)


class AlphaNet(nn.Module):
    def __init__(
        self,
        filters=FILTERS,
        res_blocks=RES_BLOCKS,
        policy_channels=POLICY_CHANNELS,
        value_channels=VALUE_CHANNELS,
        value_fc_hidden=VALUE_LIN_CHANNELS,
    ):
        super().__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(18, filters, 3, padding=1, bias=False),
            nn.BatchNorm2d(filters),
            nn.ReLU(),
        )
        self.res_blocks = nn.Sequential(*(ResBlock(filters) for _ in range(res_blocks)))
        self.policy_head = nn.Sequential(
            nn.Conv2d(filters, policy_channels, 1, bias=False),
            nn.BatchNorm2d(policy_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(policy_channels * 64, 1880),
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(filters, value_channels, 1, bias=False),
            nn.BatchNorm2d(value_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(value_channels * 64, value_fc_hidden),
            nn.ReLU(),
            nn.Linear(value_fc_hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.convblock(x)
        x = self.res_blocks(x)
        return self.policy_head(x), self.value_head(x)

    @torch.no_grad()
    def evaluate(self, board):
        tensor = boards_to_tensor([board])
        raw_policy, raw_value = self(
            torch.from_numpy(tensor).to(next(self.parameters()).device)
        )
        policy = legal_policy(raw_policy.cpu().numpy()[0], board)
        value = float(raw_value.cpu().numpy()[0, 0])
        return policy, value

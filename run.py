"""
Minimal single-threaded UCI chess engine.

Usage:
    python run.py
"""

import sys
from math import cos
from time import time

import chess
import torch

from model import AlphaNet
from mcts import search

WEIGHTS = "model.pt"


def send(s):
    sys.stdout.write(s + "\n")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# Time management
# ---------------------------------------------------------------------------


def calc_movetime_ms(moves_number, time_left_ms, inc_ms, overhead_ms=600):
    moves_left = (23 * cos(moves_number / 25) + 26) / (0.01 * moves_number + 1)
    remaining = time_left_ms + moves_left * inc_ms
    return max(1, remaining / moves_left - overhead_ms)


def make_stopper(tokens, board):
    """Parse `go` tokens and return a should_stop callable."""
    params = {}
    i = 0
    while i < len(tokens):
        if tokens[i] in ("wtime", "btime", "winc", "binc", "nodes", "movetime"):
            params[tokens[i]] = int(tokens[i + 1])
            i += 2
        elif tokens[i] == "infinite":
            params["infinite"] = True
            i += 1
        else:
            i += 1

    called = [False]

    if params.get("infinite"):

        def stop():
            if not called[0]:
                called[0] = True
                return False
            return False

        return stop

    if "movetime" in params:
        deadline = time() + params["movetime"] / 1000.0

        def stop():
            if not called[0]:
                called[0] = True
                return False
            return time() > deadline

        return stop

    wtime, btime = params.get("wtime"), params.get("btime")
    if wtime is not None and btime is not None:
        t = wtime if board.turn else btime
        inc = (params.get("winc") if board.turn else params.get("binc")) or 0
        mt = calc_movetime_ms(board.fullmove_number, t, inc)
        deadline = time() + mt / 1000.0

        def stop():
            if not called[0]:
                called[0] = True
                return False
            return time() > deadline

        return stop

    if "nodes" in params:
        count = [0]
        limit = params["nodes"]

        def stop():
            count[0] += 1
            return count[0] > limit

        return stop

    # fallback: infinite
    def stop():
        if not called[0]:
            called[0] = True
            return False
        return False

    return stop


# ---------------------------------------------------------------------------
# UCI loop
# ---------------------------------------------------------------------------


def uci_loop(model):
    board = chess.Board()

    while True:
        try:
            line = input()
        except (EOFError, KeyboardInterrupt):
            break

        line = line.strip()
        if not line:
            continue

        command, *tokens = line.split()

        if command == "uci":
            send("id name minidinora")
            send("id author Saegl")
            send("uciok")

        elif command == "isready":
            send("readyok")

        elif command == "ucinewgame":
            board = chess.Board()

        elif command == "position":
            if tokens[0] == "startpos":
                board = chess.Board()
            elif tokens[0] == "fen":
                board = chess.Board(" ".join(tokens[1:7]))
            if "moves" in tokens:
                for m in tokens[tokens.index("moves") + 1 :]:
                    board.push_uci(m)

        elif command == "go":
            stopper = make_stopper(tokens, board)
            move = search(board, model.evaluate, stopper)
            send(f"bestmove {move}")

        elif command == "quit":
            break


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AlphaNet()
    model.load_state_dict(torch.load(WEIGHTS, map_location=device, weights_only=True))
    model.to(device)
    model.eval()
    send(f"info string minidinora loaded on {device}")

    uci_loop(model)


if __name__ == "__main__":
    main()

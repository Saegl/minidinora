"""
Monte Carlo Tree Search with PUCT selection and terminal solver.
"""

import math
import sys
import time


class Node:
    __slots__ = ("parent", "children", "value_sum", "visits", "prior", "move")

    def __init__(self, parent, value, prior, move):
        self.parent = parent
        self.children = {}
        self.value_sum = value
        self.visits = 1
        self.prior = prior
        self.move = move

    def q(self):
        return self.value_sum / self.visits

    def puct(self, cpuct):
        return self.q() + cpuct * math.sqrt(self.parent.visits) * self.prior / self.visits


def _select_leaf(root, board, cpuct):
    node = root
    while node.children:
        node = max(node.children.values(), key=lambda n: n.puct(cpuct))
        board.push(node.move)
    return node


def _expand(node, priors, fpu):
    for move, prior in priors.items():
        node.children[move] = Node(node, fpu, prior, move)


def _backup(leaf, board, value):
    node = leaf
    v = value
    while node.parent:
        v = -v
        node.value_sum += v
        node.visits += 1
        board.pop()
        node = node.parent
    node.value_sum += -v
    node.visits += 1


def _terminal_value(board):
    result = board.result(claim_draw=True)
    if result == "*":
        return None
    if result == "1/2-1/2":
        return 0.0
    return -1.0  # current side lost


def _most_visited(node):
    return max(node.children, key=lambda m: node.children[m].visits)


def _cp(q):
    return int(85.59 * q + 584.76 * q ** 3)


def _get_pv(root, cpuct, maxlen=15):
    node, moves = root, []
    while node.children and len(moves) < maxlen:
        node = max(node.children.values(), key=lambda n: n.puct(cpuct))
        moves.append(node.move.uci())
    return " ".join(moves), len(moves)


def _send(s):
    sys.stdout.write(s + "\n")
    sys.stdout.flush()


def search(board, evaluator, stopper, cpuct=3.0, fpu=-1.0, silent=False):
    """
    Run MCTS and return the best move.

    evaluator: callable(board) -> (priors_dict, value_float)
    stopper:   callable() -> bool  (return True to stop)
    silent:    if True, suppress UCI info output
    """
    priors, value = evaluator(board)
    root = Node(None, value, 1.0, None)
    _expand(root, priors, fpu)

    start = time.time()
    last_log = start

    while not stopper():
        leaf = _select_leaf(root, board, cpuct)
        tv = _terminal_value(board)
        if tv is not None:
            priors, value = {}, tv
        else:
            priors, value = evaluator(board)
        _expand(leaf, priors, fpu)
        _backup(leaf, board, value)

        if not silent:
            now = time.time()
            if now - last_log >= 1.0:
                last_log = now
                pv, depth = _get_pv(root, cpuct)
                elapsed_ms = int((now - start) * 1000)
                nps = int(root.visits / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0
                _send(f"info depth {depth} time {elapsed_ms} nodes {root.visits} score cp {_cp(-root.q())} nps {nps} pv {pv}")

    if not silent:
        pv, depth = _get_pv(root, cpuct)
        elapsed_ms = int((time.time() - start) * 1000)
        nps = int(root.visits / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0
        _send(f"info depth {depth} time {elapsed_ms} nodes {root.visits} score cp {_cp(-root.q())} nps {nps} pv {pv}")

    return _most_visited(root)

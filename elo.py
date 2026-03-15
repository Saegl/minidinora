"""
Evaluate engine ELO by playing games against Stockfish.
Uses Glicko2 rating system with adaptive Stockfish strength.
Engine-agnostic: communicates with the student via UCI protocol.

Requires `stockfish` binary on PATH.

Usage:
    python elo.py
"""

import datetime
import math
import multiprocessing
import os
import random

import chess
import chess.engine
import chess.pgn

STUDENT_COMMAND = ["python", "run.py"]
NUM_PROCS = 3
GAMES = 30
STUDENT_MOVETIME = 0.5  # seconds per move for student
STOCKFISH_MOVETIME = 0.01  # seconds per move for stockfish
ELO_RUNS_DIR = "elo_runs"

# Stockfish ELO clamp range
SF_MIN_ELO = 1320
SF_MAX_ELO = 3190


# ---------------------------------------------------------------------------
# Glicko2 (minimal inline implementation)
# ---------------------------------------------------------------------------

MU = 1500.0
PHI = 350.0
SIGMA = 0.06
TAU = 1.0
EPSILON = 1e-6
RATIO = 173.7178

WIN, DRAW, LOSS = 1.0, 0.5, 0.0


class Rating:
    def __init__(self, mu=MU, phi=PHI, sigma=SIGMA):
        self.mu = mu
        self.phi = phi
        self.sigma = sigma


def _glicko2_update(rating, opponent_rating, outcome):
    """Update rating after a single game."""
    mu = (rating.mu - MU) / RATIO
    phi = rating.phi / RATIO
    omu = (opponent_rating.mu - MU) / RATIO
    ophi = opponent_rating.phi / RATIO

    g = 1.0 / math.sqrt(1 + 3 * ophi**2 / math.pi**2)
    e = 1.0 / (1 + math.exp(-g * (mu - omu)))
    v = 1.0 / (g**2 * e * (1 - e))

    diff = g * (outcome - e)

    a = math.log(rating.sigma**2)
    diff_sq = (diff / v) ** 2 * v**2

    def f(x):
        tmp = phi**2 + v + math.exp(x)
        return (math.exp(x) * (diff_sq - tmp) / (2 * tmp**2)) - (x - a) / TAU**2

    b = math.log(diff_sq - phi**2 - v) if diff_sq > phi**2 + v else a - TAU
    while f(b) < 0:
        b -= TAU

    fa, fb = f(a), f(b)
    while abs(b - a) > EPSILON:
        c = a + (a - b) * fa / (fb - fa)
        fc = f(c)
        if fc * fb < 0:
            a, fa = b, fb
        else:
            fa /= 2
        b, fb = c, fc

    new_sigma = math.exp(a / 2)
    phi_star = math.sqrt(phi**2 + new_sigma**2)

    new_phi = 1.0 / math.sqrt(1 / phi_star**2 + 1 / v)
    new_mu = mu + new_phi**2 * g * (outcome - e)

    return Rating(new_mu * RATIO + MU, new_phi * RATIO, new_sigma)


# ---------------------------------------------------------------------------
# UCI engine wrappers
# ---------------------------------------------------------------------------


def open_engine(command, elo=None):
    """Open a UCI engine. If elo is set, configure Stockfish strength limit."""
    engine = chess.engine.SimpleEngine.popen_uci(command)
    if elo is not None:
        clamped = min(SF_MAX_ELO, max(SF_MIN_ELO, elo))
        engine.configure({"UCI_LimitStrength": True, "UCI_Elo": clamped})
    return engine


# ---------------------------------------------------------------------------
# Match
# ---------------------------------------------------------------------------


def play_game(student, stockfish, student_is_white, game_ind, sf_elo, student_rating):
    board = chess.Board()

    white_name = (
        student.id.get("name", "student") if student_is_white else f"Stockfish {sf_elo}"
    )
    black_name = (
        f"Stockfish {sf_elo}" if student_is_white else student.id.get("name", "student")
    )

    game = chess.pgn.Game(
        headers={
            "Event": "ELO evaluation",
            "Date": datetime.date.today().strftime("%Y.%m.%d"),
            "Round": str(game_ind),
            "White": white_name,
            "Black": black_name,
            "WhiteElo": str(int(student_rating.mu))
            if student_is_white
            else str(sf_elo),
            "BlackElo": str(sf_elo)
            if student_is_white
            else str(int(student_rating.mu)),
        }
    )
    node = game

    white_engine = student if student_is_white else stockfish
    black_engine = stockfish if student_is_white else student
    white_limit = chess.engine.Limit(
        time=STUDENT_MOVETIME if student_is_white else STOCKFISH_MOVETIME
    )
    black_limit = chess.engine.Limit(
        time=STOCKFISH_MOVETIME if student_is_white else STUDENT_MOVETIME
    )

    student_nodes = []
    sf_nodes = []
    num_moves = 0

    while not board.outcome(claim_draw=True):
        if board.turn == chess.WHITE:
            result = white_engine.play(board, white_limit, info=chess.engine.INFO_ALL)
        else:
            result = black_engine.play(board, black_limit, info=chess.engine.INFO_ALL)

        is_student_move = (board.turn == chess.WHITE) == student_is_white
        nodes = result.info.get("nodes")
        if nodes is not None:
            if is_student_move:
                student_nodes.append(nodes)
            else:
                sf_nodes.append(nodes)

        node = node.add_variation(result.move)
        board.push(result.move)
        num_moves += 1

    result_str = board.result(claim_draw=True)
    game.headers["Result"] = result_str

    if student_is_white:
        outcome = WIN if result_str == "1-0" else LOSS if result_str == "0-1" else DRAW
    else:
        outcome = WIN if result_str == "0-1" else LOSS if result_str == "1-0" else DRAW

    stats = {
        "num_moves": num_moves,
        "avg_student_nodes": sum(student_nodes) / len(student_nodes)
        if student_nodes
        else 0,
        "avg_sf_nodes": sum(sf_nodes) / len(sf_nodes) if sf_nodes else 0,
    }

    return outcome, game, stats


def worker(
    game_counter,
    lock,
    pgn_path,
    shared_mu,
    shared_phi,
    shared_sigma,
    shared_wins,
    shared_draws,
    shared_losses,
    print_counter,
):
    student = open_engine(STUDENT_COMMAND)

    while True:
        with game_counter.get_lock():
            i = game_counter.value
            if i > GAMES:
                break
            game_counter.value += 1

        with lock:
            current_mu = shared_mu.value
            current_phi = shared_phi.value
            current_sigma = shared_sigma.value
        student_rating = Rating(mu=current_mu, phi=current_phi, sigma=current_sigma)

        sf_elo = int(min(SF_MAX_ELO, max(SF_MIN_ELO, student_rating.mu)))
        stockfish = open_engine(["./stockfishbin"], elo=sf_elo)
        sf_rating = Rating(mu=sf_elo, phi=50.0)

        student_is_white = random.choice([True, False])
        outcome, game, stats = play_game(
            student, stockfish, student_is_white, i, sf_elo, student_rating
        )
        stockfish.close()

        with lock:
            current_rating = Rating(
                mu=shared_mu.value,
                phi=shared_phi.value,
                sigma=shared_sigma.value,
            )
            updated = _glicko2_update(current_rating, sf_rating, outcome)
            shared_mu.value = updated.mu
            shared_phi.value = updated.phi
            shared_sigma.value = updated.sigma

            with open(pgn_path, "a") as f:
                print(game, file=f, end="\n\n")

            if outcome == WIN:
                tag = "Win"
                shared_wins.value += 1
            elif outcome == LOSS:
                tag = "Loss"
                shared_losses.value += 1
            else:
                tag = "Draw"
                shared_draws.value += 1

            display_num = print_counter.value
            print_counter.value += 1

            color = "W" if student_is_white else "B"
            print(
                f"Game {display_num}/{GAMES} [{color}]: {tag}  "
                f"vs SF {sf_elo}  "
                f"Rating: {updated.mu:.0f} (+/-{updated.phi:.0f})  "
                f"Moves: {stats['num_moves']}  "
                f"Nodes/move: student={stats['avg_student_nodes']:.0f} sf={stats['avg_sf_nodes']:.0f}",
                flush=True,
            )

    student.close()


def main():
    student = open_engine(STUDENT_COMMAND)
    student_name = student.id.get("name", "student")
    student.close()

    print(f"Student: {student_name} ({STUDENT_MOVETIME}s/move)")
    print(f"Stockfish: {STOCKFISH_MOVETIME}s/move")
    print(f"Games: {GAMES}  Processes: {NUM_PROCS}")

    os.makedirs(ELO_RUNS_DIR, exist_ok=True)
    pgn_path = os.path.join(
        ELO_RUNS_DIR,
        datetime.datetime.now().strftime("%Y-%m-%d-%H-%M") + ".pgn",
    )
    open(pgn_path, "w").close()

    game_counter = multiprocessing.Value("i", 1)
    shared_mu = multiprocessing.Value("d", MU)
    shared_phi = multiprocessing.Value("d", PHI)
    shared_sigma = multiprocessing.Value("d", SIGMA)
    shared_wins = multiprocessing.Value("i", 0)
    shared_draws = multiprocessing.Value("i", 0)
    shared_losses = multiprocessing.Value("i", 0)
    lock = multiprocessing.Lock()

    print_counter = multiprocessing.Value("i", 1)

    procs = []
    for _ in range(NUM_PROCS):
        p = multiprocessing.Process(
            target=worker,
            args=(
                game_counter,
                lock,
                pgn_path,
                shared_mu,
                shared_phi,
                shared_sigma,
                shared_wins,
                shared_draws,
                shared_losses,
                print_counter,
            ),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    print()
    print(f"Final rating: {shared_mu.value:.0f} (+/-{shared_phi.value:.0f})")
    print(f"Score: +{shared_wins.value} ={shared_draws.value} -{shared_losses.value}")
    print(f"Games saved to {pgn_path}")


if __name__ == "__main__":
    main()

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
import random

import chess
import chess.engine
import chess.pgn

STUDENT_COMMAND = ["python", "run.py"]
GAMES = 40
STUDENT_MOVETIME = 0.5  # seconds per move for student
STOCKFISH_MOVETIME = 0.1  # seconds per move for stockfish
PGN_PATH = "elo_games.pgn"

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

    while not board.outcome(claim_draw=True):
        if board.turn == chess.WHITE:
            result = white_engine.play(board, white_limit)
        else:
            result = black_engine.play(board, black_limit)
        node = node.add_variation(result.move)
        board.push(result.move)

    result_str = board.result(claim_draw=True)
    game.headers["Result"] = result_str

    if student_is_white:
        outcome = WIN if result_str == "1-0" else LOSS if result_str == "0-1" else DRAW
    else:
        outcome = WIN if result_str == "0-1" else LOSS if result_str == "1-0" else DRAW

    return outcome, game


def main():
    student = open_engine(STUDENT_COMMAND)
    student_name = student.id.get("name", "student")
    print(f"Student: {student_name} ({STUDENT_MOVETIME}s/move)")
    print(f"Stockfish: {STOCKFISH_MOVETIME}s/move")
    print(f"Games: {GAMES}")

    student_rating = Rating()
    wins, draws, losses = 0, 0, 0

    pgn_file = open(PGN_PATH, "w")

    for i in range(1, GAMES + 1):
        sf_elo = int(min(SF_MAX_ELO, max(SF_MIN_ELO, student_rating.mu)))
        stockfish = open_engine("stockfish", elo=sf_elo)
        sf_rating = Rating(mu=sf_elo, phi=50.0)

        student_is_white = random.choice([True, False])
        outcome, game = play_game(
            student, stockfish, student_is_white, i, sf_elo, student_rating
        )
        stockfish.close()

        student_rating = _glicko2_update(student_rating, sf_rating, outcome)

        print(game, file=pgn_file, end="\n\n", flush=True)

        if outcome == WIN:
            tag, wins = "Win", wins + 1
        elif outcome == LOSS:
            tag, losses = "Loss", losses + 1
        else:
            tag, draws = "Draw", draws + 1

        color = "W" if student_is_white else "B"
        print(
            f"Game {i}/{GAMES} [{color}]: {tag}  "
            f"vs SF {sf_elo}  "
            f"Rating: {student_rating.mu:.0f} (+/-{student_rating.phi:.0f})"
        )

    student.close()
    pgn_file.close()

    print()
    print(f"Final rating: {student_rating.mu:.0f} (+/-{student_rating.phi:.0f})")
    print(f"Score: +{wins} ={draws} -{losses}")
    print(f"Games saved to {PGN_PATH}")


if __name__ == "__main__":
    main()

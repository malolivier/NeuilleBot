"""
Évalue le modèle entraîné sur une batterie de positions connues,
et compare ses scores à ceux de Stockfish (vérité terrain).

Test clé : le "Problème de la Dame" — une position où un camp a perdu
sa dame. Un bon modèle doit donner un score très négatif/positif.
"""

import torch
import chess
import numpy as np
from src.architecture.transformer import ChessBot
from src.data.encoder import ChessEncoder
from src.data.engine import StockfishEvaluator
from src.engine.search import ChessEngine

MODEL_PATH     = "models/chess_bot_best.pth"
STOCKFISH_PATH = "bin/stockfish/stockfish-ubuntu-x86-64-avx2"


def load_model(path, device):
    model = ChessBot(depth=6, embed_dim=128).to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model


def tanh_to_cp(t):
    """Inverse de tanh(cp/400) → centipawns (clampé pour éviter l'infini)."""
    t = float(np.clip(t, -0.9999, 0.9999))
    return 400 * np.arctanh(t)


def build(moves_uci):
    board = chess.Board()
    for mv in moves_uci:
        board.push(chess.Move.from_uci(mv))
    return board


# ─── Positions de test ────────────────────────────────────────────────────────
TEST_POSITIONS = [
    ("Position initiale",
     chess.Board()),

    ("Apres 1.e4",
     build(["e2e4"])),

    ("Gambit roi accepte (1.e4 e5 2.f4 exf4)",
     build(["e2e4", "e7e5", "f2f4", "e5f4"])),

    # ─── TEST CLÉ : Les noirs n'ont pas de dame ───
    (">>> TEST DAME : Noirs sans dame (blancs +Q) <<<",
     chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")),

    # ─── Variante : blancs sans dame ───
    (">>> TEST DAME : Blancs sans dame (noirs +Q) <<<",
     chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")),

    ("K+Q vs K (blancs ecrasent)",
     chess.Board("8/8/8/4k3/8/8/3QK3/8 w - - 0 1")),

    ("Mat en 1 blancs (back rank : Ra8#)",
     chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")),

    ("Finale K+P vs K (blancs gagnent)",
     chess.Board("8/5k2/8/5P2/5K2/8/8/8 w - - 0 1")),
]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositif : {device}")
    print(f"Chargement du modele : {MODEL_PATH}\n")

    model     = load_model(MODEL_PATH, device)
    encoder   = ChessEncoder()
    stockfish = StockfishEvaluator(STOCKFISH_PATH)

    col = f"{'Position':<52} {'Trait':>7} {'Modele':>20} {'Stockfish':>20} {'Diff':>7}"
    print(col)
    print("-" * len(col))

    total_diff = 0.0
    worst_diff = 0.0
    worst_name = ""

    for name, board in TEST_POSITIONS:
        tensor = encoder.board_to_tensor(board).unsqueeze(0).to(device)
        with torch.no_grad():
            model_score = model(tensor).item()

        # Eval Stockfish plus profonde pour une référence fiable
        sf_score = stockfish.evaluate(board, time_limit=0.5)

        diff = abs(model_score - sf_score)
        total_diff += diff
        if diff > worst_diff:
            worst_diff = diff
            worst_name = name

        turn      = "Blancs" if board.turn == chess.WHITE else "Noirs"
        model_cp  = tanh_to_cp(model_score)
        sf_cp     = tanh_to_cp(sf_score)
        model_str = f"{model_score:+.3f} ({model_cp:+5.0f}cp)"
        sf_str    = f"{sf_score:+.3f} ({sf_cp:+5.0f}cp)"

        print(f"{name:<52} {turn:>7} {model_str:>20} {sf_str:>20} {diff:>7.3f}")

    print("-" * len(col))
    print(f"\nErreur moyenne absolue : {total_diff / len(TEST_POSITIONS):.3f}")
    print(f"Pire erreur            : {worst_diff:.3f}  ({worst_name})")
    print("\nLegende :")
    print("  |diff| < 0.15 : tres bon")
    print("  |diff| < 0.30 : acceptable")
    print("  |diff| > 0.30 : faible")
    print("\nScores en tanh(cp/400), perspective du camp au trait, plage [-1, +1].")

    stockfish.close()

    # 1. Initialiser le moteur de recherche
    engine = ChessEngine(model, encoder, device)

    # 2. On prend la position du Mat en 1 (que le bot ratait tout à l'heure)
    board = chess.Board("6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1")

    # 3. On demande le meilleur coup avec profondeur 2
    best_move, value = engine.get_best_move(board, depth=2)

    print(f"Coup suggéré : {best_move}")
    if str(best_move) == "a1a8":
        print("✅ SUCCÈS : Le bot a trouvé le Mat !")
    else:
        print("❌ ÉCHEC : Il n'a toujours pas vu le Mat...")



if __name__ == "__main__":
    main()

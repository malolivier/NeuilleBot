import os
from src.data.dataset import stream_lichess_data

# ─── Configuration ────────────────────────────────────────────────────────────

PGN_PATH        = os.path.join("data", "raw", "lichess_db_standard_rated_2014-09.pgn.zst")
OUTPUT_PATH     = os.path.join("data", "processed", "train_data_2.h5")
STOCKFISH_PATH  = "bin/stockfish/stockfish-ubuntu-x86-64-avx2"

MIN_ELO         = 1500    # Filtre qualité : ne garder que les parties ≥ ce niveau
MAX_POSITIONS   = 1_000_000 # Nombre de positions à extraire
SAVE_EVERY      = 1_000   # Flush H5 tous les N coups (protection anti-crash)

# ─── Génération du dataset ────────────────────────────────────────────────────

if not os.path.exists(PGN_PATH):
    print(f"Erreur : fichier introuvable → {PGN_PATH}")
else:
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    stream_lichess_data(
        file_path=PGN_PATH,
        output_path=OUTPUT_PATH,
        stockfish_path=STOCKFISH_PATH,
        min_elo=MIN_ELO,
        max_positions=MAX_POSITIONS,
        save_every=SAVE_EVERY,
    )

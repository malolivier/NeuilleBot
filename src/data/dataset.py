import chess.pgn
import zstandard as zstd
import io
import os
import h5py
import numpy as np
from tqdm import tqdm
from src.data.encoder import ChessEncoder
from src.data.engine import StockfishEvaluator
import torch
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    def __init__(self, h5_path):
        # Chargement complet en RAM (≈33 MB pour 100k positions).
        # Beaucoup plus rapide que d'ouvrir le H5 à chaque __getitem__.
        with h5py.File(h5_path, 'r') as f:
            self.positions = torch.from_numpy(f['positions'][:])            # [N, 13, 8, 8]
            self.labels    = torch.from_numpy(f['labels'][:]).unsqueeze(1)  # [N, 1]
        print(f"Dataset chargé en RAM : {self.positions.shape[0]} positions "
              f"({self.positions.element_size() * self.positions.nelement() / 1e6:.1f} MB)")

    def __len__(self):
        return self.positions.shape[0]

    def __getitem__(self, idx):
        return self.positions[idx], self.labels[idx]


def stream_lichess_data(file_path, output_path, stockfish_path="/usr/games/stockfish",
                        min_elo=1500, max_positions=100000, save_every=1000):
    
    dctx = zstd.ZstdDecompressor()
    encoder = ChessEncoder()
    evaluator = StockfishEvaluator(stockfish_path)

    positions_buffer = []
    labels_buffer = []
    
    # --- LOGIQUE DE REPRISE ---
    total_saved = 0
    first_save = True
    
    if os.path.exists(output_path):
        try:
            with h5py.File(output_path, 'r') as f:
                if 'positions' in f:
                    total_saved = f['positions'].shape[0]
                    first_save = False
                    print(f"Reprise détectée : {total_saved} positions déjà présentes.")
        except Exception as e:
            print(f"Erreur lors de la lecture du fichier existant : {e}")
    
    if total_saved >= max_positions:
        print("Le dataset est déjà complet !")
        evaluator.close()
        return
    # ---------------------------

    try:
        with open(file_path, 'rb') as fh:
            with dctx.stream_reader(fh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                
                # On initialise la barre de progression à l'endroit où on s'est arrêté
                pbar = tqdm(total=max_positions, desc="Extraction + Éval Stockfish")
                pbar.update(total_saved)

                current_scan_count = 0 # Compteur de positions rencontrées dans le PGN

                while total_saved + len(positions_buffer) < max_positions:
                    game = chess.pgn.read_game(text_stream)
                    if game is None: break

                    white_elo = get_elo(game.headers, "WhiteElo")
                    black_elo = get_elo(game.headers, "BlackElo")
                    if white_elo < min_elo or black_elo < min_elo: continue

                    board = game.board()
                    for move in game.mainline_moves():
                        board.push(move)
                        current_scan_count += 1

                        # SI LA POSITION EST DÉJÀ DANS LE H5, ON PASSE
                        if current_scan_count <= total_saved:
                            continue

                        if total_saved + len(positions_buffer) >= max_positions:
                            break

                        tensor = encoder.board_to_tensor(board)
                        label = evaluator.evaluate(board)

                        positions_buffer.append(tensor.numpy())
                        labels_buffer.append(label)
                        pbar.update(1)

                        if len(positions_buffer) >= save_every:
                            _flush_to_h5(positions_buffer, labels_buffer, output_path, first_save)
                            total_saved += len(positions_buffer)
                            first_save = False
                            positions_buffer.clear()
                            labels_buffer.clear()
                            pbar.set_postfix({"sauvegardés": total_saved})

                pbar.close()

        if positions_buffer:
            _flush_to_h5(positions_buffer, labels_buffer, output_path, first_save)
            total_saved += len(positions_buffer)

    finally:
        evaluator.close()

    print(f"\nTerminé : {total_saved} positions sauvegardées dans {output_path}")


def _flush_to_h5(positions, labels, output_path, first_chunk):
    """Écrit un chunk en mémoire vers le fichier H5 (création ou extension)."""
    pos_arr = np.array(positions, dtype=np.float32)  # [N, 13, 8, 8]
    lab_arr = np.array(labels, dtype=np.float32)      # [N]

    if first_chunk:
        with h5py.File(output_path, 'w') as f:
            f.create_dataset('positions', data=pos_arr,
                             maxshape=(None, 13, 8, 8),
                             compression="gzip", chunks=(256, 13, 8, 8))
            f.create_dataset('labels', data=lab_arr,
                             maxshape=(None,),
                             compression="gzip", chunks=(256,))
    else:
        with h5py.File(output_path, 'a') as f:
            n = f['positions'].shape[0]
            k = pos_arr.shape[0]
            f['positions'].resize(n + k, axis=0)
            f['positions'][n:] = pos_arr
            f['labels'].resize(n + k, axis=0)
            f['labels'][n:] = lab_arr


def get_elo(headers, key):
    value = headers.get(key, "0")
    if value == "?" or not value:
        return 0
    try:
        return int(value)
    except ValueError:
        return 0

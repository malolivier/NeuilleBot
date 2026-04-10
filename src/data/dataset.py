import chess.pgn
import zstandard as zstd
import io
import h5py
import numpy as np
from tqdm import tqdm
from src.data.encoder import ChessEncoder

OUTPUT_PATH = "data/processed/train_data.h5"

def stream_lichess_data(file_path, min_elo=0, max_positions=100000):
    # Initialisation du décompresseur Zstandard
    dctx = zstd.ZstdDecompressor()
    # Instanciation de ChessEncoder
    encoder = ChessEncoder()
    
    with open(file_path, 'rb') as fh:
        with dctx.stream_reader(fh) as reader:
            # On utilise un wrapper pour transformer le flux binaire en flux texte
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')
            
            positions = []
            labels = [] # Le score de la position
            
            pbar = tqdm(total=max_positions, desc="Extraction des positions")
            
            while len(positions) < max_positions:
                game = chess.pgn.read_game(text_stream)
                if game is None: break
                
                # Filtre de qualité : On ne veut que le top niveau
                white_elo = get_elo(game.headers, "WhiteElo")
                black_elo = get_elo(game.headers, "BlackElo")
                
                if white_elo < min_elo or black_elo < min_elo:
                    continue

                board = game.board()
                # On parcourt les coups de la partie
                for move in game.mainline_moves():
                    board.push(move)
                    
                    # 1. Encodage du plateau 
                    tensor = encoder.board_to_tensor(board)
                    
                    # 2. On stocke
                    positions.append(tensor.numpy())
                    
                    # Pour l'instant on stocke le résultat final (1=Gagne, 0=Nulle, -1=Perd)
                    # On remplacera ça par l'éval Stockfish plus tard
                    res = game.headers.get("Result", "*")
                    val = 1.0 if res == "1-0" else (-1.0 if res == "0-1" else 0.0)
                    labels.append(val)
                    
                    pbar.update(1)
                    if len(positions) >= max_positions: break
            
            pbar.close()
            return np.array(positions), np.array(labels)
        
def get_elo(headers, key):
    value = headers.get(key, "0")
    # Si c'est un point d'interrogation ou vide, on retourne 0
    if value == "?" or not value:
        return 0
    try:
        return int(value)
    except ValueError:
        return 0

def save_to_h5(positions, labels, output_path=OUTPUT_PATH):
    # Convertir les listes en tableaux numpy
    pos_array = np.array(positions, dtype=np.float32)
    label_array = np.array(labels, dtype=np.float32)

    with h5py.File(output_path, 'w') as f:
        # Création du dataset pour les positions
        # On utilise 'gzip' pour économiser 70-80% d'espace disque
        f.create_dataset('positions', data=pos_array, compression="gzip", chunks=True)
        
        # Création du dataset pour les résultats
        f.create_dataset('labels', data=label_array, compression="gzip", chunks=True)
        
    print(f"Fichier sauvegardé avec succès dans {output_path}")
    print(f"Taille finale : {pos_array.shape[0]} positions.")
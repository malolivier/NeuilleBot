import torch 
from src.data.dataset import stream_lichess_data, save_to_h5
import os

file_path = os.path.join('data', 'raw', 'lichess_db_standard_rated_2013-01.pgn.zst')

if not os.path.exists(file_path):
    print(f"Erreur : Le fichier {file_path} est introuvable !")
else:
    positions, results = stream_lichess_data(file_path)
    print(f"Extraction terminée : {len(positions)} positions chargées.")
    save_to_h5(positions, results)


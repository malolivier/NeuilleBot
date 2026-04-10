import chess
import torch
import numpy as np

class ChessEncoder:
    def __init__(self):
        # On définit l'ordre des pièces pour les plans
        self.piece_types = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]

    def board_to_tensor(self, board: chess.Board):
        # Initialisation d'un tenseur vide (13 plans de 8x8)
        # 6 blancs + 6 noirs + 1 trait = 13
        tensor = np.zeros((13, 8, 8), dtype=np.float32)
        
        for color in [chess.WHITE, chess.BLACK]:
            color_offset = 0 if color == chess.WHITE else 6
            for i, piece_type in enumerate(self.piece_types):
                # On récupère les positions des pièces
                pieces = board.pieces(piece_type, color)
                for square in pieces:
                    row = 7 - (square // 8)
                    col = square % 8
                    tensor[color_offset + i, row, col] = 1.0

        # 13ème plan : le trait (Full 1 si blanc, Full 0 si noir)
        if board.turn == chess.WHITE:
            tensor[12, :, :] = 1.0
            
        return torch.from_numpy(tensor)


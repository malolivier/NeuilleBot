"""
ChessBot - Homemade engine powered by a Transformer + Alpha-Beta search.
"""

import sys
import os
import torch
import chess

# Ajoute le chemin du repo Chessbot/ (parent de lichess-bot/) pour pouvoir importer src/
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from lib.engine_wrapper import MinimalEngine
from src.architecture.transformer import ChessBot
from src.data.encoder import ChessEncoder
from src.engine.search import ChessEngine


# Base class for test engines (required by test_bot)
class ExampleEngine(MinimalEngine):
    """Base class that test engines can inherit from."""
    pass


class homemade(MinimalEngine):
    """
    Engine powered by a trained Transformer model + minimax alpha-beta search.
    """

    def __init__(self, commands, options, stderr, draw_or_resign, game, debug, **popen_args):
        super().__init__(commands, options, stderr, draw_or_resign, game, debug, **popen_args)

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[ChessBot] Device: {self.device}")

        # Load the trained model
        try:
            model = ChessBot(depth=6, embed_dim=128).to(self.device)
            # Chemin relatif au parent de lichess-bot (Chessbot/)
            model_path = os.path.join(parent_dir, "models", "chess_bot_best.pth")
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.eval()
            print(f"[ChessBot] Model loaded from {model_path}")
        except Exception as e:
            print(f"[ChessBot] ERROR loading model: {e}")
            raise

        # Initialize encoder and engine
        self.encoder = ChessEncoder()
        self.chess_engine = ChessEngine(model, self.encoder, self.device)
        print("[ChessBot] Ready to play!")

    def search(self, board: chess.Board, time_limit: chess.engine.Limit, ponder: bool,
               draw_offered: bool, root_moves) -> chess.engine.PlayResult:
        """
        Find the best move using minimax alpha-beta search.

        Args:
            board: Current board position
            time_limit: Time control (we use depth-based search, not time-based)
            ponder: Whether to ponder (ignored)
            draw_offered: Whether opponent offered draw (ignored)
            root_moves: Legal moves to consider (ignored, uses all legal moves)

        Returns:
            chess.engine.PlayResult with the best move
        """
        # Determine search depth based on time control.
        # IMPORTANT : depth < 3 fait blunder le bot (effet d'horizon sur les captures).
        # En correspondance, time_limit.time peut être un temps par coup court (<180s)
        # qui ferait tomber en depth=2 → on force un minimum de 3.
        if time_limit.time is not None:
            remaining_seconds = time_limit.time
            if remaining_seconds < 60:       # Bullet très rapide
                depth = 3
            elif remaining_seconds < 480:    # Blitz
                depth = 3
            else:                            # Rapid/Classical/Correspondence
                depth = 4
        else:
            depth = 4  # Correspondance illimitée : on se permet depth=4

        try:
            best_move, value = self.chess_engine.get_best_move(board, depth=depth)

            if best_move is None:
                # No legal move found (shouldn't happen if board.legal_moves exists)
                best_move = list(board.legal_moves)[0]

            return chess.engine.PlayResult(best_move, None)
        except Exception as e:
            print(f"[ChessBot] ERROR in search: {e}")
            # Fallback: return first legal move
            best_move = list(board.legal_moves)[0]
            return chess.engine.PlayResult(best_move, None)

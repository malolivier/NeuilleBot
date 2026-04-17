import chess.engine
import numpy as np

class StockfishEvaluator:
    def __init__(self, path="/usr/games/stockfish"):
        # On ouvre le moteur UNE SEULE FOIS au début
        self.engine = chess.engine.SimpleEngine.popen_uci(path)

    def evaluate(self, board, time_limit=0.01):
        info = self.engine.analyse(board, chess.engine.Limit(time=time_limit))
        score = info["score"].relative
        
        if score.is_mate():
            cp = 10000 if score.mate() > 0 else -10000
        else:
            cp = score.score()
            
        # Normalisation Tanh (Score entre -1 et 1)
        return np.tanh(cp / 400.0)

    def close(self):
        # On ferme proprement à la fin
        self.engine.quit()
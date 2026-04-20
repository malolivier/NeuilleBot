import torch
import chess

class ChessEngine:
    def __init__(self, model, encoder, device):
        self.model = model
        self.encoder = encoder
        self.device = device

    # Valeurs des pièces pour MVV/LVA (Most Valuable Victim / Least Valuable Attacker)
    PIECE_VALUES = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0,
    }

    def move_order_value(self, board, move):
        """
        Retourne un score pour ordonner les coups.
        Score plus élevé = coup plus prioritaire.

        Ordre de priorité :
        1. Checks (+500)
        2. Captures (MVV/LVA : +1000 + valeur_victime - valeur_attaquant)
        3. Promotions (+400)
        4. Quiet moves (0)
        """
        score = 0

        # 1. Check : très haute priorité
        board.push(move)
        if board.is_check():
            score += 500
        board.pop()

        # 2. Capture : MVV/LVA (Most Valuable Victim / Least Valuable Attacker)
        if board.is_capture(move):
            victim_value = self.PIECE_VALUES.get(board.piece_type_at(move.to_square), 0)
            attacker_piece = board.piece_type_at(move.from_square)
            attacker_value = self.PIECE_VALUES.get(attacker_piece, 0)
            # Score = 1000 + valeur_victime - valeur_attaquant
            # (les captures avantageuses en premier, les sacrifices en dernier)
            score += 1000 + victim_value - attacker_value

        # 3. Promotion : aussi importante
        if move.promotion:
            promotion_value = self.PIECE_VALUES.get(move.promotion, 0)
            score += 400 + promotion_value

        return score

    def get_sorted_moves(self, board):
        """Retourne les coups légaux triés par ordre de priorité."""
        moves = list(board.legal_moves)
        # Trier par ordre de priorité décroissant (meilleurs coups en premier)
        return sorted(moves, key=lambda m: self.move_order_value(board, m), reverse=True)

    def evaluate_board(self, board):
        """
        Évalue une position. Retourne TOUJOURS un score en WHITE PERSPECTIVE :
        + = blanc gagne, - = noir gagne.

        Le modèle sort en side-to-move (labels score.relative de Stockfish).
        Si c'est aux Noirs → on inverse pour obtenir la white-perspective.
        """
        if board.is_checkmate():
            return 1.0 if board.turn == chess.BLACK else -1.0

        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0

        tensor = self.encoder.board_to_tensor(board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            score = self.model(tensor).item()

        score = max(min(score, 0.99), -0.99)
        if board.turn == chess.BLACK:
            score = -score
        return score

    # Quiescence search : encore expérimental (le modèle évalue mal certaines
    # positions Black-to-move, ce qui produit des blunders quand on prolonge
    # la recherche tactique). À réactiver après un réentraînement avec encoding
    # side-to-move propre (plateau miroité quand Black joue).
    USE_QUIESCENCE = False
    QUIESCENCE_MAX_DEPTH = 8

    def get_tactical_moves(self, board):
        """Retourne uniquement les captures + promotions, triées par MVV/LVA.
        Pas de push/pop pour les échecs → plus rapide que get_sorted_moves."""
        moves = [m for m in board.legal_moves if board.is_capture(m) or m.promotion]

        def order(move):
            score = 0
            if board.is_capture(move):
                victim_value = self.PIECE_VALUES.get(board.piece_type_at(move.to_square), 0)
                attacker_value = self.PIECE_VALUES.get(board.piece_type_at(move.from_square), 0)
                score += 1000 + victim_value - attacker_value
            if move.promotion:
                score += 400 + self.PIECE_VALUES.get(move.promotion, 0)
            return score

        return sorted(moves, key=order, reverse=True)

    def quiescence(self, board, alpha, beta, maximizing_player, q_depth=0):
        """
        Quiescence search : prolonge la recherche tant qu'il y a des coups tactiques
        (captures, promotions) pour éviter l'effet d'horizon.

        Principe du 'stand pat' : si l'évaluation statique est déjà suffisante
        pour provoquer un cutoff, on peut couper sans explorer les captures
        (on suppose qu'on peut toujours "ne rien faire" et garder le stand_pat).
        """
        stand_pat = self.evaluate_board(board)

        if q_depth >= self.QUIESCENCE_MAX_DEPTH or board.is_game_over():
            return stand_pat

        if maximizing_player:
            if stand_pat >= beta:
                return stand_pat
            if stand_pat > alpha:
                alpha = stand_pat
            for move in self.get_tactical_moves(board):
                board.push(move)
                score = self.quiescence(board, alpha, beta, False, q_depth + 1)
                board.pop()
                if score > alpha:
                    alpha = score
                if alpha >= beta:
                    break
            return alpha
        else:
            if stand_pat <= alpha:
                return stand_pat
            if stand_pat < beta:
                beta = stand_pat
            for move in self.get_tactical_moves(board):
                board.push(move)
                score = self.quiescence(board, alpha, beta, True, q_depth + 1)
                board.pop()
                if score < beta:
                    beta = score
                if alpha >= beta:
                    break
            return beta

    def minimax(self, board, depth, alpha, beta, maximizing_player):
        """Alpha-Beta avec move ordering + quiescence aux feuilles."""
        if board.is_game_over():
            return self.evaluate_board(board)
        if depth == 0:
            if self.USE_QUIESCENCE:
                return self.quiescence(board, alpha, beta, maximizing_player)
            return self.evaluate_board(board)

        # Trier les coups par ordre de priorité (move ordering)
        moves = self.get_sorted_moves(board)

        if maximizing_player:
            max_eval = -float('inf')
            for move in moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, False)
                board.pop()
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break # Élagage : on peut arrêter ici (cut-off)
            return max_eval
        else:
            min_eval = float('inf')
            for move in moves:
                board.push(move)
                eval = self.minimax(board, depth - 1, alpha, beta, True)
                board.pop()
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break # Élagage
            return min_eval

    def get_best_move(self, board, depth=3):
        """
        Trouve le meilleur coup possible à une profondeur donnée (avec move ordering).

        ⚠️ IMPORTANT : pas de pruning alpha-beta à la racine !
        On DOIT explorer tous les coups pour trouver le meilleur.
        Le pruning se fait seulement dans les appels récursifs à minimax().
        """
        best_move = None
        best_value = -float('inf') if board.turn == chess.WHITE else float('inf')

        # Trier les coups au niveau racine (move ordering = plus rapide)
        moves = self.get_sorted_moves(board)

        for move in moves:
            board.push(move)
            # Appel minimax avec alpha/beta pour élagage en profondeur
            # Après board.push(move), board.turn = adversaire.
            # WHITE (True) doit maximiser, BLACK (False) doit minimiser → maximizing_player = board.turn
            board_value = self.minimax(board, depth - 1, -float('inf'), float('inf'), board.turn)
            board.pop()

            # Mettre à jour le meilleur coup trouvé (sans élagage)
            if board.turn == chess.WHITE:
                if board_value > best_value:
                    best_value = board_value
                    best_move = move
            else:
                if board_value < best_value:
                    best_value = board_value
                    best_move = move

        return best_move, best_value
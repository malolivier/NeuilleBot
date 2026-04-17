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
        # --- 1. Gérer les états terminaux (Priorité absolue) ---
        if board.is_checkmate():
            # Si c'est au tour des Noirs de jouer, les Blancs ont gagné (+1)
            # Si c'est au tour des Blancs de jouer, les Noirs ont gagné (-1)
            return 1.0 if board.turn == chess.BLACK else -1.0
        
        if board.is_stalemate() or board.is_insufficient_material():
            return 0.0 # Match nul

        # --- 2. Si la partie continue, on demande au Transformer ---
        tensor = self.encoder.board_to_tensor(board).unsqueeze(0).to(self.device)
        with torch.no_grad():
            score = self.model(tensor).item()
        
        # On "clappe" le score du modèle pour qu'il ne dépasse jamais 
        # la valeur d'un vrai Mat (0.99 max)
        return max(min(score, 0.99), -0.99)

    def minimax(self, board, depth, alpha, beta, maximizing_player):
        """L'algorithme de recherche Alpha-Beta avec move ordering."""
        if depth == 0 or board.is_game_over():
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
            board_value = self.minimax(board, depth - 1, -float('inf'), float('inf'), not board.turn)
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
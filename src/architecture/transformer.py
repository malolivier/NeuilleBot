import torch
import torch.nn as nn

class ChessPatchEmbedding(nn.Module):
    def __init__(self, in_channels=13, embed_dim=128):
        super().__init__()
        # On utilise une convolution 1x1 pour transformer chaque case
        # indépendamment en un vecteur de taille 'embed_dim'
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=1)

    def forward(self, x):
        # x shape: [batch, 13, 8, 8]
        x = self.projection(x) # shape: [batch, embed_dim, 8, 8]
        # On "aplatit" les dimensions spatiales : 8x8 -> 64
        x = x.flatten(2) # shape: [batch, embed_dim, 64]
        # On transpose pour avoir le format Transformer: [batch, sequence_length, embed_dim]
        x = x.transpose(1, 2) # shape: [batch, 64, embed_dim]
        return x

class ChessPositionalEncoding(nn.Module):
    def __init__(self, seq_len=64, embed_dim=128):
        super().__init__()
        # On crée une matrice de [1, 64, 128] remplie de petits nombres aléatoires.
        # nn.Parameter indique à PyTorch que ces nombres doivent être ajustés pendant l'entraînement.
        self.pos_embed = nn.Parameter(torch.randn(1, seq_len, embed_dim) * 0.02)

    def forward(self, x):
        # x est ton tenseur qui sort du Patch Embedding : [batch_size, 64, 128]
        # On additionne simplement la position aux données de la case
        x = x + self.pos_embed
        return x

class ChessInput(nn.Module):
    def __init__(self, in_channels=13, embed_dim=128):
        super().__init__()
        self.patch_embed = ChessPatchEmbedding(in_channels, embed_dim)
        self.pos_embed = ChessPositionalEncoding(seq_len=64, embed_dim=128)

    def forward(self, x):
        # 1. On traduit l'image en mots
        x = self.patch_embed(x)
        # 2. On colle les étiquettes de position sur les mots
        x = self.pos_embed(x)
        return x

class ChessTransformerBlock(nn.Module):
    def __init__(self, embed_dim=128, num_heads=8, ff_dim=512, dropout=0.1):
        super().__init__()
        
        # --- 1. L'Attention (La communication entre les pièces) ---
        # num_heads=8 signifie qu'il y a 8 "groupes de discussion" en parallèle
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # --- 2. Le Feed Forward (La réflexion individuelle) ---
        # Après avoir discuté, chaque case "réfléchit" à ce qu'elle vient d'apprendre
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(), # Une fonction d'activation plus moderne que ReLU
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # x est ton plateau de jeu formaté : [Batch, 64, 128]
        
        # ÉTAPE 1 : Communication (Attention)
        # On normalise les données pour éviter que les chiffres n'explosent
        norm_x = self.norm1(x)
        
        # Dans l'auto-attention, le plateau se pose des questions à lui-même
        # Donc Q, K et V sont tous 'norm_x'
        attn_output, _ = self.attention(query=norm_x, key=norm_x, value=norm_x)
        
        # Connexion résiduelle (Crucial !) : On ajoute la nouvelle info aux anciennes
        x = x + attn_output
        
        # ÉTAPE 2 : Réflexion (Feed Forward)
        norm_x = self.norm2(x)
        ffn_output = self.ffn(norm_x)
        
        # Nouvelle connexion résiduelle
        x = x + ffn_output
        
        # Le tenseur ressort avec exactement la même forme : [Batch, 64, 128]
        return x

class ChessBot(nn.Module):
    def __init__(self, depth=6, embed_dim=128, num_heads=8):
        super().__init__()
        
        # 1. Encodage d'entrée (Patch + Position)
        self.input_layer = ChessInput(in_channels=13, embed_dim=embed_dim)
        
        # 2. Le "Cerveau" : Une suite de blocs Transformer
        # On utilise nn.ModuleList pour que PyTorch suive bien toutes les couches
        self.transformer_blocks = nn.ModuleList([
            ChessTransformerBlock(embed_dim=embed_dim, num_heads=num_heads)
            for _ in range(depth)
        ])
        
        # 3. La Tête d'Évaluation (Value Head)
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Tanh() # Pour forcer le résultat entre -1 et 1
        )

    def forward(self, x):
        # Passage dans l'input layer -> [Batch, 64, 128]
        x = self.input_layer(x)
        
        # Passage dans chaque bloc Transformer
        for block in self.transformer_blocks:
            x = block(x)
            
        # x est toujours [Batch, 64, 128]. 
        # Pour l'évaluation globale, on fait la moyenne des 64 cases
        x = x.mean(dim=1) # -> [Batch, 128]
        
        # On passe dans la tête finale pour avoir le score
        evaluation = self.value_head(x) # -> [Batch, 1]
        
        return evaluation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from src.architecture.transformer import ChessBot
from src.data.dataset import ChessDataset

# ─── Configuration ────────────────────────────────────────────────────────────
# Sur Google Colab, monte ton Drive puis pointe ici :
#   from google.colab import drive
#   drive.mount('/content/drive')
#   H5_PATH    = "/content/drive/MyDrive/chessbot/train_data.h5"
#   MODELS_DIR = "/content/drive/MyDrive/chessbot/models/"

H5_PATH     = "data/processed/train_data_2.h5"
MODELS_DIR  = "models/"

BATCH_SIZE  = 512
EPOCHS      = 20
LR          = 1e-4
VAL_SPLIT   = 0.1   # 10% des données pour la validation


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositif : {device}")

    # ── Données ──────────────────────────────────────────────────────────────
    full_dataset = ChessDataset(H5_PATH)
    val_size  = int(len(full_dataset) * VAL_SPLIT)
    train_size = len(full_dataset) - val_size
    train_set, val_set = random_split(full_dataset, [train_size, val_size])

    # Données déjà en RAM → num_workers=0 est optimal (pas de fork overhead)
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    print(f"Train : {train_size} positions | Val : {val_size} positions")

    # ── Modèle ───────────────────────────────────────────────────────────────
    model = ChessBot(depth=6, embed_dim=128).to(device)

    # ── Optimiseur + Scheduler ────────────────────────────────────────────────
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.MSELoss()

    import os
    os.makedirs(MODELS_DIR, exist_ok=True)

    best_val_loss = float('inf')

    # ── Boucle d'entraînement ─────────────────────────────────────────────────
    for epoch in range(EPOCHS):
        # -- Train
        model.train()
        running_loss = 0.0
        for boards, labels in train_loader:
            boards, labels = boards.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(boards)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping : évite les explosions de gradient
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)

        # -- Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for boards, labels in val_loader:
                boards, labels = boards.to(device), labels.to(device)
                val_loss += criterion(model(boards), labels).item()
        val_loss /= len(val_loader)

        scheduler.step()

        print(f"Époque {epoch+1:02d}/{EPOCHS} | Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | LR: {scheduler.get_last_lr()[0]:.2e}")

        # Sauvegarde du meilleur modèle seulement
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), f"{MODELS_DIR}chess_bot_best.pth")
            print(f"  ✓ Meilleur modèle sauvegardé (val_loss={val_loss:.4f})")

    print(f"\nEntraînement terminé. Meilleure val_loss : {best_val_loss:.4f}")


if __name__ == "__main__":
    train()

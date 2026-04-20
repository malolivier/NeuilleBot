# NeuilleBot — Entraînement sur Google Colab

## Setup Colab (5 min)

1. Va sur https://colab.research.google.com
2. Crée un nouveau notebook
3. **Change le runtime** : `Runtime > Change runtime type > GPU (T4)`
4. Copie-colle ce code dans la première cellule :

```python
# ═══════════════════════════════════════════════════════════════
# 1. CLONE LE REPO
# ═══════════════════════════════════════════════════════════════
!git clone https://github.com/malolivier/NeuilleBot.git /content/NeuilleBot
%cd /content/NeuilleBot

# ═══════════════════════════════════════════════════════════════
# 2. MONTE GOOGLE DRIVE
# ═══════════════════════════════════════════════════════════════
from google.colab import drive
drive.mount('/content/drive')

# ═══════════════════════════════════════════════════════════════
# 3. CONFIGURE LES CHEMINS
# ═══════════════════════════════════════════════════════════════
import os
os.chdir('/content/NeuilleBot')

# Modifie ces chemins selon où tu as mis ton H5 sur Drive
H5_PATH = "/content/drive/MyDrive/chessbot/train_data.h5"  # ← À ADAPTER
MODELS_DIR = "/content/drive/MyDrive/chessbot/models/"     # ← À ADAPTER

# Crée les répertoires s'ils n'existent pas
os.makedirs(os.path.dirname(MODELS_DIR), exist_ok=True)

print(f"✓ H5 file: {H5_PATH}")
print(f"✓ Models dir: {MODELS_DIR}")
print(f"✓ H5 exists: {os.path.exists(H5_PATH)}")
```

---

## Configuration (5 min)

5. **Modifie les chemins** dans la cellule ci-dessus :
   - `H5_PATH` : chemin exact de ton fichier H5 sur Google Drive
   - `MODELS_DIR` : où sauvegarder les modèles entraînés

6. **Exécute la cellule** — tu verras un lien pour autoriser Drive

---

## Entraînement

7. Crée une **nouvelle cellule** avec :

```python
# ═══════════════════════════════════════════════════════════════
# 4. LANCE L'ENTRAÎNEMENT
# ═══════════════════════════════════════════════════════════════

# Override les paths dans le module train
import sys
sys.path.insert(0, '/content/NeuilleBot')

# Importe et modifie les config
import train as train_module
train_module.H5_PATH = H5_PATH
train_module.MODELS_DIR = MODELS_DIR

# Lance l'entraînement
train_module.train()

print("\n✅ Entraînement terminé !")
print(f"✅ Modèle sauvegardé dans : {MODELS_DIR}")
```

8. **Exécute** — l'entraînement démarre ! ☕

---

## Temps estimé

| Dataset | Temps (GPU T4) | Temps (TPU v5e) |
|---------|---|---|
| 100k positions | ~5 min | ~2 min |
| 500k positions | ~25 min | ~8 min |
| 1M positions | ~45 min | ~15 min |

---

## Après l'entraînement

Le modèle `chess_bot_best.pth` sera sur Google Drive dans `MODELS_DIR`. Tu peux :
- Le télécharger et l'utiliser localement
- Le pousser sur GitHub
- L'utiliser dans lichess-bot

---

## Troubleshooting

**"ModuleNotFoundError: No module named 'src'"**
→ Assure-toi que tu es dans `/content/NeuilleBot` (le `%cd` au début)

**"FileNotFoundError: train_data.h5"**
→ Vérifie que `H5_PATH` pointe au bon endroit sur Drive

**GPU out of memory**
→ Baisse `BATCH_SIZE` dans train.py (ou utilise TPU)

---

**C'est prêt !** Lance le notebook et dis-moi comment ça va 🚀

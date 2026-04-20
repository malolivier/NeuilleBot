# NeuilleBot

A chess bot combining a **Vision Transformer** position evaluator (trained on
Stockfish-labeled positions from real Lichess games) with an **alpha-beta
minimax search** — deployable on Lichess via the
[lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) framework.

Personal project built to explore how far a pure-eval neural network plus a
modest search can get against human players.

---

## Architecture

```
PGN (Lichess) ──► Stockfish labeling ──► H5 dataset ──► Transformer ──► Alpha-beta search ──► Lichess
```

### Position encoder ([src/data/encoder.py](src/data/encoder.py))
13-plane tensor `[13, 8, 8]` : 6 planes for white pieces, 6 for black, 1 for
side-to-move. Feeds a Vision-Transformer-style patch embedding.

### Model ([src/architecture/transformer.py](src/architecture/transformer.py))
- 1×1 conv patch embedding `13 → 128`
- Learned positional embedding (64 squares)
- 6 transformer blocks, 8 attention heads, FFN width 512
- Value head → `tanh` in `[-1, 1]`

### Labels
Each position is labeled with `tanh(stockfish_cp / 400)` where `cp` comes from
`score.relative` (side-to-move perspective). Stockfish runs at 10 ms/position
— fast enough to label 1 M positions in a few hours on CPU.

### Search ([src/engine/search.py](src/engine/search.py))
- **Alpha-beta minimax** with a white-perspective evaluation
- **Move ordering** : checks (+500), captures (MVV/LVA, +1000 + victim −
  attacker), promotions (+400), quiet moves last. Yields much tighter alpha-
  beta pruning.
- **Perspective conversion** : model outputs side-to-move, search works in
  white perspective — `evaluate_board()` flips the sign when Black is to move.
- **Adaptive depth** : bullet/blitz depth 3, rapid/classical/correspondence
  depth 4. Minimum 3 to avoid horizon-effect blunders.
- **Quiescence search** is implemented but gated behind `USE_QUIESCENCE` (see
  note in source) — it exposes weaknesses in the current model's side-to-move
  learning, which a clean retrain should fix.

---

## Repo layout

| Path | Purpose |
|---|---|
| `src/architecture/transformer.py` | Vision Transformer model |
| `src/data/encoder.py` | Board → tensor encoder |
| `src/data/engine.py` | Stockfish wrapper for labeling |
| `src/data/dataset.py` | PGN streaming + H5 storage + PyTorch Dataset |
| `src/engine/search.py` | Alpha-beta + move ordering + evaluation |
| `main.py` | Extract and label positions from a PGN into H5 |
| `train.py` | Train the Transformer on the H5 dataset |
| `evaluate.py` | Sanity-check the model against Stockfish on key positions |
| `lichess_integration/` | Custom homemade engine + config for lichess-bot |
| `COLAB_TRAINING.md` | Instructions to train on a free Colab GPU |

---

## Quick start

```bash
pip install -r requirements.txt

# 1. Extract + label training positions (needs a Stockfish binary + a Lichess PGN.zst file)
python main.py

# 2. Train the Transformer (local or Colab — see COLAB_TRAINING.md)
python train.py

# 3. Sanity-check the trained model
python evaluate.py

# 4. Play on Lichess (see lichess_integration/README.md)
```

---

## Status / roadmap

Current strength : plays a reasonable opening, trades and develops sensibly
against a 1500-ELO human opponent, but still loses on tactical errors
(horizon effect, imperfect endgame play).

Next improvements, in order of expected ELO gain :

1. **Mirror-board encoding + retrain** — make side-to-move perspective clean
   so the model learns the convention reliably. Unblocks quiescence.
2. **Re-enable quiescence search** — remove the horizon effect on captures.
3. **Iterative deepening + time management** — go deeper when the clock
   allows.
4. **Transposition table (Zobrist hashing)** — free depth from repeated
   positions.
5. **Opening book** (Polyglot) — avoid the early moves the net plays poorly.

---

## Credits

- [python-chess](https://github.com/niklasf/python-chess) for chess primitives
- [Stockfish](https://stockfishchess.org/) for label generation
- [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) for the
  Lichess API wrapper
- Training data : [Lichess open database](https://database.lichess.org/)

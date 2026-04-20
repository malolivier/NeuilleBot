# Lichess Integration

Files to plug NeuilleBot into the [lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) framework and play on Lichess.

## Setup

1. Clone lichess-bot **inside** this repo root:
   ```bash
   git clone https://github.com/lichess-bot-devs/lichess-bot.git
   cd lichess-bot && pip install -r requirements.txt && cd ..
   ```

2. Copy the custom engine and config:
   ```bash
   cp lichess_integration/homemade.py         lichess-bot/homemade.py
   cp lichess_integration/config.yml.example  lichess-bot/config.yml
   ```

3. Edit `lichess-bot/config.yml` and replace `YOUR_LICHESS_OAUTH_TOKEN_HERE` with your
   Lichess OAuth token (see [lichess-bot setup docs](https://github.com/lichess-bot-devs/lichess-bot/wiki/How-to-Install) — your bot account must be upgraded to `BOT` status via `/api/bot/account/upgrade`).

4. Make sure a trained model is at `models/chess_bot_best.pth` (relative to this repo root).

5. Run:
   ```bash
   cd lichess-bot && python lichess-bot.py
   ```

## How it works

`homemade.py` defines the `homemade` engine class, which:
- Loads the trained Transformer from `../models/chess_bot_best.pth`
- Picks search depth based on time control (min 3 to avoid horizon-effect blunders)
- Delegates move selection to `ChessEngine.get_best_move()` (alpha-beta + move ordering)

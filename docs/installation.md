# Installation and configuration

## Freeciv server

CivHarness drives a headless freeciv-server 3.2.5. The install script downloads the pinned release, checks its
SHA-256 and builds it into `$HOME/opt/freeciv-3.2.5` (override with `PREFIX=...`):

```bash
bash scripts/install_freeciv_server.sh
export CIVHARNESS_SERVER=$HOME/opt/freeciv-3.2.5/bin/freeciv-server
```

Othello and Catan run in-process and need no server.

## Python package

The repository installs one distribution, `civmarshmallow`, with two packages: `civharness` (the harness) and
`civmarsh` (everything built on it).

With [uv](https://docs.astral.sh/uv/). `uv.toml` restricts uv to its own managed Python builds, so the system
Python is never used:

```bash
uv sync --extra civtelescope --extra games
uv run python -c "import civharness, civmarsh"
```

With pip (Python 3.10 or newer):

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[civtelescope,games]"
pip install --group dev        # pytest and ruff (pip >= 25.1)
```

| Extra | Installs | Needed for |
|---|---|---|
| `civtelescope` | PyTorch, Transformers | training and scoring CivTelescope and the value head |
| `games` | python-chess, OpenSpiel, Catanatron | cross-game trap banks, Catan under `civharness.games` |

`uv sync` also installs the `dev` group (pytest, ruff). If the default PyTorch wheel does not match your
accelerator, install the matching PyTorch build first.

## slime (RL training only)

Policy training runs on [slime](https://github.com/THUDM/slime) at a pinned commit with a small patch. Run the
setup inside the slime Docker image, which ships Megatron-LM, sglang and the CUDA stack:

```bash
bash scripts/slime/setup_slime.sh third_party/slime
```

## Environment variables

| Variable | Used by | Content |
|---|---|---|
| `CIVHARNESS_SERVER` | everything that plays Freeciv | freeciv-server binary (default `$HOME/opt/freeciv-3.2.5/bin/freeciv-server`) |
| `CIVMARSH_API_BASE`, `CIVMARSH_API_KEY` | zero-shot baselines, preference labels | OpenAI-compatible endpoint (e.g. `https://api.openai.com/v1`) and key |
| `STOCKFISH_BIN` | chess trap bank | Stockfish binary |
| `CIVMARSH_RULESET_DIR` | action menu | Freeciv ruleset directory (default: `civ2civ3` of the server install) |
| `SLIME_DIR`, `MEGATRON_DIR`, `BASE_MODEL`, `DATA_ROOT`, `CIVTELESCOPE_HF` | RL training and evaluation | patched slime, Megatron-LM, Qwen3-8B weights, start banks and reference sets, merged CivTelescope model; see the header of `scripts/train_policy.sh` |
| `WANDB_PROJECT`, `WANDB_API_KEY` | RL training | optional; logging is skipped without them |

Scripts write their data and results under the paths you pass. The examples in these docs use `data/`,
`results/`, `runs/`, `ckpts/` and `models/`, all git-ignored.

## Hardware

| Workload | Resources |
|---|---|
| Freeciv position banks | CPU cores; one game server per worker |
| Othello and Catan banks, trap rates | CPU |
| CivTelescope training and scoring | one GPU with 80 GB |
| One RL phase (Freeciv) | one node with eight 80 GB GPUs; the CivTelescope server shares four of them |
| Policy evaluation | one GPU per checkpoint |

## Check the install

The harness acceptance suite checks replica identity, deterministic re-seeded branching, faithful client
actions, restricted observation and action spaces, and complete game replay. Tests that need the server are
skipped when `CIVHARNESS_SERVER` does not point to a binary:

```bash
uv run pytest tests/civharness/test_determinism.py    # byte-identical replay from a mid-game save
uv run pytest                                         # everything
```

Format and lint with ruff: `ruff format .` and `ruff check .`

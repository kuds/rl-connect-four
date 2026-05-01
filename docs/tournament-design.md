# Tournament Eval — Design Doc

Status: proposed
Branch: `claude/review-tournament-setup-tGXlG`

## Goal

Compare multiple RL algorithms (PPO, DQN, AlphaZero) plus a uniform-random
baseline on OpenSpiel's `connect_four` via a seeded round-robin, and
report a win-rate matrix, per-pairing 95% Wilson CIs, and an Elo
leaderboard. Each algorithm trains in its own notebook; a single
tournament notebook discovers the trained agents and produces the eval.

## Non-goals

- Inter-snapshot play in v1 (e.g., `ppo_main_v0` vs `dqn_main_v1`). Each
  training notebook exports one canonical agent. Adding snapshots later
  is a folder-globbing change in the tournament notebook only.
- Production-grade AlphaZero. The AZ notebook ships a demo-grade config
  that finishes on a Colab L4 in ~10–15 min.
- New training infrastructure. PPO/DQN reuse RLlib's `SelfPlayCallback`;
  AZ uses OpenSpiel's reference implementation.
- A standalone Python package. The repo stays notebook-first.

## High-level architecture

```
[Connect Four] Self Play.ipynb        ──► artifacts/agents/ppo/
[Connect Four] DQN Self Play.ipynb    ──► artifacts/agents/dqn/         ──┐
[Connect Four] AlphaZero.ipynb        ──► artifacts/agents/alphazero/   ──┴──► [Connect Four] Tournament.ipynb ──► artifacts/tournament/
                                                                          │
                                              (uniform-random baseline) ──┘
```

Training notebooks are independent. The tournament notebook globs
`agents/*/` and silently skips any agent that hasn't been trained yet,
so partial runs are fine.

## Artifact layout

`$ARTIFACT_DIR` is `/content/drive/MyDrive/rl-connect-four/` in Colab and
`./artifacts/` otherwise — the existing behavior in the PPO notebook,
preserved across all four notebooks.

```
$ARTIFACT_DIR/
  agents/
    ppo/
      checkpoint/         # full Ray Algorithm checkpoint (used to load)
      rl_module/          # standalone main RLModule (lighter to load)
      adapter.py          # exposes load_agent(agent_dir, env_name) -> act
      meta.json           # algo, env, training metadata, wall clock, etc.
      learning_curves.png
      win_rate.png
      reward.png
      videos/*.gif        # vs random + vs first league snapshot
      results.json        # per-algo eval (vs random + vs own snapshots)
    dqn/                  # same shape as ppo/
    alphazero/
      checkpoint/         # OpenSpiel AZ net checkpoint
      adapter.py
      meta.json
      learning_curves.png
      videos/*.gif
      results.json
  tournament/
    win_rate_matrix.csv
    win_rate_matrix.png   # heatmap, agents on both axes
    leaderboard.csv       # Elo-sorted, with W/L/D and Wilson CIs
    leaderboard.png       # bar chart of Elo
    pairings.json         # per-pair stats (per-side + overall + CI)
    games/*.gif           # one showcase GIF per pairing
    results.json          # full machine-readable summary
```

## Adapter contract

Each agent directory contains an `adapter.py` exposing one function:

```python
def load_agent(agent_dir: pathlib.Path, env_name: str) -> Callable:
    """Return an act(obs, legal_actions, state=None) -> int callable.

    obs:           OpenSpiel info_state for the player to move (np.float32).
    legal_actions: list[int] of legal action ids from OpenSpiel.
    state:         optional pyspiel.State for agents that need full game
                   state (AlphaZero MCTS). Net-based agents ignore it.
    """
```

The tournament notebook globs `agents/*/adapter.py`, dynamically imports
each module, and calls `load_agent`. A built-in `random` baseline is
always added to the agent set.

### Why `state` is part of the contract

The PPO/DQN adapters only need `(obs, legal_actions)`, but AlphaZero's
`MCTSBot` needs the full `pyspiel.State` to roll out simulations. The
two options were:

- **A (chosen):** Extend the contract with an optional `state` kwarg.
  Net-based adapters ignore it; AZ uses it.
- **B:** Reconstruct `pyspiel.State` from `info_state` inside the AZ
  adapter on every step. Possible but expensive and effectively
  reimplements OpenSpiel's state tracking.

Option A is a small contract change for a big simplification on the AZ
side, and costs the tournament loop one extra kwarg per call.

## Per-notebook responsibilities

### `[Connect Four] Self Play.ipynb` (PPO — existing, refactored)

- Training is unchanged: `SelfPlayCallback` + `policy_mapping_fn` against
  a `random` policy until `min_league_size` snapshots are added.
- All artifact writes get re-routed through
  `AGENT_DIR = ARTIFACT_DIR / "agents" / "ppo"`.
- New "Export agent for tournament" cell at the end of the post-training
  section:
  - Save the standalone `main` RLModule via `module.save_to_path(...)` to
    `AGENT_DIR / "rl_module"` so the adapter can load it without
    spinning up a full RLlib `Algorithm`.
  - Write `adapter.py` (RLlib variant — see below).
  - Write `meta.json` capturing algo, env, training_iterations,
    total_env_steps, league_size, wall_clock_seconds, in-training
    win-rate from `results.json`.

The RLlib adapter handles both PPO and DQN by dispatching on the
`forward_inference` output keys:

- `action_dist_inputs` present → treat as logits, mask illegal moves,
  argmax. (PPO and other policy-gradient algos.)
- `q_values` present → mask illegal moves on the Q-vector, argmax. (DQN.)
- Only `actions` present → return the greedy action; if illegal, fall
  back to a uniform-random legal action (rare).

### `[Connect Four] DQN Self Play.ipynb` (new)

A near-copy of the PPO notebook with `args.algo = "DQN"` and
DQN-specific knobs (no `num_epochs`; smaller replay buffer + faster
target-net updates so it converges in a Colab session). The same
`SelfPlayCallback` pattern works because the callback is
algorithm-agnostic — it only inspects the `win_rate` env-runner metric.

The export cell is identical to PPO's; the adapter dispatches on
`forward_inference` output keys so a single adapter.py template covers
both algos.

### `[Connect Four] AlphaZero.ipynb` (new)

OpenSpiel ships a reference AlphaZero (`open_spiel.python.algorithms`)
that runs MCTS-driven self-play and trains a value+policy net. RLlib's
AlphaZero is deprecated, so we use OpenSpiel's directly.

Demo-grade config targeted at ~10–15 min on Colab L4:

- Small MLP-or-shallow-resnet body (TBD when implementing).
- ~few hundred self-play games per training iteration, ~10 iterations.
- MCTS simulations during training: 50.
- MCTS simulations at eval time: configurable (smaller than training
  for faster tournament evals).

The notebook plots loss / value-loss / move-quality curves to
`AGENT_DIR / learning_curves.png` and exports the trained net checkpoint
under `AGENT_DIR / "checkpoint"`.

The AZ adapter:

- Loads the trained net.
- Builds an `MCTSBot` wrapping `(net + UCT)` at eval time.
- `act(obs, legal_actions, state)` calls `bot.step(state)`.

### `[Connect Four] Tournament.ipynb` (new)

1. Mounts the same `ARTIFACT_DIR`.
2. Globs `ARTIFACT_DIR/agents/*/adapter.py`. For each, imports the
   module, calls `load_agent(agent_dir, "connect_four")`, and adds the
   resulting callable to the agent registry under the directory name.
3. Adds a built-in `random` baseline.
4. Runs the round-robin (see "Tournament mechanics" below).
5. Writes the win-rate matrix (PNG + CSV), Wilson-CI pairings JSON,
   Elo leaderboard (CSV + PNG), showcase GIFs, and results.json.

The notebook silently skips any missing agent dir, so it works fine
when only PPO and the random baseline are available.

## Tournament mechanics

**Round-robin.** For every unordered pair `(i, j)` with `i != j`, play
`N_GAMES_PER_PAIR_PER_SIDE` games with `i` as p0 / `j` as p1 and another
`N_GAMES_PER_PAIR_PER_SIDE` games with sides swapped. Default 50/side =
**100 games per pair**, symmetric across the first-move advantage.

**Win-rate matrix.** `M[i][j]` = win rate of `i` vs `j` averaged across
both sides. `M` is roughly antisymmetric:
`M[i][j] + M[j][i] + draw_rate(i,j) = 1`. Saved as CSV plus a heatmap
PNG with annotated cells.

**Confidence intervals.** **Wilson 95% CI** for each pairing's overall
win rate. Wilson is meaningfully better than the existing
normal-approximation CI when win rates are near 0 or 1, which is exactly
where strong/weak pairings land.

**Elo.** Bradley–Terry MLE (logistic regression on pairwise outcomes;
draws weighted as 0.5 wins + 0.5 losses) anchored so the random baseline
sits at 1000. Reported alongside total W/L/D in the leaderboard.

**Reproducibility.** A single `TOURNAMENT_SEED` constant seeds numpy
and torch at the top of the tournament notebook. The random baseline
uses a seeded `numpy.random.Generator`. AlphaZero MCTS is seeded via
its bot constructor.

**Recorded games.** For each pairing the tournament records one
showcase GIF (the top-Elo agent vs the opponent), reusing the existing
PettingZoo renderer + OpenSpiel-driver pattern from the PPO notebook.

## Compute envelope

Per pair: 100 games × ~30 moves × 1 inference (or MCTS rollout) per
move. With ~4 agents the round-robin is `C(4,2) = 6` pairs = 600 games
total. Net-based agents are ~ms/move; AlphaZero with 50 eval sims is
~100ms/move. Worst-case wall clock for the tournament: a few minutes
on a Colab L4 — much shorter than any individual training run.

## Open questions / future work

- **Snapshots in the tournament.** Easy follow-up: export each
  `main_v*` from the PPO notebook to its own `agents/ppo_main_v0/` dir.
  No tournament-side change required.
- **Multiple seeds per algo.** Today each algorithm trains once.
  Averaging Elo across seeds would tighten the leaderboard but multiplies
  training time.
- **Time controls / sim sweeps.** AlphaZero strength scales with MCTS
  sim count. The v1 tournament fixes a single eval sim count; sweeping
  it would surface a strength-vs-compute curve but lengthens evals.
- **Longitudinal Elo.** `results.json` captures Elo for each run, but
  nothing aggregates Elo across runs over time.

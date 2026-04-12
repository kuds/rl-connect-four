# Multi-Agent Reinforcement Learning with OpenSpiel’s Connect 4 Using RLlib

## Results

Hardware: Google Colab L4

| Algo | In-training win rate | Training time | Total env steps | League size |
|------|----------------------|---------------|-----------------|-------------|
| PPO  | 0.97                 | 00:04:18      | 172,000         | 3           |

The "in-training win rate" is the rolling 100-episode win rate the
`SelfPlayCallback` sees while `main` plays the current league opponent —
this is the signal that triggers new snapshots being added. For a
fixed-baseline number, the notebook also evaluates `main` against a
uniform-random policy and against every league snapshot at the end of
training and writes the results to `artifacts/results.json`.

## Post-training artifacts

Running the notebook end-to-end writes everything below to
`$ARTIFACT_DIR`, which points at Google Drive when running in Colab
(`/content/drive/MyDrive/rl-connect-four/`) and falls back to a local
`./artifacts/` directory otherwise. Flip `USE_GDRIVE = False` in the
"Mount Google Drive" cell to force the local path.

- `checkpoint/` — stable copy of the final Ray checkpoint (main + every
  `main_v*` snapshot the `SelfPlayCallback` added during league play).
- `learning_curves.png` — combined 2×3 panel: win rate, episode return
  (reward), league size, policy loss, value loss, policy entropy.
- `win_rate.png` — dedicated win rate vs. training iteration, with the
  self-play threshold line overlaid.
- `reward.png` — dedicated episode-return-over-time graph.
- `videos/*.gif` — animated replays of the trained agent playing Connect
  Four, rendered via the Farama PettingZoo `connect_four_v3` environment.
  The trained policy still consumes OpenSpiel observations (that's what
  it was trained on); PettingZoo is driven in lockstep purely as the
  renderer. Includes `main_vs_random_p0.gif`, `main_vs_random_p1.gif`,
  and — if any league snapshots exist — `main_vs_main_v0.gif`.
- `results.json` — machine-readable summary: training metadata,
  win/loss/draw rates vs. the random baseline (with 95% CIs), win rates
  vs. every league snapshot, and paths to every graph / video / checkpoint.

## Training Notes
- For this experiment, Ray's rllib requires at least 3 CPUs available and 1 GPU to run the notebook
- The notebook runs headless by default. Set `args.num_episodes_human_play > 0` in the Args cell to play against the trained agent on the command line after training.
- Video recording requires `pettingzoo[classic]` (added to the pip install cell). If PettingZoo isn't available the video cell prints a message and skips gracefully — everything else still runs.

## Finding Theta Blog Posts
- [Using Multi-Agent Reinforcement Learning to play OpenSpiel's Connect 4 with Ray's RLlib](https://www.findingtheta.com/blog/using-multi-agent-reinforcement-learning-to-play-openspiels-connect-4-with-rays-rllib)

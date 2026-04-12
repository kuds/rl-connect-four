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

Running the notebook end-to-end produces the following under `./artifacts/`:

- `checkpoint/` — stable copy of the final Ray checkpoint (main + every
  `main_v*` snapshot the league callback added). Survives the Colab
  runtime once downloaded.
- `learning_curves.png` — win rate, episode return, and league size vs.
  training iteration.
- `results.json` — machine-readable summary including training metadata,
  win/loss/draw rates vs. the random baseline (with 95% CIs), and win
  rates vs. each league snapshot.

## Training Notes
- For this experiment, Ray's rllib requires at least 3 CPUs available and 1 GPU to run the notebook
- The notebook runs headless by default. Set `args.num_episodes_human_play > 0` in the Args cell to play against the trained agent on the command line after training.

## Finding Theta Blog Posts
- [Using Multi-Agent Reinforcement Learning to play OpenSpiel's Connect 4 with Ray's RLlib](https://www.findingtheta.com/blog/using-multi-agent-reinforcement-learning-to-play-openspiels-connect-4-with-rays-rllib)

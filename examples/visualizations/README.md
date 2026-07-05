# Example visualizations

Pre-rendered example outputs from `scripts/visualize_match.py`. Each image is a
row of heatmaps telling the full story of a reconstruction:

```
ground truth  →  observation mask  →  what the agent sees  →  prediction  →  hidden-cell error
```

Colors: signed values use a diverging red/blue map centered at zero; observation
masks show observed cells in **black**; cells the agent cannot see are drawn in
neutral **gray**; the error panel highlights reconstruction error on **hidden
cells only** (the entries that count toward the score).

| File | What it shows | Command |
|---|---|---|
| `solve_hybrid_bilinear.png` | Hybrid agent on a bilinear-regime instance (does well) | `solve --agent baselines/hybrid_agent.py --seed 0 --budget small` |
| `solve_lowrank_lowrank-regime.png` | Low-rank agent on a low-rank-regime instance | `solve --agent baselines/low_rank_agent.py --seed 1 --budget small` |
| `duel_hybrid_vs_lowrank.png` | Duel: both solve the *same* hidden matrix under each other's attack masks; hybrid wins | `duel --agent-a baselines/hybrid_agent.py --agent-b baselines/low_rank_agent.py --seed 0 --budget small` |
| `duel_ridge_vs_mean_medium.png` | Duel at the medium budget (96×96) | `duel --agent-a baselines/ridge_agent.py --agent-b baselines/mean_agent.py --seed 2 --budget medium` |
| `regime_gallery.png` | One ground-truth matrix per generative regime | `gallery --budget small` |

Regenerate any of these (after `pip install -e ".[viz]"`):

```bash
python scripts/visualize_match.py <command above> --output examples/visualizations/<file>.png
```

In a duel, both players solve the **same** ground-truth matrix — only the
opponent-chosen observation masks differ. Compare the two rows' error panels to
see which agent reconstructed the hidden entries better.

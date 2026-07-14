# PlanCritic

A learned trajectory evaluator for autonomous vehicles. Combines a neural critic network with physics-based pseudo-labels to score candidate trajectories on risk, comfort, and progress.

<!-- TODO: add web viewer screenshot -->
![Web viewer screenshot](docs/screenshot.png)

## Why I built this

<!-- TODO: write this section -->

## Results

| Metric | Value |
|---|---|
| AUROC | TODO |
| Spearman correlation | TODO |
| Collision rate | TODO |

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Ego State      │    │  Lane Graph      │    │  Candidate      │
│   Features       │    │  Encoder         │    │  Trajectories   │
└─────────┬───────┘    └─────────┬────────┘    └─────────┬───────┘
          │                      │                       │
          └──────────────────────┼───────────────────────┘
                                 │
                    ┌────────────▼────────────┐
                    │   TrajectoryCritic      │
                    │   (Neural Network)      │
                    └────────────┬────────────┘
                                 │
          ┌──────────────────────┼──────────────────────┐
          │                      │                      │
    ┌─────▼─────┐         ┌──────▼──────┐        ┌─────▼─────┐
    │   Risk    │         │   Comfort   │        │ Progress  │
    │   Score   │         │    Score    │        │   Score   │
    └───────────┘         └─────────────┘        └───────────┘
```

The critic is a multi-head neural network. A state encoder processes ego vehicle features, a GNN-based lane graph encoder captures road topology, and a trajectory encoder processes candidate paths. The scoring head outputs separate risk, comfort, and progress scores plus a composite.

Training uses physics-based pseudo-labels: collision checks, time-to-collision heuristics, jerk penalties, and kinematic feasibility constraints. No human annotations required.

## Quickstart

```bash
# Clone and install
git clone https://github.com/aayushimalhotra3/plancritic.git
cd plancritic
pip install -e .

# Train on WOMD data
python -m plancritic.cli.train \
  --data-path /path/to/womd \
  --dataset womd \
  --output-dir ./outputs \
  --num-epochs 50 \
  --batch-size 32

# Score trajectories
python -m plancritic.cli.score ./outputs/critic_best.pth \
  --data-path /path/to/scenarios \
  --output scores.json

# Export data for web viewer
python -m plancritic.cli.export \
  --data-path /path/to/dataset \
  --dataset womd \
  --output-dir ./web \
  --format web

# Open web viewer (static HTML, no build step)
open web/index.html
```

## Tech stack

- Python, PyTorch, PyTorch Geometric
- Waymax (optional, for closed-loop evaluation)
- Vanilla JS + HTML for the web viewer
- GitHub Actions for CI

## Limitations

- Physics checks use simplified vehicle models
- Lane graph encoding does not use full HD map features
- Closed-loop evaluation requires separate Waymax setup
- Single-agent only; no multi-agent interaction modeling

## Citation

```bibtex
@software{plancritic2025,
  title={PlanCritic: A Learned Trajectory Evaluator for Autonomous Vehicles},
  author={Aayushi Malhotra},
  year={2025},
  url={https://github.com/aayushimalhotra3/plancritic}
}
```

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

- Waymo Open Motion Dataset for trajectory data
- Argoverse 2 for lane graph structure
- Waymax for the simulation framework

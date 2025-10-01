# PlanCritic: Physics-Informed Trajectory Evaluation

A framework for evaluating autonomous vehicle trajectories by combining neural networks with physics-based constraints. Built for researchers and engineers working on motion planning and trajectory optimization.

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/plancritic.git
cd plancritic

# Install Python dependencies
pip install -e .

# Install web viewer dependencies (optional)
cd web && npm install && cd ..
```

### Basic Usage

```python
from plancritic.models.critic import TrajectoryCritic
from plancritic.data.adapters import WOMDAdapter
from plancritic.eval.physics_checks import PhysicsChecker

# Initialize components
critic = TrajectoryCritic(state_dim=32, lane_dim=64, cand_dim=64)
adapter = WOMDAdapter(data_path="path/to/womd")
physics_checker = PhysicsChecker()

# Load and evaluate trajectories
scene_data = adapter.load_scene("scene_001")
physics_score = physics_checker.evaluate_trajectory(scene_data.trajectory)
critic_score = critic.score_trajectory(scene_data)
```

### Training a Critic

```bash
# Train on WOMD dataset
python -m plancritic.cli.train \
  --data-path /path/to/womd \
  --dataset womd \
  --output-dir ./models \
  --epochs 100 \
  --batch-size 32 \
  --learning-rate 1e-4
```

### Scoring Trajectories

```bash
# Score trajectory candidates
python -m plancritic.cli.score \
  --model-path ./models/critic_best.pth \
  --data-path /path/to/scenarios \
  --output-path ./scores.json
```

### Web Visualization

```bash
# Export data for web viewer
python -m plancritic.cli.export \
  --data-path /path/to/dataset \
  --dataset womd \
  --output-dir ./web/public/data \
  --format web

# Start web viewer
cd web && npm run dev
```

## 📋 Features

### Core Capabilities

- **Neural + Physics**: Combines learned trajectory evaluation with kinematic constraints
- **Multi-Dataset**: Supports Waymo Open Motion Dataset (WOMD) and Argoverse2
- **Trajectory Scoring**: Evaluates path quality using risk, comfort, and progress metrics
- **Lane Integration**: Uses road network topology for context-aware evaluation
- **Fast Inference**: Optimized for real-time trajectory evaluation

### Advanced Features

- **Multi-Candidate**: Compare multiple trajectory options simultaneously
- **Physics Analysis**: Collision detection, comfort scoring, and feasibility checks
- **Web Visualization**: Interactive viewer for trajectory analysis and debugging
- **📈 Evaluation Metrics**: ADE, FDE, collision rates, comfort scores, and more
- **🔧 Extensible Architecture**: Modular design for easy customization and extension

## 🏗️ Architecture

### System Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │  Model Layer    │    │  Eval Layer     │
│                 │    │                 │    │                 │
│ • WOMD Adapter  │───▶│ • TrajectoryCritic │───▶│ • PhysicsChecker│
│ • Argoverse     │    │ • MultiCandidate│    │ • CriticEvaluator│
│ • Samplers      │    │ • Lane Encoders │    │ • Metrics       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   CLI Tools     │    │  Web Viewer     │    │   Notebooks     │
│                 │    │                 │    │                 │
│ • train.py      │    │ • Next.js App   │    │ • Examples      │
│ • score.py      │    │ • Deck.gl Viz   │    │ • Tutorials     │
│ • export.py     │    │ • Real-time UI  │    │ • Analysis      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Model Architecture

The trajectory critic uses a multi-modal neural network:

1. **State Encoder**: Processes vehicle states and dynamics
2. **Lane Graph Encoder**: Encodes road topology using graph neural networks
3. **Trajectory Encoder**: Processes candidate paths with attention mechanisms
4. **Physics Integration**: Incorporates kinematic constraints and safety checks
5. **Scoring Head**: Outputs trajectory quality scores

### Physics-Based Evaluation

- **Collision Detection**: Geometric intersection and safety analysis
- **Kinematic Constraints**: Acceleration and jerk feasibility checks
- **Comfort Metrics**: Smooth driving behavior evaluation
- **Time-to-Collision**: Safety-critical interaction analysis

## 📁 Project Structure

```
plancritic/
├── src/plancritic/
│   ├── models/              # Neural network models
│   │   ├── critic.py        # Main critic architectures
│   │   ├── encoders.py      # Feature encoders
│   │   └── losses.py        # Loss functions
│   ├── data/                # Data processing
│   │   ├── adapters/        # Dataset adapters
│   │   └── samplers.py      # Data sampling logic
│   ├── eval/                # Evaluation metrics
│   │   ├── physics_checks.py # Physics-based evaluation
│   │   └── metrics.py       # Performance metrics
│   ├── maps/                # Lane graph processing
│   │   └── lanegraph.py     # Lane graph representation
│   └── cli/                 # Command-line tools
│       ├── train.py         # Training script
│       ├── score.py         # Scoring script
│       └── export.py        # Data export
├── web/                     # Web visualization
│   ├── src/                 # Next.js application
│   ├── components/          # React components
│   └── public/              # Static assets
├── notebooks/               # Jupyter notebooks
├── tests/                   # Unit tests
└── docs/                    # Documentation
```

## 🔧 Configuration

### Model Configuration

```python
# config/model.yaml
model:
  state_dim: 32
  lane_dim: 64
  cand_dim: 64
  hidden_dim: 128
  dropout: 0.1
  
physics:
  collision_threshold: 2.0
  comfort_weight: 0.3
  progress_weight: 0.4
  
training:
  learning_rate: 1e-4
  batch_size: 32
  epochs: 100
  weight_decay: 1e-5
```

### Data Configuration

```python
# config/data.yaml
data:
  womd:
    data_path: "/path/to/womd"
    splits: ["train", "val", "test"]
    sequence_length: 80
    prediction_horizon: 80
    
  argoverse:
    data_path: "/path/to/argoverse2"
    splits: ["train", "val", "test"]
    sequence_length: 50
    prediction_horizon: 60
```

## 📊 Evaluation Metrics

### Trajectory Quality

- **Average Displacement Error (ADE)**: Mean distance error over trajectory
- **Final Displacement Error (FDE)**: Distance error at final timestep
- **Collision Rate**: Percentage of trajectories with collisions
- **Off-road Rate**: Percentage of trajectories leaving drivable area

### Physics-Based Metrics

- **Comfort Score**: Based on acceleration and jerk smoothness
- **Progress Score**: Efficiency of reaching destination
- **Safety Score**: Based on collision risk and time-to-collision
- **Feasibility Score**: Kinematic and dynamic constraint satisfaction

### Model Performance

- **Ranking Correlation**: Spearman correlation with ground truth rankings
- **Classification Accuracy**: Binary classification of trajectory quality
- **Calibration Error**: Reliability of predicted scores
- **Inference Speed**: Trajectories evaluated per second

## 🧪 Examples

### Training a Model

```python
from plancritic.cli.train import Trainer, TrainingConfig

config = TrainingConfig(
    data_path="/path/to/dataset",
    dataset="womd",
    output_dir="./models",
    epochs=50,
    batch_size=16,
    learning_rate=5e-4,
    physics_weight=0.3
)

trainer = Trainer(config)
trainer.train()
```

### Scoring Trajectories

```python
from plancritic.cli.score import TrajectoryScorer

scorer = TrajectoryScorer(
    model_path="./models/critic_best.pth",
    device="cuda"
)

# Score multiple trajectories
results = scorer.score_batch([
    {"trajectory": traj1, "scene_context": scene1},
    {"trajectory": traj2, "scene_context": scene2}
])

for result in results:
    print(f"Trajectory {result['id']}: Score = {result['score']:.3f}")
```

### Physics Analysis

```python
from plancritic.eval.physics_checks import PhysicsChecker

checker = PhysicsChecker()
analysis = checker.evaluate_trajectory(
    trajectory=candidate_trajectory,
    scene_context=scene_data,
    agent_states=agent_history
)

print(f"Collision Risk: {analysis.collision_risk:.2f}")
print(f"Comfort Score: {analysis.comfort_score:.2f}")
print(f"Min TTC: {analysis.ttc_analysis.min_ttc:.1f}s")
```

## 🚀 Advanced Usage

### Custom Dataset Integration

```python
from plancritic.data.adapters.base import BaseAdapter

class CustomAdapter(BaseAdapter):
    def load_scene(self, scene_id: str):
        # Implement custom data loading
        return scene_data
    
    def get_scene_ids(self, split: str):
        # Return list of scene IDs
        return scene_ids
```

### Custom Physics Constraints

```python
from plancritic.eval.physics_checks import PhysicsChecker

class CustomPhysicsChecker(PhysicsChecker):
    def check_custom_constraint(self, trajectory):
        # Implement custom physics check
        return constraint_satisfied, violation_score
```

### Real-time Integration

```python
# Integration with planning system
class PlanningSystem:
    def __init__(self):
        self.critic = TrajectoryCritic.load("model.pth")
    
    def select_trajectory(self, candidates, scene_context):
        scores = self.critic.score_candidates(candidates, scene_context)
        return candidates[scores.argmax()]
```

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/test_models.py -v
pytest tests/test_physics.py -v
pytest tests/test_data.py -v

# Run with coverage
pytest tests/ --cov=plancritic --cov-report=html
```

### Test Categories

- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow testing
- **Performance Tests**: Speed and memory benchmarks
- **Physics Tests**: Validation of physics constraints

## 📚 Documentation

### API Reference

- [Model API](docs/api/models.md) - Neural network architectures
- [Data API](docs/api/data.md) - Data loading and processing
- [Evaluation API](docs/api/eval.md) - Metrics and physics checks
- [CLI Reference](docs/cli.md) - Command-line tool usage

### Tutorials

- [Getting Started](notebooks/01_getting_started.ipynb)
- [Training Custom Models](notebooks/02_training.ipynb)
- [Physics Analysis](notebooks/03_physics_analysis.ipynb)
- [Web Visualization](notebooks/04_visualization.ipynb)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup

```bash
# Clone repository
git clone https://github.com/your-org/plancritic.git
cd plancritic

# Create development environment
conda create -n plancritic python=3.9
conda activate plancritic

# Install in development mode
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

### Code Style

- **Python**: Black formatting, isort imports, flake8 linting
- **TypeScript**: Prettier formatting, ESLint rules
- **Documentation**: Google-style docstrings

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-org/plancritic/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-org/plancritic/discussions)
- **Email**: plancritic-support@your-org.com

## 🙏 Acknowledgments

- Waymo Open Motion Dataset team for providing high-quality trajectory data
- Argoverse team for the autonomous driving dataset
- Deck.gl team for the excellent visualization framework
- The autonomous driving research community for inspiration and feedback

## 📈 Roadmap

### Version 1.0 (Current)
- ✅ Core trajectory critic implementation
- ✅ Physics-based supervision
- ✅ Multi-dataset support
- ✅ Web visualization
- ✅ CLI tools

### Version 1.1 (Planned)
- 🔄 Real-time planning integration
- 🔄 Advanced physics constraints
- 🔄 Multi-agent scenarios
- 🔄 Uncertainty quantification

### Version 2.0 (Future)
- 📋 Reinforcement learning integration
- 📋 Sim-to-real transfer
- 📋 Distributed training
- 📋 Mobile deployment

---

**Built with ❤️ for the autonomous driving community**

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Ego State     │    │  Lane Graph      │    │  Candidate      │
│   Features      │    │  Encoder         │    │  Trajectories   │
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

## Quickstart

```bash
# Create environment
conda create -n plancritic python=3.9
conda activate plancritic

# Install package
pip install -e .

# Run tiny example
python examples/quickstart.py

# Train on toy samples
python -m plancritic.cli.train --config configs/toy_config.yaml

# Score trajectories
python -m plancritic.cli.score --input examples/tiny_scene.json --output scores.json

# Launch web viewer
cd webviewer && npm install && npm run dev
```

## Core Features

- **Data Adapters**: Simple loaders for WOMD/Argoverse2 (metadata only; raw data optional)
- **LaneGraph Encoder**: GNN/attention over polyline features (centerline, speed limit, turn type)
- **Trajectory Critic**: Multi-head neural network scoring Risk, Comfort, Progress, and Composite scores
- **Self-Supervision**: Physics-based pseudo-labels (collision checks, TTC heuristics, jerk penalties)
- **Closed-Loop Validation**: Waymax integration for rollout testing
- **Web Viewer**: Interactive Next.js + Deck.gl visualization

## Training & Evaluation

### Open-Loop Metrics
- AUROC for risky vs safe trajectories
- Spearman correlation with physics scores
- Calibration metrics for score distributions

### Closed-Loop Metrics (Optional)
- Off-route percentage
- Collision rate
- Average jerk
- Route completion time

## Tech Stack

- **Python**: PyTorch + PyTorch Geometric
- **Evaluation**: Waymax (optional, for closed-loop)
- **Web**: Next.js + Tailwind + Deck.gl
- **CI**: GitHub Actions

## Limitations & Future Work

- Physics checks use simplified models for computational efficiency
- Lane graph encoding could benefit from HD map features
- Closed-loop evaluation requires additional simulation setup
- Future work: Multi-agent interactions, uncertainty quantification

## Citation

If you use PlanCritic in your research, please cite:

```bibtex
@software{plancritic2024,
  title={PlanCritic: A Learned Trajectory Evaluator for Autonomous Vehicles},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/plancritic}
}
```

## Acknowledgments

- Waymo Open Motion Dataset for trajectory data format
- Argoverse 2 for lane graph structure
- Waymax team for simulation framework
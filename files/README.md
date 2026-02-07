# Spot Quadruped Robot - Reinforcement Learning Training

Train a simulated Spot-like quadruped robot to walk towards goals using PPO (Proximal Policy Optimization).

## Prerequisites

- Python 3.8+

## Setup

Install all dependencies:

```bash
pip install -r requirements.txt
```

Verify everything is installed correctly:

```bash
python quick_start.py
```

## Train the Robot

Run the default training (5 million timesteps, 8 parallel environments):

```bash
python train_spot.py
```

Or customize the training parameters:

```bash
python train_spot.py --timesteps 10000000 --n_envs 16 --save_dir ./spot_models --log_dir ./spot_logs
```

To continue training from a previously saved checkpoint:

```bash
python train_spot.py --continue_from spot_models/best_model.zip --timesteps 2000000
```

Models are saved automatically to `./spot_models/` during training. The best-performing model is saved as `best_model.zip`.

## Monitor Training Progress

In a separate terminal, launch TensorBoard:

```bash
tensorboard --logdir spot_logs/
```

Then open http://localhost:6006 in your browser.

## Test the Trained Robot

Visualize the trained robot walking towards goals:

```bash
python test_spot.py spot_models/best_model.zip
```

Run more episodes or disable rendering for faster evaluation:

```bash
python test_spot.py spot_models/best_model.zip --episodes 10
python test_spot.py spot_models/best_model.zip --no-render
```

Run in interactive mode (continuous episodes, press Ctrl+C to stop):

```bash
python test_spot.py spot_models/best_model.zip --interactive
```

## Analyze Policy Performance

Generate performance plots and statistics from a trained model:

```bash
python analyze_policy.py spot_models/best_model.zip
```

Compare multiple model checkpoints side by side:

```bash
python analyze_policy.py spot_models/spot_model_1000000_steps.zip spot_models/best_model.zip --compare
```

## View Current Configuration

Print all training, reward, and environment parameters:

```bash
python config.py
```

Edit `config.py` to adjust reward weights, PPO hyperparameters, or environment settings before training.

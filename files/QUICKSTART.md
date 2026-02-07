# Quick Setup Guide - Spot Robot RL Training

## 🚀 Getting Started in 3 Steps

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Verify Installation
```bash
python quick_start.py
```

### Step 3: Start Training
```bash
# Basic training (recommended for first time)
python train_spot.py

# Or customize parameters
python train_spot.py --timesteps 10000000 --n_envs 16
```

## 📊 Monitor Training

Open a new terminal and run:
```bash
tensorboard --logdir spot_logs/
```

Then open http://localhost:6006 in your browser.

## 🎮 Test Your Model

After training (or during training):
```bash
# Visualize the robot
python test_spot.py spot_models/best_model.zip

# Interactive mode
python test_spot.py spot_models/best_model.zip --interactive

# Analyze performance
python analyze_policy.py spot_models/best_model.zip
```

## 📁 Project Files

| File | Purpose |
|------|---------|
| `spot_env.py` | Gymnasium environment (observation/action/reward) |
| `spot_scene.xml` | MuJoCo robot model (physics definition) |
| `train_spot.py` | Main training script (PPO algorithm) |
| `test_spot.py` | Test and visualize trained models |
| `analyze_policy.py` | Generate analysis plots and statistics |
| `config.py` | Configuration parameters (easy to modify) |
| `quick_start.py` | Verify installation and run quick tests |
| `requirements.txt` | Python dependencies |
| `README.md` | Full documentation |

## 🎯 What You're Training

The robot learns to:
1. **Walk** without falling over
2. **Navigate** to randomly placed goals (3-8 meters away)
3. **Balance** while moving
4. **Move efficiently** (minimize energy use)

## ⏱️ Training Timeline

- **First 100K steps** (~20 min): Robot learns to stand
- **500K steps** (~2 hours): Basic walking emerges
- **2M steps** (~8 hours): Stable locomotion
- **5M steps** (~20 hours): Goal-directed navigation

*Times are approximate for 8 parallel environments on modern CPU*

## 🔧 Common Issues

**Robot falls immediately**
- Train longer (try 10M steps)
- Check that all files are in same directory

**Training is slow**
- Reduce `--n_envs` if out of memory
- Ensure MuJoCo is properly installed
- Consider using GPU (automatic if available)

**Can't see training progress**
- Run TensorBoard: `tensorboard --logdir spot_logs/`
- Check `spot_logs/` directory is being created

## 🎓 Next Steps

1. **Tune parameters**: Edit `config.py` to adjust rewards, network size, etc.
2. **Add complexity**: Modify `spot_env.py` to add obstacles or rough terrain
3. **Improve robot**: Edit `spot_scene.xml` to change robot morphology
4. **Sim-to-real**: Add domain randomization for real robot deployment

## 📚 Learn More

- Read `README.md` for detailed documentation
- Check [Stable Baselines3 docs](https://stable-baselines3.readthedocs.io/)
- Explore [MuJoCo documentation](https://mujoco.readthedocs.io/)

## 💡 Tips

- Start with default settings first
- Use `quick_start.py` to verify everything works
- Monitor training with TensorBoard
- Save checkpoints frequently (automatic)
- Test early and often to see progress

---

**Need help?** Check the full README.md or the comments in each Python file.

**Happy Training! 🤖**

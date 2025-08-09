# ResNet-18 Hyperparameter Tuning with Curvy Optimizer + W&B Sweeps

This repository contains an automated hyperparameter optimization pipeline for training ResNet-18 models using the Curvy optimizer on the SVHN dataset. The script uses **Weights & Biases Sweeps** for intelligent hyperparameter search with Bayesian optimization.

## 🚀 Features

- **W&B Sweeps Integration**: Automated hyperparameter optimization with Bayesian search
- **Multi-Agent Support**: Run multiple sweep agents in parallel across different machines
- **Early Termination**: Stop poor performing runs automatically (Hyperband)
- **Comprehensive Logging**: Track all training metrics in real-time
- **Flexible Execution**: Support for sweeps, single runs, and manual configurations
- **Easy Setup**: One-command setup and execution scripts

## 🎯 Why W&B Sweeps?

W&B Sweeps provide several advantages over manual grid search:
- **Intelligent Search**: Bayesian optimization finds optimal hyperparameters faster
- **Early Termination**: Stop bad runs early to save compute time
- **Parallel Execution**: Run multiple agents simultaneously
- **Real-time Visualization**: Monitor progress and results in the W&B dashboard
- **Automatic Logging**: All experiments tracked with zero additional code

## 📦 Quick Start

### 1. Setup Environment
```bash
./run_sweeps.sh setup
```
This will:
- Install all dependencies
- Setup W&B authentication
- Create necessary directories

### 2. Create a Sweep
```bash
./run_sweeps.sh create
```
This creates a new W&B sweep and provides a sweep ID.

### 3. Run Sweep Agents
```bash
./run_sweeps.sh agent <SWEEP_ID>
```
Replace `<SWEEP_ID>` with the ID from step 2.

**For multiple agents (parallel execution):**
```bash
# On machine 1:
./run_sweeps.sh agent <SWEEP_ID> 20

# On machine 2:
./run_sweeps.sh agent <SWEEP_ID> 20

# On machine 3:
./run_sweeps.sh agent <SWEEP_ID> 20
```

### 4. Monitor Results
- Check the W&B dashboard at https://wandb.ai
- View real-time training curves and hyperparameter importance
- Compare runs and identify optimal configurations

## 🛠 Configuration

### Experiment Settings
Update `EXPERIMENT_CONFIG` in the Python script:

```python
EXPERIMENT_CONFIG = {
    'project_name': 'curvy-optimizer-sweeps',
    'entity': 'your-wandb-username',  # Update this!
    'num_epochs': 20,
    'dataset': 'SVHN',
    'model': 'ResNet-18',
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}
```

### Sweep Configuration
The sweep configuration is defined in `sweep_config.yaml`:

```yaml
method: bayes  # Bayesian optimization
metric:
  goal: maximize
  name: final_test_accuracy

parameters:
  batch_size:
    values: [64, 128]
  learning_rate:
    distribution: log_uniform_values
    min: 1e-4
    max: 1e-1
  hessian_epsilon:
    distribution: log_uniform_values
    min: 1e-4
    max: 1e-1
  momentum:
    distribution: uniform
    min: 0.0
    max: 0.99
  hessian_compute_interval:
    values: [10, 25, 50, 100]

early_terminate:
  type: hyperband
  min_iter: 5
  eta: 2
```

## 📚 Usage Examples

### Single Experiment (Testing)
```bash
./run_sweeps.sh single
```
Runs one experiment with default hyperparameters.

### Create Custom Sweep
```bash
python resnet-18-hyperparam-tune.py --mode sweep
```

### Run Specific Number of Experiments
```bash
python resnet-18-hyperparam-tune.py --mode agent --sweep_id <ID> --count 50
```

### Check W&B Status
```bash
./run_sweeps.sh status
```

## 🎮 Command Reference

### Helper Script Commands
```bash
./run_sweeps.sh setup     # Install dependencies and setup W&B
./run_sweeps.sh create    # Create a new W&B sweep
./run_sweeps.sh agent     # Run a sweep agent
./run_sweeps.sh single    # Run single experiment
./run_sweeps.sh status    # Check W&B login status
```

### Python Script Commands
```bash
# Create sweep
python resnet-18-hyperparam-tune.py --mode sweep

# Run sweep agent
python resnet-18-hyperparam-tune.py --mode agent --sweep_id <ID> --count <N>

# Single run
python resnet-18-hyperparam-tune.py --mode single
```

## 📊 Logged Metrics

**Per Epoch:**
- `train_loss`, `train_accuracy`
- `test_loss`, `test_accuracy`
- `learning_rate`, `epoch`

**Final Summary:**
- `final_train_loss/accuracy`
- `final_test_loss/accuracy`
- `best_train_accuracy`, `best_test_accuracy`
- `min_train_loss`, `min_test_loss`

## 🔧 Advanced Configuration

### Modify Search Space
Edit the sweep configuration in `create_sweep()` function:

```python
'parameters': {
    'batch_size': {'values': [32, 64, 128, 256]},
    'learning_rate': {
        'distribution': 'log_uniform_values',
        'min': 1e-5,
        'max': 1e-1
    },
    # Add new parameters here
}
```

### Change Optimization Strategy
```python
sweep_config = {
    'method': 'random',  # or 'grid', 'bayes'
    # ... rest of config
}
```

### Early Termination Settings
```python
'early_terminate': {
    'type': 'hyperband',
    'min_iter': 3,      # Minimum epochs before termination
    'eta': 3            # Termination aggressiveness
}
```

## 🚦 Multi-Agent Parallel Execution

### Local Multi-Processing
Run multiple agents on the same machine:
```bash
# Terminal 1
./run_sweeps.sh agent <SWEEP_ID> 10

# Terminal 2  
./run_sweeps.sh agent <SWEEP_ID> 10

# Terminal 3
./run_sweeps.sh agent <SWEEP_ID> 10
```

### Multi-Machine Setup
1. **Setup each machine** with the same code and dependencies
2. **Use the same sweep ID** across all machines
3. **Run agents** on each machine simultaneously
4. **Monitor progress** in the shared W&B dashboard

### Cluster/SLURM Integration
```bash
# Example SLURM job script
#!/bin/bash
#SBATCH --job-name=curvy_sweep
#SBATCH --array=1-10
#SBATCH --gres=gpu:1

./run_sweeps.sh agent <SWEEP_ID> 5
```

## 🎯 Best Practices

### 1. Start Small
- Begin with a few runs to test the setup
- Gradually increase the search space

### 2. Monitor Early
- Check W&B dashboard after first few runs
- Adjust search space if needed

### 3. Use Early Termination
- Enable Hyperband to save compute time
- Set appropriate `min_iter` based on your dataset

### 4. Resource Management
- Monitor GPU/CPU usage
- Adjust `num_workers` in data loaders if needed
- Use appropriate batch sizes for your hardware

## 🔍 Troubleshooting

### Common Issues

**W&B Authentication:**
```bash
wandb login
# or
export WANDB_API_KEY=your_key_here
```

**CUDA Memory Issues:**
- Reduce batch size in sweep config
- Check GPU memory with `nvidia-smi`

**Slow Training:**
- Increase `num_workers` in data loaders
- Use larger batch sizes if memory allows
- Enable mixed precision training

**Sweep Not Starting:**
- Verify sweep ID is correct
- Check W&B project permissions
- Ensure entity name is set correctly

### Performance Optimization

**Data Loading:**
```python
# In create_data_loaders()
pin_memory=True,     # Faster GPU transfer
num_workers=4,       # Parallel data loading
persistent_workers=True  # Keep workers alive
```

**Training Speed:**
- Use larger batch sizes when possible
- Enable mixed precision training
- Use multiple GPUs with DataParallel

## 📈 Results Analysis

### W&B Dashboard Features
- **Parallel Coordinates Plot**: Visualize hyperparameter relationships
- **Parameter Importance**: See which hyperparameters matter most
- **Run Comparison**: Compare training curves side-by-side
- **Best Runs**: Automatically identify top performing configurations

### Export Results
```python
# Access sweep results programmatically
import wandb
api = wandb.Api()
sweep = api.sweep("entity/project/sweep_id")
runs = sweep.runs

for run in runs:
    print(f"Run: {run.name}, Accuracy: {run.summary['final_test_accuracy']}")
```

## 🚀 What's Next?

After finding optimal hyperparameters:
1. **Run longer training** with best configuration
2. **Implement learning rate scheduling**
3. **Add data augmentation**
4. **Try different model architectures**
5. **Experiment with ensemble methods**

## 📝 License

This script is provided for research and educational purposes.

---

**Happy Sweeping! 🧹✨**

For more information on W&B Sweeps, visit: https://docs.wandb.ai/guides/sweeps

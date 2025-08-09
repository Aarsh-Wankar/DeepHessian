# Example Alternative Sweep Configurations
# Copy these into your create_sweep() function in the Python script

# 1. GRID SEARCH (Exhaustive)
grid_search_config = {
    'method': 'grid',
    'metric': {
        'goal': 'maximize',
        'name': 'final_test_accuracy'
    },
    'parameters': {
        'batch_size': {'values': [64, 128]},
        'learning_rate': {'values': [1e-2, 1e-3, 1e-4]},
        'hessian_epsilon': {'values': [1e-2, 1e-3, 1e-4]},
        'momentum': {'values': [0.9, 0.5, 0]},
        'hessian_compute_interval': {'values': [10, 50, 100]},
        'hessian_n_iter': {'value': 100}
    }
}

# 2. RANDOM SEARCH (Fast exploration)
random_search_config = {
    'method': 'random',
    'metric': {
        'goal': 'maximize',
        'name': 'final_test_accuracy'
    },
    'parameters': {
        'batch_size': {'values': [64, 128, 256]},
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-1
        },
        'hessian_epsilon': {
            'distribution': 'log_uniform_values',
            'min': 1e-5,
            'max': 1e-1
        },
        'momentum': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.99
        },
        'hessian_compute_interval': {
            'distribution': 'int_uniform',
            'min': 10,
            'max': 100
        },
        'hessian_n_iter': {'value': 100}
    }
}

# 3. BAYESIAN OPTIMIZATION (Recommended - most efficient)
bayesian_config = {
    'method': 'bayes',
    'metric': {
        'goal': 'maximize',
        'name': 'final_test_accuracy'
    },
    'parameters': {
        'batch_size': {'values': [64, 128]},
        'learning_rate': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 1e-1
        },
        'hessian_epsilon': {
            'distribution': 'log_uniform_values',
            'min': 1e-4,
            'max': 1e-1
        },
        'momentum': {
            'distribution': 'uniform',
            'min': 0.0,
            'max': 0.99
        },
        'hessian_compute_interval': {'values': [10, 25, 50, 100]},
        'hessian_n_iter': {'value': 100}
    },
    'early_terminate': {
        'type': 'hyperband',
        'min_iter': 5,
        'eta': 2
    }
}

# 4. QUICK TEST CONFIG (Small search space for testing)
test_config = {
    'method': 'grid',
    'metric': {
        'goal': 'maximize',
        'name': 'final_test_accuracy'
    },
    'parameters': {
        'batch_size': {'value': 64},
        'learning_rate': {'values': [1e-2, 1e-3]},
        'hessian_epsilon': {'values': [1e-2, 1e-3]},
        'momentum': {'values': [0.9, 0.5]},
        'hessian_compute_interval': {'values': [25, 50]},
        'hessian_n_iter': {'value': 50}
    }
}

# HOW TO USE:
# Replace the sweep_config in create_sweep() function with any of the above configs.
# For example, to use Bayesian optimization:
# 
# def create_sweep():
#     sweep_config = bayesian_config  # Use this line
#     sweep_id = wandb.sweep(
#         sweep=sweep_config,
#         project=EXPERIMENT_CONFIG['project_name'],
#         entity=EXPERIMENT_CONFIG['entity']
#     )
#     return sweep_id

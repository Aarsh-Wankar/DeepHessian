#!/bin/bash
"""
W&B Sweeps Helper Script for Curvy Optimizer Experiments
This script provides easy commands to create and run W&B sweeps
"""

SCRIPT_NAME="resnet-18-hyperparam-tune.py"
PROJECT_NAME="hessian-project-curvy"

echo "🚀 Curvy Optimizer W&B Sweeps Helper"
echo "======================================"

# Check if script exists
if [ ! -f "$SCRIPT_NAME" ]; then
    echo "❌ Error: $SCRIPT_NAME not found in current directory"
    exit 1
fi

# Function to show usage
show_usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  setup     - Install dependencies and setup W&B"
    echo "  create    - Create a new W&B sweep"
    echo "  agent     - Run a sweep agent (requires sweep ID)"
    echo "  single    - Run a single experiment"
    echo "  status    - Check W&B login status"
    echo "  help      - Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 setup                    # First time setup"
    echo "  $0 create                   # Create a new sweep"
    echo "  $0 agent sweep_id_here      # Run sweep agent"
    echo "  $0 single                   # Run single experiment"
}

# Function to setup environment
setup_env() {
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
    
    echo "🔑 Setting up W&B authentication..."
    wandb login
    
    echo "✅ Setup complete!"
    echo "💡 Next: Run '$0 create' to create a sweep"
}

# Function to create sweep
create_sweep() {
    echo "🎯 Creating W&B sweep..."
    python $SCRIPT_NAME --mode sweep
    echo ""
    echo "💡 Copy the sweep ID from above and run:"
    echo "   $0 agent <SWEEP_ID>"
}

# Function to run sweep agent
run_agent() {
    if [ -z "$2" ]; then
        echo "❌ Error: Sweep ID required"
        echo "Usage: $0 agent <SWEEP_ID> [count]"
        echo "Example: $0 agent 12345678 20"
        exit 1
    fi
    
    SWEEP_ID="$2"
    COUNT="${3:-200}"  # Default to 10 runs
    
    echo "🤖 Running sweep agent..."
    echo "Sweep ID: $SWEEP_ID"
    echo "Max runs: $COUNT"
    echo ""
    
    python $SCRIPT_NAME --mode agent --sweep_id "$SWEEP_ID" --count "$COUNT"
}

# Function to run single experiment
run_single() {
    echo "🧪 Running single experiment..."
    python $SCRIPT_NAME --mode single
}

# Function to check W&B status
check_status() {
    echo "🔍 Checking W&B status..."
    wandb status
}

# Main command handling
case "${1:-help}" in
    "setup")
        setup_env
        ;;
    "create")
        create_sweep
        ;;
    "agent")
        run_agent "$@"
        ;;
    "single")
        run_single
        ;;
    "status")
        check_status
        ;;
    "help"|"--help"|"-h")
        show_usage
        ;;
    *)
        echo "❌ Unknown command: $1"
        echo ""
        show_usage
        exit 1
        ;;
esac

# MNIST Neural Network Visualization

Interactive visualization of neural network training and inference on MNIST dataset.

## Features

- **Training Visualization**: Watch weights evolve and loss decrease during training
- **Activation Patterns**: See how neurons activate for different inputs
- **Interactive Explorer**: Slider-based exploration of network behavior on test samples
- **Architecture Diagram**: Visual representation of the network structure
- **Statistical Analysis**: Correlation matrices and sparsity analysis

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Train and visualize the model
```bash
python train_visualized.py
```

This generates:
- `weight_evolution.png` - Weight distribution changes during training
- `activation_patterns.png` - Neuron activation heatmaps
- `inference_visualization.png` - Single inference breakdown
- `training_animation.gif` - Animated training progress

### Interactive visualization
```bash
python interactive_visualization.py
```

This creates:
- Interactive neuron explorer with sample slider
- Network architecture diagram
- Activation correlation analysis

## Network Architecture

- Input: 784 neurons (28x28 flattened MNIST images)
- Hidden Layer 1: 128 neurons with ReLU
- Hidden Layer 2: 64 neurons with ReLU
- Output: 10 neurons (digit classes 0-9)

## Key Insights

The visualizations reveal:
- How different neurons specialize for different digit patterns
- Weight distribution evolution from random to structured
- Activation sparsity patterns
- Layer-wise information flow
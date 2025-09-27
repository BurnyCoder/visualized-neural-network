import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.gridspec import GridSpec
import seaborn as sns
from IPython.display import HTML
import os

# Set style for better visualizations
plt.style.use('dark_background')
sns.set_palette("husl")

# Simple neural network with 2 hidden layers
class SimpleNN(nn.Module):
    def __init__(self, input_size=784, hidden1=128, hidden2=64, output_size=10):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, output_size)

        # Store activations for visualization
        self.activations = {}

    def forward(self, x):
        x = x.view(-1, 784)

        # First layer
        z1 = self.fc1(x)
        a1 = F.relu(z1)
        self.activations['layer1'] = a1.detach()

        # Second layer
        z2 = self.fc2(a1)
        a2 = F.relu(z2)
        self.activations['layer2'] = a2.detach()

        # Output layer
        out = self.fc3(a2)
        self.activations['output'] = F.softmax(out, dim=1).detach()

        return out

    def get_weight_stats(self):
        stats = {}
        stats['fc1_weights'] = self.fc1.weight.detach().cpu().numpy()
        stats['fc2_weights'] = self.fc2.weight.detach().cpu().numpy()
        stats['fc3_weights'] = self.fc3.weight.detach().cpu().numpy()
        stats['fc1_bias'] = self.fc1.bias.detach().cpu().numpy()
        stats['fc2_bias'] = self.fc2.bias.detach().cpu().numpy()
        stats['fc3_bias'] = self.fc3.bias.detach().cpu().numpy()
        return stats

class NetworkVisualizer:
    def __init__(self, model):
        self.model = model
        self.weight_history = []
        self.activation_history = []
        self.loss_history = []
        self.accuracy_history = []

    def record_training_step(self, loss, accuracy):
        self.weight_history.append(self.model.get_weight_stats())
        self.loss_history.append(loss)
        self.accuracy_history.append(accuracy)

    def record_activations(self, batch_data):
        self.activation_history.append({
            'layer1': self.model.activations['layer1'].cpu().numpy(),
            'layer2': self.model.activations['layer2'].cpu().numpy(),
            'output': self.model.activations['output'].cpu().numpy()
        })

    def plot_weight_evolution(self):
        fig, axes = plt.subplots(2, 3, figsize=(15, 8))
        fig.suptitle('Weight Distribution Evolution During Training', fontsize=16)

        # Plot weight distributions over time
        layer_names = ['fc1_weights', 'fc2_weights', 'fc3_weights']
        for idx, layer in enumerate(layer_names):
            ax = axes[0, idx]
            weights_over_time = [w[layer].flatten() for w in self.weight_history[::10]]

            # Create violin plot
            positions = range(len(weights_over_time))
            parts = ax.violinplot(weights_over_time, positions=positions,
                                 widths=0.7, showmeans=True)

            ax.set_title(f'Layer {idx+1} Weights')
            ax.set_xlabel('Training Step (x10)')
            ax.set_ylabel('Weight Value')
            ax.grid(alpha=0.3)

        # Plot bias evolution
        bias_names = ['fc1_bias', 'fc2_bias', 'fc3_bias']
        for idx, bias in enumerate(bias_names):
            ax = axes[1, idx]
            bias_values = np.array([w[bias] for w in self.weight_history])

            # Plot mean and std of biases
            mean_bias = np.mean(bias_values, axis=1)
            std_bias = np.std(bias_values, axis=1)

            ax.plot(mean_bias, label='Mean', linewidth=2)
            ax.fill_between(range(len(mean_bias)),
                           mean_bias - std_bias,
                           mean_bias + std_bias,
                           alpha=0.3)

            ax.set_title(f'Layer {idx+1} Bias Evolution')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Bias Value')
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_activation_patterns(self, test_loader):
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Get device
        device = next(self.model.parameters()).device

        # Get a batch of test data
        data, labels = next(iter(test_loader))
        data = data[:8].to(device)  # Take first 8 samples and move to device
        labels = labels[:8]

        # Forward pass to get activations
        with torch.no_grad():
            outputs = self.model(data)
            predictions = outputs.argmax(dim=1).cpu()

        # Plot original images
        for i in range(8):
            ax = fig.add_subplot(gs[i//4, i%4])
            ax.imshow(data[i].cpu().squeeze(), cmap='hot')
            ax.set_title(f'True: {labels[i]}, Pred: {predictions[i]}')
            ax.axis('off')

        # Plot activation heatmaps for different layers
        layer1_acts = self.model.activations['layer1'][:8].cpu().numpy()
        layer2_acts = self.model.activations['layer2'][:8].cpu().numpy()

        # Create activation heatmap for layer 1
        ax = fig.add_subplot(gs[2, :2])
        layer1_heatmap = layer1_acts.reshape(-1, layer1_acts.shape[-1])
        im = ax.imshow(layer1_heatmap, aspect='auto', cmap='viridis')
        ax.set_title('Layer 1 Activations (128 neurons)')
        ax.set_xlabel('Neuron Index')
        ax.set_ylabel('Sample')
        plt.colorbar(im, ax=ax)

        # Create activation heatmap for layer 2
        ax = fig.add_subplot(gs[2, 2:])
        layer2_heatmap = layer2_acts.reshape(-1, layer2_acts.shape[-1])
        im = ax.imshow(layer2_heatmap, aspect='auto', cmap='plasma')
        ax.set_title('Layer 2 Activations (64 neurons)')
        ax.set_xlabel('Neuron Index')
        ax.set_ylabel('Sample')
        plt.colorbar(im, ax=ax)

        return fig

    def create_training_animation(self):
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Initialize plots
        ax1, ax2, ax3, ax4 = axes.flatten()

        # Loss plot
        line1, = ax1.plot([], [], 'b-', linewidth=2)
        ax1.set_xlim(0, len(self.loss_history))
        ax1.set_ylim(0, max(self.loss_history) * 1.1)
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Batch')
        ax1.set_ylabel('Loss')
        ax1.grid(alpha=0.3)

        # Accuracy plot
        line2, = ax2.plot([], [], 'g-', linewidth=2)
        ax2.set_xlim(0, len(self.accuracy_history))
        ax2.set_ylim(0, 1.1)
        ax2.set_title('Training Accuracy')
        ax2.set_xlabel('Batch')
        ax2.set_ylabel('Accuracy')
        ax2.grid(alpha=0.3)

        # Weight histogram
        ax3.set_title('Weight Distribution (Layer 1)')
        ax3.set_xlabel('Weight Value')
        ax3.set_ylabel('Frequency')

        # Gradient flow
        ax4.set_title('Weight Gradient Magnitudes')
        ax4.set_xlabel('Layer')
        ax4.set_ylabel('Gradient Magnitude')

        def animate(frame):
            # Update loss
            line1.set_data(range(frame), self.loss_history[:frame])

            # Update accuracy
            line2.set_data(range(frame), self.accuracy_history[:frame])

            # Update weight histogram
            if frame < len(self.weight_history):
                ax3.clear()
                weights = self.weight_history[frame]['fc1_weights'].flatten()
                ax3.hist(weights, bins=50, alpha=0.7, color='purple')
                ax3.set_title(f'Weight Distribution (Layer 1) - Step {frame}')
                ax3.set_xlabel('Weight Value')
                ax3.set_ylabel('Frequency')

                # Update gradient magnitudes
                ax4.clear()
                layers = ['Layer 1', 'Layer 2', 'Layer 3']
                grad_mags = [
                    np.std(self.weight_history[frame]['fc1_weights']),
                    np.std(self.weight_history[frame]['fc2_weights']),
                    np.std(self.weight_history[frame]['fc3_weights'])
                ]
                ax4.bar(layers, grad_mags, color=['cyan', 'magenta', 'yellow'])
                ax4.set_title('Weight Standard Deviation by Layer')
                ax4.set_ylabel('Std Dev')

            return line1, line2

        anim = animation.FuncAnimation(
            fig, animate, frames=len(self.loss_history),
            interval=50, blit=False, repeat=True
        )

        return anim

def train_model():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Initialize model and training components
    model = SimpleNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    visualizer = NetworkVisualizer(model)

    # Training loop
    print("Starting training...")
    epochs = 3

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            accuracy = correct / total

            running_loss += loss.item()

            # Record for visualization (sample every 10 batches)
            if batch_idx % 10 == 0:
                visualizer.record_training_step(loss.item(), accuracy)
                visualizer.record_activations(data)

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}: Loss = {loss.item():.4f}, Acc = {accuracy:.4f}')

    # Test the model
    model.eval()
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            test_total += target.size(0)
            test_correct += (predicted == target).sum().item()

    test_accuracy = test_correct / test_total
    print(f'\nTest Accuracy: {test_accuracy:.4f}')

    return model, visualizer, test_loader

def visualize_inference(model, test_loader):
    # Create figure for live inference visualization
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.4, wspace=0.3)

    # Get device
    device = next(model.parameters()).device

    # Get a single test sample
    data, label = next(iter(test_loader))
    sample = data[0:1].to(device)
    true_label = label[0].item()

    # Forward pass
    with torch.no_grad():
        output = model(sample)
        prediction = output.argmax(dim=1).item()
        probs = F.softmax(output, dim=1).squeeze().cpu().numpy()

    # Plot input image
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(sample.cpu().squeeze(), cmap='hot')
    ax1.set_title(f'Input: {true_label}')
    ax1.axis('off')

    # Plot layer 1 activations
    ax2 = fig.add_subplot(gs[0, 1:3])
    layer1_acts = model.activations['layer1'].squeeze().cpu().numpy()
    ax2.bar(range(len(layer1_acts)), layer1_acts, color='cyan', alpha=0.7)
    ax2.set_title('Layer 1 Activations (128 neurons)')
    ax2.set_xlabel('Neuron Index')
    ax2.set_ylabel('Activation')
    ax2.set_ylim([0, max(layer1_acts) * 1.1])

    # Plot layer 2 activations
    ax3 = fig.add_subplot(gs[1, :2])
    layer2_acts = model.activations['layer2'].squeeze().cpu().numpy()
    ax3.bar(range(len(layer2_acts)), layer2_acts, color='magenta', alpha=0.7)
    ax3.set_title('Layer 2 Activations (64 neurons)')
    ax3.set_xlabel('Neuron Index')
    ax3.set_ylabel('Activation')

    # Plot output probabilities
    ax4 = fig.add_subplot(gs[1, 2:])
    colors = ['green' if i == prediction else 'red' if i == true_label else 'gray'
              for i in range(10)]
    bars = ax4.bar(range(10), probs, color=colors, alpha=0.7)
    ax4.set_title(f'Output Probabilities (Pred: {prediction})')
    ax4.set_xlabel('Digit Class')
    ax4.set_ylabel('Probability')
    ax4.set_xticks(range(10))

    # Visualize weight connections for top activated neurons
    ax5 = fig.add_subplot(gs[2, :])

    # Get top 10 most activated neurons from layer 1
    top_neurons = np.argsort(layer1_acts)[-10:]

    # Get their input weights (connections from input)
    fc1_weights = model.fc1.weight.detach().cpu().numpy()
    top_weights = fc1_weights[top_neurons]

    # Reshape and visualize as heatmap
    im = ax5.imshow(top_weights, aspect='auto', cmap='RdBu', vmin=-2, vmax=2)
    ax5.set_title('Input Weights of Top 10 Most Activated Neurons (Layer 1)')
    ax5.set_xlabel('Input Pixel Index')
    ax5.set_ylabel('Neuron Index')
    ax5.set_yticks(range(10))
    ax5.set_yticklabels([f'N{i}' for i in top_neurons])
    plt.colorbar(im, ax=ax5)

    plt.suptitle('Neural Network Inference Visualization', fontsize=16, y=0.98)
    return fig

if __name__ == "__main__":
    # Train the model
    model, visualizer, test_loader = train_model()

    # Create visualizations
    print("\nGenerating visualizations...")

    # 1. Weight evolution
    weight_fig = visualizer.plot_weight_evolution()
    plt.savefig('weight_evolution.png', dpi=100, bbox_inches='tight')
    plt.show()

    # 2. Activation patterns
    activation_fig = visualizer.plot_activation_patterns(test_loader)
    plt.savefig('activation_patterns.png', dpi=100, bbox_inches='tight')
    plt.show()

    # 3. Inference visualization
    inference_fig = visualize_inference(model, test_loader)
    plt.savefig('inference_visualization.png', dpi=100, bbox_inches='tight')
    plt.show()

    # 4. Training animation (save as gif)
    print("Creating training animation...")
    anim = visualizer.create_training_animation()
    anim.save('training_animation.gif', writer='pillow', fps=10)
    print("Animation saved as 'training_animation.gif'")

    print("\nVisualization complete! Generated files:")
    print("- weight_evolution.png")
    print("- activation_patterns.png")
    print("- inference_visualization.png")
    print("- training_animation.gif")
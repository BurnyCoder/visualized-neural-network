import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.collections import LineCollection
import seaborn as sns

plt.style.use('dark_background')

class InteractiveNNVisualizer:
    def __init__(self, model, test_loader):
        self.model = model
        self.device = next(model.parameters()).device
        self.test_loader = test_loader
        self.test_data, self.test_labels = self._load_test_batch()
        self.current_idx = 0

    def _load_test_batch(self):
        data, labels = next(iter(self.test_loader))
        return data.to(self.device), labels.to(self.device)

    def create_interactive_neuron_explorer(self):
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        # Initial sample
        sample = self.test_data[0:1]
        label = self.test_labels[0].item()

        # Forward pass
        with torch.no_grad():
            output = self.model(sample)
            prediction = output.argmax(dim=1).item()

        # Main visualization axes
        ax_input = fig.add_subplot(gs[0:2, 0])
        ax_layer1 = fig.add_subplot(gs[0, 1:3])
        ax_layer2 = fig.add_subplot(gs[1, 1:3])
        ax_output = fig.add_subplot(gs[0:2, 3])
        ax_weights = fig.add_subplot(gs[2, :])

        # Slider for sample selection
        ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
        sample_slider = Slider(ax_slider, 'Sample', 0, len(self.test_data)-1,
                              valinit=0, valstep=1, color='cyan')

        def update(val):
            idx = int(sample_slider.val)
            sample = self.test_data[idx:idx+1]
            label = self.test_labels[idx].item()

            # Forward pass
            with torch.no_grad():
                output = self.model(sample)
                prediction = output.argmax(dim=1).item()
                probs = F.softmax(output, dim=1).squeeze().cpu().numpy()

            # Clear all axes
            for ax in [ax_input, ax_layer1, ax_layer2, ax_output, ax_weights]:
                ax.clear()

            # Update input display
            ax_input.imshow(sample.squeeze().cpu(), cmap='hot')
            ax_input.set_title(f'Input Digit\nTrue: {label}, Pred: {prediction}',
                              fontsize=12, color='white' if label == prediction else 'red')
            ax_input.axis('off')

            # Update layer 1 activations
            layer1_acts = self.model.activations['layer1'].squeeze().cpu().numpy()
            bars1 = ax_layer1.bar(range(len(layer1_acts)), layer1_acts,
                                 color='cyan', alpha=0.6)

            # Highlight top 10 neurons
            top10_layer1 = np.argsort(layer1_acts)[-10:]
            for idx in top10_layer1:
                bars1[idx].set_color('yellow')
                bars1[idx].set_alpha(1.0)

            ax_layer1.set_title('Layer 1 Activations (128 neurons)\nYellow = Top 10')
            ax_layer1.set_xlabel('Neuron Index')
            ax_layer1.set_ylabel('Activation')
            ax_layer1.grid(alpha=0.2)

            # Update layer 2 activations
            layer2_acts = self.model.activations['layer2'].squeeze().cpu().numpy()
            bars2 = ax_layer2.bar(range(len(layer2_acts)), layer2_acts,
                                 color='magenta', alpha=0.6)

            # Highlight top 5 neurons
            top5_layer2 = np.argsort(layer2_acts)[-5:]
            for idx in top5_layer2:
                bars2[idx].set_color('orange')
                bars2[idx].set_alpha(1.0)

            ax_layer2.set_title('Layer 2 Activations (64 neurons)\nOrange = Top 5')
            ax_layer2.set_xlabel('Neuron Index')
            ax_layer2.set_ylabel('Activation')
            ax_layer2.grid(alpha=0.2)

            # Update output probabilities
            colors = ['green' if i == prediction else 'red' if i == label else 'gray'
                     for i in range(10)]
            ax_output.barh(range(10), probs, color=colors, alpha=0.8)
            ax_output.set_title(f'Output Probabilities')
            ax_output.set_ylabel('Digit Class')
            ax_output.set_xlabel('Probability')
            ax_output.set_yticks(range(10))
            ax_output.set_xlim([0, 1])
            ax_output.grid(alpha=0.2)

            # Add probability values as text
            for i, prob in enumerate(probs):
                ax_output.text(prob + 0.01, i, f'{prob:.3f}', va='center')

            # Visualize weight patterns for most active neurons
            fc1_weights = self.model.fc1.weight.detach().cpu().numpy()
            fc2_weights = self.model.fc2.weight.detach().cpu().numpy()

            # Get weights connecting to most active layer 2 neurons
            top_layer2_idx = np.argmax(layer2_acts)
            connecting_weights = fc2_weights[top_layer2_idx]

            # Find which layer 1 neurons contribute most to this layer 2 neuron
            top_contributors = np.argsort(np.abs(connecting_weights))[-20:]

            # Visualize input patterns that activate these neurons
            input_patterns = fc1_weights[top_contributors].reshape(20, 28, 28)

            # Create a grid of input patterns
            for i in range(20):
                ax_sub = ax_weights.inset_axes([i*0.05, 0, 0.045, 0.9])
                ax_sub.imshow(input_patterns[i], cmap='RdBu', vmin=-2, vmax=2)
                ax_sub.set_xticks([])
                ax_sub.set_yticks([])

            ax_weights.set_title(f'Input Patterns → Top Layer1 Neurons → Most Active Layer2 Neuron #{top_layer2_idx}')
            ax_weights.axis('off')

            plt.draw()

        sample_slider.on_changed(update)

        # Initial draw
        update(0)

        plt.suptitle('Interactive Neural Network Explorer', fontsize=16, y=0.98)
        return fig

    def create_network_architecture_viz(self):
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.set_xlim(-0.5, 4.5)
        ax.set_ylim(-0.5, 10.5)
        ax.axis('off')

        # Network structure
        layers = {
            'input': {'pos': 0, 'size': 10, 'neurons': 784},
            'hidden1': {'pos': 1.5, 'size': 8, 'neurons': 128},
            'hidden2': {'pos': 2.5, 'size': 6, 'neurons': 64},
            'output': {'pos': 4, 'size': 10, 'neurons': 10}
        }

        # Draw neurons and connections
        neuron_positions = {}

        for layer_name, layer_info in layers.items():
            x_pos = layer_info['pos']
            n_display = min(layer_info['size'], layer_info['neurons'])

            # Calculate y positions
            y_positions = np.linspace(1, 9, n_display)

            for i, y_pos in enumerate(y_positions):
                # Neuron circle
                if layer_name == 'input':
                    color = 'cyan'
                elif layer_name == 'output':
                    color = 'green'
                else:
                    color = 'yellow'

                circle = Circle((x_pos, y_pos), 0.15, color=color, alpha=0.7)
                ax.add_patch(circle)

                if layer_name not in neuron_positions:
                    neuron_positions[layer_name] = []
                neuron_positions[layer_name].append((x_pos, y_pos))

            # Add layer label
            ax.text(x_pos, 0, f'{layer_name.capitalize()}\n({layer_info["neurons"]} neurons)',
                   ha='center', fontsize=10, weight='bold')

        # Draw connections with varying opacity based on typical weights
        np.random.seed(42)
        for i, (layer1, layer2) in enumerate(zip(['input', 'hidden1', 'hidden2'],
                                                ['hidden1', 'hidden2', 'output'])):
            for n1 in neuron_positions[layer1][:3]:  # Sample connections
                for n2 in neuron_positions[layer2][:3]:
                    alpha = np.random.uniform(0.1, 0.5)
                    ax.plot([n1[0], n2[0]], [n1[1], n2[1]],
                           'w-', alpha=alpha, linewidth=0.5)

        ax.set_title('Neural Network Architecture\n(Simplified View - Showing Sample Connections)',
                    fontsize=14, pad=20)

        # Add legend
        legend_elements = [
            mpatches.Patch(color='cyan', label='Input Layer (784)'),
            mpatches.Patch(color='yellow', label='Hidden Layers (128, 64)'),
            mpatches.Patch(color='green', label='Output Layer (10)')
        ]
        ax.legend(handles=legend_elements, loc='upper right')

        return fig

    def create_activation_heatmap(self):
        # Process multiple samples to get activation statistics
        n_samples = 100
        all_acts_layer1 = []
        all_acts_layer2 = []
        predictions = []
        true_labels = []

        with torch.no_grad():
            for i in range(min(n_samples, len(self.test_data))):
                sample = self.test_data[i:i+1]
                label = self.test_labels[i].item()

                output = self.model(sample)
                pred = output.argmax(dim=1).item()

                all_acts_layer1.append(self.model.activations['layer1'].squeeze().cpu().numpy())
                all_acts_layer2.append(self.model.activations['layer2'].squeeze().cpu().numpy())
                predictions.append(pred)
                true_labels.append(label)

        # Convert to arrays
        acts1 = np.array(all_acts_layer1)
        acts2 = np.array(all_acts_layer2)

        # Create correlation heatmaps
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))

        # Activation patterns per digit class
        ax = axes[0, 0]
        mean_acts_per_digit = np.zeros((10, 64))
        for digit in range(10):
            mask = np.array(true_labels) == digit
            if np.sum(mask) > 0:
                mean_acts_per_digit[digit] = np.mean(acts2[mask], axis=0)

        im1 = ax.imshow(mean_acts_per_digit, aspect='auto', cmap='plasma')
        ax.set_title('Mean Layer 2 Activations per Digit Class')
        ax.set_xlabel('Neuron Index (Layer 2)')
        ax.set_ylabel('Digit Class')
        ax.set_yticks(range(10))
        plt.colorbar(im1, ax=ax)

        # Neuron correlation matrix (Layer 1)
        ax = axes[0, 1]
        corr_matrix1 = np.corrcoef(acts1.T)
        im2 = ax.imshow(corr_matrix1[:50, :50], cmap='RdBu', vmin=-1, vmax=1)
        ax.set_title('Layer 1 Neuron Correlations (First 50)')
        ax.set_xlabel('Neuron Index')
        ax.set_ylabel('Neuron Index')
        plt.colorbar(im2, ax=ax)

        # Activation distribution
        ax = axes[1, 0]
        ax.hist([acts1.flatten(), acts2.flatten()],
               bins=50, label=['Layer 1', 'Layer 2'],
               alpha=0.7, color=['cyan', 'magenta'])
        ax.set_title('Activation Value Distribution')
        ax.set_xlabel('Activation Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(alpha=0.2)

        # Sparsity analysis
        ax = axes[1, 1]
        sparsity1 = np.mean(acts1 == 0, axis=1)
        sparsity2 = np.mean(acts2 == 0, axis=1)

        ax.scatter(sparsity1, sparsity2, alpha=0.5, c=true_labels, cmap='tab10')
        ax.set_title('Activation Sparsity per Sample')
        ax.set_xlabel('Layer 1 Sparsity (% zeros)')
        ax.set_ylabel('Layer 2 Sparsity (% zeros)')
        ax.grid(alpha=0.2)

        # Add colorbar for digit labels
        sm = plt.cm.ScalarMappable(cmap='tab10', norm=plt.Normalize(0, 9))
        sm.set_array([])
        plt.colorbar(sm, ax=ax, label='Digit Class')

        plt.suptitle('Neural Activation Analysis', fontsize=16, y=1.02)
        plt.tight_layout()

        return fig

if __name__ == "__main__":
    # Load pre-trained model or train a new one
    from train_visualized import SimpleNN

    # Setup
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Load or create model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SimpleNN().to(device)

    # Try to load pre-trained weights if available
    try:
        model.load_state_dict(torch.load('model_weights.pth', map_location=device))
        print("Loaded pre-trained model")
    except:
        print("No pre-trained model found. Training new model...")
        # Train quickly for demo
        from train_visualized import train_model
        model, _, _ = train_model()
        model = model.to(device)
        torch.save(model.state_dict(), 'model_weights.pth')

    # Create visualizer
    visualizer = InteractiveNNVisualizer(model, test_loader)

    # Generate interactive visualizations
    print("Creating interactive visualizations...")

    # 1. Interactive neuron explorer
    fig1 = visualizer.create_interactive_neuron_explorer()
    plt.savefig('interactive_explorer.png', dpi=100, bbox_inches='tight')

    # 2. Network architecture visualization
    fig2 = visualizer.create_network_architecture_viz()
    plt.savefig('network_architecture.png', dpi=100, bbox_inches='tight')

    # 3. Activation heatmap analysis
    fig3 = visualizer.create_activation_heatmap()
    plt.savefig('activation_analysis.png', dpi=100, bbox_inches='tight')

    plt.show()

    print("\nInteractive visualizations created!")
    print("Files generated:")
    print("- interactive_explorer.png")
    print("- network_architecture.png")
    print("- activation_analysis.png")
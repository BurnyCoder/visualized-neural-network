import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.widgets import Slider, Button, RadioButtons
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
from IPython.display import HTML
import time
from collections import deque

plt.style.use('dark_background')

class WeightEvolutionVisualizer:
    def __init__(self, input_size=784, hidden1_size=128, hidden2_size=64, output_size=10):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.input_size = input_size
        self.hidden1_size = hidden1_size
        self.hidden2_size = hidden2_size
        self.output_size = output_size

        self.weight_history = []
        self.activation_history = []
        self.loss_history = []
        self.accuracy_history = []
        self.inference_states = {}

        self.model = None
        self.optimizer = None

        self.training_data = None
        self.test_data = None

    def create_model(self):
        class TrackableNN(nn.Module):
            def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
                super().__init__()
                self.fc1 = nn.Linear(input_size, hidden1_size)
                self.fc2 = nn.Linear(hidden1_size, hidden2_size)
                self.fc3 = nn.Linear(hidden2_size, output_size)
                self.activations = {}

            def forward(self, x):
                x = x.view(-1, 784)
                x = F.relu(self.fc1(x))
                self.activations['layer1'] = x
                x = F.relu(self.fc2(x))
                self.activations['layer2'] = x
                x = self.fc3(x)
                self.activations['output'] = x
                return x

        self.model = TrackableNN(self.input_size, self.hidden1_size,
                                 self.hidden2_size, self.output_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def load_data(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('data', train=False, transform=transform)

        self.train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
        self.test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

        self.training_data = next(iter(self.train_loader))
        self.test_data = next(iter(self.test_loader))

    def capture_weights(self):
        weights = {}
        weights['fc1'] = self.model.fc1.weight.detach().cpu().numpy().flatten()
        weights['fc2'] = self.model.fc2.weight.detach().cpu().numpy().flatten()
        weights['fc3'] = self.model.fc3.weight.detach().cpu().numpy().flatten()

        weights['fc1_bias'] = self.model.fc1.bias.detach().cpu().numpy()
        weights['fc2_bias'] = self.model.fc2.bias.detach().cpu().numpy()
        weights['fc3_bias'] = self.model.fc3.bias.detach().cpu().numpy()

        return weights

    def weights_to_image(self, weights):
        all_weights = np.concatenate([
            weights['fc1'],
            weights['fc2'],
            weights['fc3'],
            weights['fc1_bias'],
            weights['fc2_bias'],
            weights['fc3_bias']
        ])

        total_params = len(all_weights)
        img_size = int(np.ceil(np.sqrt(total_params)))

        padded_weights = np.zeros(img_size * img_size)
        padded_weights[:total_params] = all_weights

        weight_image = padded_weights.reshape(img_size, img_size)

        return weight_image

    def train_and_record(self, epochs=5, record_every=10):
        print("Training and recording weight evolution...")
        self.create_model()

        step = 0
        for epoch in range(epochs):
            for batch_idx, (data, target) in enumerate(self.train_loader):
                if batch_idx > 100:
                    break

                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                self.optimizer.step()

                if step % record_every == 0:
                    weights = self.capture_weights()
                    self.weight_history.append({
                        'weights': weights,
                        'step': step,
                        'epoch': epoch,
                        'loss': loss.item()
                    })

                    with torch.no_grad():
                        test_data, test_target = self.test_data[0].to(self.device), self.test_data[1].to(self.device)
                        test_output = self.model(test_data)
                        test_pred = test_output.argmax(dim=1, keepdim=True)
                        accuracy = test_pred.eq(test_target.view_as(test_pred)).sum().item() / len(test_target)

                    self.loss_history.append(loss.item())
                    self.accuracy_history.append(accuracy)

                    print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}, Acc: {accuracy:.3f}")

                step += 1

        print(f"Training complete. Recorded {len(self.weight_history)} weight snapshots.")

    def record_inference(self):
        print("Recording inference for different digits...")

        # Switch model to eval mode but keep track of activations
        self.model.eval()

        # First pass: collect all digit activations and compute baseline
        all_activations = []

        for digit in range(10):
            digit_samples = []
            for data, target in self.test_loader:
                mask = target == digit
                if mask.any():
                    sample = data[mask][0:1].to(self.device)

                    # Forward pass without gradients first
                    with torch.no_grad():
                        output = self.model(sample)
                        prediction = output.argmax(dim=1).item()
                        probs = F.softmax(output, dim=1).squeeze().cpu().numpy()

                        activations = {
                            'layer1': self.model.activations['layer1'].clone().cpu().numpy(),
                            'layer2': self.model.activations['layer2'].clone().cpu().numpy(),
                            'output': probs
                        }

                    # Now compute gradients for activation analysis
                    sample_grad = sample.clone().detach().requires_grad_(True)
                    output_grad = self.model(sample_grad)

                    # Compute activation gradients for this specific digit
                    grad_output = torch.zeros_like(output_grad)
                    grad_output[0, digit] = 1.0  # Gradient w.r.t. the true digit class
                    output_grad.backward(gradient=grad_output)

                    input_gradients = sample_grad.grad.clone().cpu().numpy() if sample_grad.grad is not None else None

                    weights = self.capture_weights()

                    self.inference_states[digit] = {
                        'input': sample.cpu().numpy(),
                        'weights': weights,
                        'activations': activations,
                        'prediction': prediction,
                        'probabilities': probs,
                        'input_gradients': input_gradients
                    }

                    all_activations.append(activations)
                    break

            # Compute mean activations across all digits (baseline)
            if all_activations:
                mean_activations = {
                    'layer1': np.mean([a['layer1'] for a in all_activations], axis=0),
                    'layer2': np.mean([a['layer2'] for a in all_activations], axis=0),
                }

                # Compute differential activations for each digit
                for digit in range(10):
                    if digit in self.inference_states:
                        diff_activations = {
                            'layer1': self.inference_states[digit]['activations']['layer1'] - mean_activations['layer1'],
                            'layer2': self.inference_states[digit]['activations']['layer2'] - mean_activations['layer2'],
                        }
                        self.inference_states[digit]['diff_activations'] = diff_activations

                        # Compute activation strengths (which neurons fire most strongly for this digit)
                        self.inference_states[digit]['activation_strength'] = {
                            'layer1': np.abs(diff_activations['layer1']).mean(),
                            'layer2': np.abs(diff_activations['layer2']).mean(),
                        }

        print(f"Recorded inference states for {len(self.inference_states)} digits with differential analysis.")

    def create_interactive_animation(self):
        fig = plt.figure(figsize=(18, 10))
        gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)

        ax_weights_main = fig.add_subplot(gs[0:2, 0:2])
        ax_loss = fig.add_subplot(gs[0, 2])
        ax_accuracy = fig.add_subplot(gs[1, 2])
        ax_input = fig.add_subplot(gs[0, 3])
        ax_output = fig.add_subplot(gs[1, 3])
        ax_weight_dist = fig.add_subplot(gs[2, 0:2])
        ax_controls = fig.add_subplot(gs[2, 2:])
        ax_controls.axis('off')

        self.current_frame = 0
        self.animation_mode = 'training'
        self.current_digit = 0
        self.is_playing = False

        weight_img = self.weights_to_image(self.weight_history[0]['weights'])
        # Use dynamic scaling based on actual weight values
        vmin, vmax = np.percentile(weight_img, [1, 99])
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
        im_weights = ax_weights_main.imshow(weight_img, cmap='RdBu_r', vmin=vmin, vmax=vmax, animated=True)
        ax_weights_main.set_title('Neural Network Weights as Image\n(All weights mapped to pixels)')
        ax_weights_main.axis('off')

        cbar = plt.colorbar(im_weights, ax=ax_weights_main, fraction=0.046, pad=0.04)

        line_loss, = ax_loss.plot([], [], 'cyan', linewidth=2)
        ax_loss.set_xlim(0, len(self.loss_history))
        ax_loss.set_ylim(0, max(self.loss_history) * 1.1)
        ax_loss.set_xlabel('Step')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title('Training Loss')
        ax_loss.grid(alpha=0.2)

        line_acc, = ax_accuracy.plot([], [], 'green', linewidth=2)
        ax_accuracy.set_xlim(0, len(self.accuracy_history))
        ax_accuracy.set_ylim(0, 1)
        ax_accuracy.set_xlabel('Step')
        ax_accuracy.set_ylabel('Accuracy')
        ax_accuracy.set_title('Test Accuracy')
        ax_accuracy.grid(alpha=0.2)

        ax_slider = plt.axes([0.15, 0.02, 0.5, 0.03])
        frame_slider = Slider(ax_slider, 'Frame', 0, len(self.weight_history)-1,
                             valinit=0, valstep=1, color='cyan')

        ax_play = plt.axes([0.7, 0.02, 0.08, 0.04])
        btn_play = Button(ax_play, 'Play/Pause')

        ax_mode = plt.axes([0.8, 0.02, 0.15, 0.1], facecolor='lightgoldenrodyellow')
        radio = RadioButtons(ax_mode, ('Training', 'Inference'), active=0)

        def update_frame(frame=None):
            if frame is None:
                frame = self.current_frame

            if self.animation_mode == 'training':
                if frame < len(self.weight_history):
                    weight_data = self.weight_history[frame]
                    weight_img = self.weights_to_image(weight_data['weights'])

                    # Update color scaling dynamically
                    vmin, vmax = np.percentile(weight_img, [1, 99])
                    vmax = max(abs(vmin), abs(vmax))
                    vmin = -vmax
                    im_weights.set_clim(vmin=vmin, vmax=vmax)
                    im_weights.set_array(weight_img)

                    ax_weights_main.set_title(
                        f'Neural Network Weights - Training\n'
                        f'Epoch {weight_data["epoch"]}, Step {weight_data["step"]}, Loss: {weight_data["loss"]:.3f}'
                    )

                    line_loss.set_data(range(frame+1), self.loss_history[:frame+1])
                    line_acc.set_data(range(frame+1), self.accuracy_history[:frame+1])

                    ax_input.clear()
                    ax_input.axis('off')
                    ax_input.set_title('Training Mode')

                    ax_output.clear()
                    ax_output.axis('off')

            else:
                if self.current_digit in self.inference_states:
                    state = self.inference_states[self.current_digit]
                    weight_img = self.weights_to_image(state['weights'])

                    # Update color scaling dynamically
                    vmin, vmax = np.percentile(weight_img, [1, 99])
                    vmax = max(abs(vmin), abs(vmax))
                    vmin = -vmax
                    im_weights.set_clim(vmin=vmin, vmax=vmax)
                    im_weights.set_array(weight_img)

                    ax_weights_main.set_title(
                        f'Neural Network Weights - Inference\n'
                        f'Processing Digit: {self.current_digit}'
                    )

                    ax_input.clear()
                    ax_input.imshow(state['input'].squeeze(), cmap='hot')
                    ax_input.set_title(f'Input: {self.current_digit}')
                    ax_input.axis('off')

                    ax_output.clear()
                    colors = ['green' if i == state['prediction'] else 'gray' for i in range(10)]
                    bars = ax_output.bar(range(10), state['probabilities'], color=colors)
                    ax_output.set_title(f'Prediction: {state["prediction"]}')
                    ax_output.set_xlabel('Digit')
                    ax_output.set_ylabel('Probability')
                    ax_output.set_ylim(0, 1)

            ax_weight_dist.clear()
            if self.animation_mode == 'training' and frame < len(self.weight_history):
                weights = self.weight_history[frame]['weights']
            elif self.animation_mode == 'inference' and self.current_digit in self.inference_states:
                weights = self.inference_states[self.current_digit]['weights']
            else:
                return

            all_weights = np.concatenate([
                weights['fc1'].flatten(),
                weights['fc2'].flatten(),
                weights['fc3'].flatten()
            ])

            ax_weight_dist.hist(all_weights, bins=50, alpha=0.7, color='magenta', edgecolor='white')
            ax_weight_dist.set_title('Weight Distribution')
            ax_weight_dist.set_xlabel('Weight Value')
            ax_weight_dist.set_ylabel('Count')
            ax_weight_dist.grid(alpha=0.2)

            ax_weight_dist.axvline(np.mean(all_weights), color='cyan', linestyle='--',
                                   label=f'Mean: {np.mean(all_weights):.3f}')
            ax_weight_dist.axvline(np.median(all_weights), color='yellow', linestyle='--',
                                   label=f'Median: {np.median(all_weights):.3f}')
            ax_weight_dist.legend()

            plt.draw()

        def on_slider_change(val):
            self.current_frame = int(frame_slider.val)
            update_frame(self.current_frame)

        def on_play_pause(event):
            self.is_playing = not self.is_playing
            if self.is_playing:
                animate()

        def on_mode_change(label):
            self.animation_mode = label.lower()
            if self.animation_mode == 'inference':
                frame_slider.set_val(0)
                frame_slider.valmax = 9
                frame_slider.ax.set_xlim(0, 9)
                frame_slider.label.set_text('Digit')
            else:
                frame_slider.valmax = len(self.weight_history) - 1
                frame_slider.ax.set_xlim(0, len(self.weight_history) - 1)
                frame_slider.label.set_text('Frame')
            update_frame()

        def animate():
            if not self.is_playing:
                return

            if self.animation_mode == 'training':
                self.current_frame = (self.current_frame + 1) % len(self.weight_history)
                frame_slider.set_val(self.current_frame)
            else:
                self.current_digit = (self.current_digit + 1) % 10
                frame_slider.set_val(self.current_digit)
                self.current_frame = self.current_digit

            update_frame()
            if self.is_playing:
                fig.canvas.draw_idle()
                plt.pause(0.05)
                animate()

        frame_slider.on_changed(on_slider_change)
        btn_play.on_clicked(on_play_pause)
        radio.on_clicked(on_mode_change)

        update_frame(0)

        plt.suptitle('Neural Network Weight Evolution Animation', fontsize=16, y=0.98)

        return fig

    def create_digit_activation_analysis(self, filename='digit_activation_analysis.png'):
        """Create visualization showing which parts of network activate for each digit"""
        print(f"Creating digit activation analysis visualization: {filename}")

        fig = plt.figure(figsize=(24, 18))
        gs = GridSpec(6, 5, figure=fig, hspace=0.4, wspace=0.3)

        # Title
        fig.suptitle('Neural Network Digit-Specific Activation Analysis\n(Showing which neurons activate most for each digit)',
                     fontsize=18, fontweight='bold', y=0.98)

        # Create subplots for each digit
        for digit in range(10):
            row = digit // 5 * 3  # 0 or 3
            col = digit % 5

            if digit in self.inference_states:
                state = self.inference_states[digit]

                # Input image
                ax_input = fig.add_subplot(gs[row, col])
                ax_input.imshow(state['input'].squeeze(), cmap='hot')
                ax_input.set_title(f'Digit {digit}', fontsize=10, fontweight='bold')
                ax_input.axis('off')

                # Differential activation heatmap for layer 1
                if 'diff_activations' in state:
                    ax_layer1 = fig.add_subplot(gs[row+1, col])
                    diff_act_1 = state['diff_activations']['layer1'].reshape(-1)

                    # Reshape to square for visualization
                    size = int(np.sqrt(len(diff_act_1)))
                    if size * size < len(diff_act_1):
                        size += 1
                    padded = np.zeros(size * size)
                    padded[:len(diff_act_1)] = diff_act_1

                    im1 = ax_layer1.imshow(padded.reshape(size, size), cmap='RdBu_r',
                                           vmin=-np.abs(diff_act_1).max(),
                                           vmax=np.abs(diff_act_1).max())
                    ax_layer1.set_title(f'L1 Diff (±{state["activation_strength"]["layer1"]:.3f})',
                                        fontsize=9)
                    ax_layer1.axis('off')

                    # Differential activation heatmap for layer 2
                    ax_layer2 = fig.add_subplot(gs[row+2, col])
                    diff_act_2 = state['diff_activations']['layer2'].reshape(-1)

                    size2 = int(np.sqrt(len(diff_act_2)))
                    if size2 * size2 < len(diff_act_2):
                        size2 += 1
                    padded2 = np.zeros(size2 * size2)
                    padded2[:len(diff_act_2)] = diff_act_2

                    im2 = ax_layer2.imshow(padded2.reshape(size2, size2), cmap='RdBu_r',
                                           vmin=-np.abs(diff_act_2).max(),
                                           vmax=np.abs(diff_act_2).max())
                    ax_layer2.set_title(f'L2 Diff (±{state["activation_strength"]["layer2"]:.3f})',
                                        fontsize=9)
                    ax_layer2.axis('off')

        # Add colorbar legend
        ax_legend = fig.add_subplot(gs[5, :])
        ax_legend.axis('off')
        ax_legend.text(0.5, 0.7, 'Red = Higher activation than average | Blue = Lower activation than average',
                      ha='center', va='center', fontsize=12,
                      bbox=dict(boxstyle="round,pad=0.3", facecolor='black', alpha=0.5))
        ax_legend.text(0.5, 0.3, 'Differential values show which neurons specialize for each digit',
                      ha='center', va='center', fontsize=11, style='italic')

        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='black')
        print(f"Digit activation analysis saved as {filename}")
        return fig

    def create_network_specialization_matrix(self, filename='network_specialization.png'):
        """Create a matrix showing which neurons are most specialized for each digit"""
        print(f"Creating network specialization matrix: {filename}")

        if not self.inference_states or 'diff_activations' not in self.inference_states[0]:
            print("No differential activation data available. Run record_inference() first.")
            return None

        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        fig.suptitle('Neural Network Specialization Matrix\n(Which neurons fire strongest for each digit)',
                     fontsize=16, fontweight='bold')

        # Collect all differential activations
        layer1_diffs = []
        layer2_diffs = []

        for digit in range(10):
            if digit in self.inference_states and 'diff_activations' in self.inference_states[digit]:
                layer1_diffs.append(self.inference_states[digit]['diff_activations']['layer1'].reshape(-1))
                layer2_diffs.append(self.inference_states[digit]['diff_activations']['layer2'].reshape(-1))

        if layer1_diffs and layer2_diffs:
            # Create matrices: rows = digits, columns = neurons
            layer1_matrix = np.array(layer1_diffs)
            layer2_matrix = np.array(layer2_diffs)

            # Plot Layer 1 specialization
            im1 = axes[0, 0].imshow(layer1_matrix, cmap='RdBu_r', aspect='auto')
            axes[0, 0].set_title('Layer 1 Neuron Specialization by Digit')
            axes[0, 0].set_xlabel('Neuron Index')
            axes[0, 0].set_ylabel('Digit')
            axes[0, 0].set_yticks(range(10))
            plt.colorbar(im1, ax=axes[0, 0])

            # Plot Layer 2 specialization
            im2 = axes[0, 1].imshow(layer2_matrix, cmap='RdBu_r', aspect='auto')
            axes[0, 1].set_title('Layer 2 Neuron Specialization by Digit')
            axes[0, 1].set_xlabel('Neuron Index')
            axes[0, 1].set_ylabel('Digit')
            axes[0, 1].set_yticks(range(10))
            plt.colorbar(im2, ax=axes[0, 1])

            # Compute and plot neuron specialization scores
            # (which digit each neuron responds to most strongly)
            layer1_specialization = np.argmax(np.abs(layer1_matrix), axis=0)
            layer2_specialization = np.argmax(np.abs(layer2_matrix), axis=0)

            # Histogram of which digits dominate each layer
            axes[1, 0].hist(layer1_specialization, bins=10, alpha=0.7, color='cyan', edgecolor='white')
            axes[1, 0].set_title('Layer 1: Distribution of Neuron Specialization')
            axes[1, 0].set_xlabel('Digit that activates neuron most')
            axes[1, 0].set_ylabel('Number of neurons')
            axes[1, 0].set_xticks(range(10))
            axes[1, 0].grid(alpha=0.2)

            axes[1, 1].hist(layer2_specialization, bins=10, alpha=0.7, color='magenta', edgecolor='white')
            axes[1, 1].set_title('Layer 2: Distribution of Neuron Specialization')
            axes[1, 1].set_xlabel('Digit that activates neuron most')
            axes[1, 1].set_ylabel('Number of neurons')
            axes[1, 1].set_xticks(range(10))
            axes[1, 1].grid(alpha=0.2)

        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='black')
        print(f"Network specialization matrix saved as {filename}")
        return fig

    def create_all_digits_inference_visualization(self, filename='all_digits_inference.png'):
        """Create a comprehensive visualization showing inference for all digits 0-9"""
        print(f"Creating all digits inference visualization: {filename}")

        fig = plt.figure(figsize=(24, 16))
        gs = GridSpec(5, 6, figure=fig, hspace=0.3, wspace=0.3)

        # Main weight visualization
        ax_weights = fig.add_subplot(gs[0:2, 0:2])

        # Individual digit visualizations (2 rows: input images and weight images)
        digit_input_axes = []
        digit_weight_axes = []
        for i in range(10):
            # Input images row
            row = 2
            col = i % 5
            if i >= 5:
                row = 3
            ax_input = fig.add_subplot(gs[row, col])
            digit_input_axes.append(ax_input)

            # Weight images row
            weight_row = 4
            weight_col = i % 6
            ax_weight = fig.add_subplot(gs[weight_row, weight_col])
            digit_weight_axes.append(ax_weight)

        # Weight distribution
        ax_dist = fig.add_subplot(gs[0, 2:4])

        # Overall accuracy
        ax_accuracy = fig.add_subplot(gs[1, 2:4])

        # Legend/info
        ax_info = fig.add_subplot(gs[0:2, 5])
        ax_info.axis('off')

        # Display final trained weights
        if self.weight_history:
            final_weights = self.weight_history[-1]['weights']
            weight_img = self.weights_to_image(final_weights)

            vmin, vmax = np.percentile(weight_img, [1, 99])
            vmax = max(abs(vmin), abs(vmax))
            vmin = -vmax

            im = ax_weights.imshow(weight_img, cmap='RdBu_r', vmin=vmin, vmax=vmax)
            ax_weights.set_title('Final Trained Network Weights', fontsize=12, fontweight='bold')
            ax_weights.axis('off')
            plt.colorbar(im, ax=ax_weights, fraction=0.046)

        # Plot each digit inference
        predictions = []
        confidences = []

        # Collect min/max for consistent color scaling across all weight visualizations
        all_weight_imgs = []
        for digit in range(10):
            if digit in self.inference_states:
                state = self.inference_states[digit]
                weight_img = self.weights_to_image(state['weights'])
                all_weight_imgs.append(weight_img)

        if all_weight_imgs:
            global_vmin = min(np.min(img) for img in all_weight_imgs)
            global_vmax = max(np.max(img) for img in all_weight_imgs)
            global_vmax = max(abs(global_vmin), abs(global_vmax))
            global_vmin = -global_vmax

        for digit in range(10):
            ax_input = digit_input_axes[digit]
            ax_weight = digit_weight_axes[digit]

            if digit in self.inference_states:
                state = self.inference_states[digit]

                # Show input image
                ax_input.imshow(state['input'].squeeze(), cmap='hot')

                pred = state['prediction']
                conf = state['probabilities'][pred]
                predictions.append(pred)
                confidences.append(conf)

                # Color code based on correctness
                color = 'green' if pred == digit else 'red'
                ax_input.set_title(f'True: {digit}\nPred: {pred} ({conf:.2%})',
                                  color=color, fontsize=10, fontweight='bold')
                ax_input.axis('off')

                # Add border to highlight incorrect predictions
                if pred != digit:
                    for spine in ax_input.spines.values():
                        spine.set_edgecolor('red')
                        spine.set_linewidth(3)

                # Show weight image for this digit's inference
                weight_img = self.weights_to_image(state['weights'])
                ax_weight.imshow(weight_img, cmap='RdBu_r', vmin=global_vmin, vmax=global_vmax)
                ax_weight.set_title(f'Weights for {digit}', fontsize=9)
                ax_weight.axis('off')

        # Weight distribution histogram
        if self.weight_history:
            all_weights = np.concatenate([
                final_weights['fc1'].flatten(),
                final_weights['fc2'].flatten(),
                final_weights['fc3'].flatten()
            ])

            ax_dist.hist(all_weights, bins=50, alpha=0.7, color='magenta', edgecolor='white')
            ax_dist.set_title('Final Weight Distribution', fontsize=12)
            ax_dist.set_xlabel('Weight Value')
            ax_dist.set_ylabel('Count')
            ax_dist.grid(alpha=0.2)

            ax_dist.axvline(np.mean(all_weights), color='cyan', linestyle='--',
                           label=f'Mean: {np.mean(all_weights):.3f}')
            ax_dist.axvline(np.median(all_weights), color='yellow', linestyle='--',
                           label=f'Median: {np.median(all_weights):.3f}')
            ax_dist.legend()

        # Accuracy bar chart
        correct_predictions = sum(1 for i, pred in enumerate(predictions) if i == pred)
        accuracy = correct_predictions / len(predictions) if predictions else 0

        colors = ['green' if i == predictions[i] else 'red'
                 for i in range(len(predictions))] if predictions else []

        if predictions:
            bars = ax_accuracy.bar(range(len(predictions)), confidences, color=colors)
            ax_accuracy.set_title(f'Prediction Confidence (Accuracy: {accuracy:.1%})', fontsize=12)
            ax_accuracy.set_xlabel('Digit')
            ax_accuracy.set_ylabel('Confidence')
            ax_accuracy.set_ylim(0, 1)
            ax_accuracy.set_xticks(range(10))
            ax_accuracy.grid(alpha=0.2, axis='y')

            # Add accuracy line
            ax_accuracy.axhline(y=accuracy, color='white', linestyle='--',
                              alpha=0.5, label=f'Overall Acc: {accuracy:.1%}')
            ax_accuracy.legend()

        # Info panel
        info_text = f"""Network Architecture:
Input: 784 neurons
Hidden 1: {self.hidden1_size} neurons
Hidden 2: {self.hidden2_size} neurons
Output: 10 classes

Training Summary:
Total Steps: {len(self.weight_history)}
Final Loss: {self.loss_history[-1]:.4f}
Final Accuracy: {self.accuracy_history[-1]:.2%}

Inference Results:
Correct: {correct_predictions}/10
Accuracy: {accuracy:.1%}
Avg Confidence: {np.mean(confidences):.2%}"""

        ax_info.text(0.1, 0.5, info_text, fontsize=10, verticalalignment='center',
                    family='monospace', bbox=dict(boxstyle="round,pad=0.3",
                    facecolor='black', alpha=0.5))

        plt.suptitle('Neural Network All-Digits Inference with Weight Visualization',
                    fontsize=16, fontweight='bold', y=0.98)

        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='black')
        print(f"All digits inference visualization saved as {filename}")

        return fig

    def create_gif(self, filename='weight_evolution.gif', fps=10, max_frames=100, include_all_inference=False):
        print(f"Creating GIF animation: {filename}")

        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Neural Network Weight Evolution', fontsize=16)

        axes = axes.flatten()

        weight_img = self.weights_to_image(self.weight_history[0]['weights'])
        # Use dynamic scaling for better visibility
        vmin, vmax = np.percentile(weight_img, [1, 99])
        vmax = max(abs(vmin), abs(vmax))
        vmin = -vmax
        im = axes[0].imshow(weight_img, cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[0].set_title('All Weights as Pixels')
        axes[0].axis('off')
        plt.colorbar(im, ax=axes[0], fraction=0.046)

        axes[1].set_xlim(0, len(self.loss_history))
        axes[1].set_ylim(0, max(self.loss_history) * 1.1)
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Training Loss')
        axes[1].grid(alpha=0.2)
        line_loss, = axes[1].plot([], [], 'cyan', linewidth=2)

        axes[2].set_xlim(0, len(self.accuracy_history))
        axes[2].set_ylim(0, 1)
        axes[2].set_xlabel('Step')
        axes[2].set_ylabel('Accuracy')
        axes[2].set_title('Test Accuracy')
        axes[2].grid(alpha=0.2)
        line_acc, = axes[2].plot([], [], 'green', linewidth=2)

        im_fc1 = axes[3].imshow(np.zeros((128, 784//128 + 1)), cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[3].set_title('FC1 Weights (reshaped)')
        axes[3].axis('off')

        im_fc2 = axes[4].imshow(np.zeros((64, 128//64 + 1)), cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[4].set_title('FC2 Weights (reshaped)')
        axes[4].axis('off')

        im_fc3 = axes[5].imshow(np.zeros((10, 64//10 + 1)), cmap='RdBu_r', vmin=vmin, vmax=vmax)
        axes[5].set_title('FC3 Weights (reshaped)')
        axes[5].axis('off')

        def animate(frame):
            if frame < len(self.weight_history):
                weight_data = self.weight_history[frame]

                weight_img = self.weights_to_image(weight_data['weights'])

                # Update color scaling dynamically
                vmin_frame, vmax_frame = np.percentile(weight_img, [1, 99])
                vmax_frame = max(abs(vmin_frame), abs(vmax_frame))
                vmin_frame = -vmax_frame
                im.set_clim(vmin=vmin_frame, vmax=vmax_frame)
                im.set_array(weight_img)

                fc1_reshaped = weight_data['weights']['fc1'][:128*(784//128)].reshape(128, -1)
                im_fc1.set_array(fc1_reshaped)
                im_fc1.set_clim(vmin=vmin_frame, vmax=vmax_frame)

                fc2_reshaped = weight_data['weights']['fc2'][:64*(128//64)].reshape(64, -1)
                im_fc2.set_array(fc2_reshaped)
                im_fc2.set_clim(vmin=vmin_frame, vmax=vmax_frame)

                fc3_reshaped = weight_data['weights']['fc3'][:10*(64//10)].reshape(10, -1)
                im_fc3.set_array(fc3_reshaped)
                im_fc3.set_clim(vmin=vmin_frame, vmax=vmax_frame)

                line_loss.set_data(range(frame+1), self.loss_history[:frame+1])
                line_acc.set_data(range(frame+1), self.accuracy_history[:frame+1])

                fig.suptitle(f'Weight Evolution - Step {weight_data["step"]}, Loss: {weight_data["loss"]:.3f}',
                           fontsize=16)

            elif frame < len(self.weight_history) + 10:
                digit_idx = frame - len(self.weight_history)
                if digit_idx in self.inference_states:
                    state = self.inference_states[digit_idx]
                    weight_img = self.weights_to_image(state['weights'])
                    im.set_array(weight_img)
                    fig.suptitle(f'Inference - Digit {digit_idx}, Prediction: {state["prediction"]}',
                               fontsize=16)

            elif include_all_inference and frame == len(self.weight_history) + 10:
                # Show all digits inference summary
                fig.suptitle('All Digits Inference Summary', fontsize=16)

                # Clear and reorganize axes for summary
                for ax in axes:
                    ax.clear()
                    ax.axis('off')

                # Show all 10 digits in a grid
                for digit in range(10):
                    if digit in self.inference_states:
                        row = digit // 5
                        col = digit % 5
                        if row < 2 and col < 3:  # Use available subplots
                            ax_idx = row * 3 + col
                            state = self.inference_states[digit]

                            axes[ax_idx].imshow(state['input'].squeeze(), cmap='hot')
                            pred = state['prediction']
                            color = 'green' if pred == digit else 'red'
                            axes[ax_idx].set_title(f'{digit}→{pred}', color=color, fontsize=10)
                            axes[ax_idx].axis('on')

            return [im, line_loss, line_acc, im_fc1, im_fc2, im_fc3]

        frames_to_use = min(len(self.weight_history) + 10, max_frames)
        if include_all_inference:
            frames_to_use = min(len(self.weight_history) + 11, max_frames)
        anim = FuncAnimation(fig, animate, frames=frames_to_use,
                           interval=1000/fps, blit=True)

        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer)
        print(f"GIF saved as {filename}")

        return anim

def main():
    import argparse

    parser = argparse.ArgumentParser(description='Visualize neural network weight evolution during training')
    parser.add_argument('--all-digits', action='store_true', default=True,
                       help='Create visualization showing inference for all digits 0-9 (default: True)')
    parser.add_argument('--epochs', type=int, default=3,
                       help='Number of epochs to train (default: 3)')
    parser.add_argument('--record-every', type=int, default=5,
                       help='Record weights every N steps (default: 5)')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for GIF animation (default: 10)')
    parser.add_argument('--max-frames', type=int, default=150,
                       help='Maximum frames for GIF animation (default: 150)')

    args = parser.parse_args()

    visualizer = WeightEvolutionVisualizer()

    print("Loading MNIST data...")
    visualizer.load_data()

    print(f"\nStarting training with weight recording (epochs={args.epochs}, record_every={args.record_every})...")
    visualizer.train_and_record(epochs=args.epochs, record_every=args.record_every)

    print("\nRecording inference states...")
    visualizer.record_inference()

    print("\nCreating interactive visualization...")
    fig = visualizer.create_interactive_animation()
    plt.savefig('weight_evolution_interactive.png', dpi=100, bbox_inches='tight')

    if args.all_digits:
        print("\nCreating all-digits inference visualization...")
        visualizer.create_all_digits_inference_visualization('all_digits_inference.png')
        print("- all_digits_inference.png (comprehensive inference visualization)")

        print("\nCreating digit-specific activation analysis...")
        visualizer.create_digit_activation_analysis('digit_activation_analysis.png')
        print("- digit_activation_analysis.png (digit-specific neuron activation patterns)")

        print("\nCreating network specialization matrix...")
        visualizer.create_network_specialization_matrix('network_specialization.png')
        print("- network_specialization.png (neuron specialization by digit)")

    print(f"\nGenerating animated GIF (fps={args.fps}, max_frames={args.max_frames})...")
    visualizer.create_gif('weight_evolution.gif', fps=args.fps, max_frames=args.max_frames,
                         include_all_inference=args.all_digits)

    print("\nVisualization complete!")
    print("Generated files:")
    print("- weight_evolution_interactive.png (interactive controls screenshot)")
    print("- weight_evolution.gif (animated GIF)")
    if args.all_digits:
        print("- all_digits_inference.png (all digits inference visualization)")
        print("- digit_activation_analysis.png (differential activation patterns)")
        print("- network_specialization.png (neuron specialization matrix)")

    plt.show()

if __name__ == "__main__":
    main()
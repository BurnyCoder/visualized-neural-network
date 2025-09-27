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

        with torch.no_grad():
            for digit in range(10):
                digit_samples = []
                for data, target in self.test_loader:
                    mask = target == digit
                    if mask.any():
                        sample = data[mask][0:1].to(self.device)

                        output = self.model(sample)
                        prediction = output.argmax(dim=1).item()
                        probs = F.softmax(output, dim=1).squeeze().cpu().numpy()

                        activations = {
                            'layer1': self.model.activations['layer1'].cpu().numpy(),
                            'layer2': self.model.activations['layer2'].cpu().numpy(),
                            'output': probs
                        }

                        weights = self.capture_weights()

                        self.inference_states[digit] = {
                            'input': sample.cpu().numpy(),
                            'weights': weights,
                            'activations': activations,
                            'prediction': prediction,
                            'probabilities': probs
                        }
                        break

        print(f"Recorded inference states for {len(self.inference_states)} digits.")

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

    def create_gif(self, filename='weight_evolution.gif', fps=10, max_frames=100):
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

            return [im, line_loss, line_acc, im_fc1, im_fc2, im_fc3]

        frames_to_use = min(len(self.weight_history) + 10, max_frames)
        anim = FuncAnimation(fig, animate, frames=frames_to_use,
                           interval=1000/fps, blit=True)

        writer = PillowWriter(fps=fps)
        anim.save(filename, writer=writer)
        print(f"GIF saved as {filename}")

        return anim

def main():
    visualizer = WeightEvolutionVisualizer()

    print("Loading MNIST data...")
    visualizer.load_data()

    print("\nStarting training with weight recording...")
    visualizer.train_and_record(epochs=3, record_every=5)

    print("\nRecording inference states...")
    visualizer.record_inference()

    print("\nCreating interactive visualization...")
    fig = visualizer.create_interactive_animation()
    plt.savefig('weight_evolution_interactive.png', dpi=100, bbox_inches='tight')

    print("\nGenerating animated GIF...")
    visualizer.create_gif('weight_evolution.gif', fps=10, max_frames=150)

    print("\nVisualization complete!")
    print("Generated files:")
    print("- weight_evolution_interactive.png (interactive controls screenshot)")
    print("- weight_evolution.gif (animated GIF)")

    plt.show()

if __name__ == "__main__":
    main()
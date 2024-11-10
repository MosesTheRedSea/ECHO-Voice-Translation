import torch
import os
import matplotlib.pyplot as plt

# Paths
checkpoint_dir = "/home/hice1/madewolu9/scratch/madewolu9/Voice-Based-Language-Translation-System/models/Text_Text/GRU/Checkpoints"
evaluation_dir = "/home/hice1/madewolu9/scratch/madewolu9/Voice-Based-Language-Translation-System/models/Text_Text/GRU/Evaluation"

# Initialize lists for metrics across epochs
epochs = []
train_losses = []
bleu_scores = []
f1_scores = []

# Load each checkpoint and aggregate metrics
for epoch in range(5):
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}.pt")
    checkpoint = torch.load(checkpoint_path)
    
    # Append metrics for each epoch
    epochs.append(epoch)
    train_losses.append(checkpoint['train_losses'][-1])  # Get the latest loss in each checkpoint
    bleu_scores.append(checkpoint['bleu_scores'][-1])    # Get the latest BLEU score in each checkpoint
    f1_scores.append(checkpoint['f1_scores'][-1])        # Get the latest F1 score in each checkpoint

# Plot the metrics
plt.figure(figsize=(15, 5))

# Training Loss Plot
plt.subplot(1, 3, 1)
plt.plot(epochs, train_losses, marker='o', color='r')
plt.title('Training Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')

# BLEU Score Plot
plt.subplot(1, 3, 2)
plt.plot(epochs, bleu_scores, marker='o', color='b')
plt.title('BLEU Score per Epoch')
plt.xlabel('Epoch')
plt.ylabel('BLEU Score')

# F1 Score Plot
plt.subplot(1, 3, 3)
plt.plot(epochs, f1_scores, marker='o', color='g')
plt.title('F1 Score per Epoch')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')

# Save the plot
os.makedirs(evaluation_dir, exist_ok=True)
output_path = os.path.join(evaluation_dir, 'metrics_per_epoch.png')
plt.tight_layout()
plt.savefig(output_path)
plt.show()

print(f"Plot saved to {output_path}")

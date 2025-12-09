import json
import matplotlib.pyplot as plt
import pandas as pd

# Load training history for 3-class
with open(r'd:\Skripsi\sentiment-analyst-ojol-review\results_3class\checkpoint-2600\trainer_state.json', 'r') as f:
    data = json.load(f)

log_history = data['log_history']

# Extract train and eval metrics
train_loss = [(entry['step'], entry['loss']) for entry in log_history if 'loss' in entry and 'eval_loss' not in entry]
eval_metrics = [(entry['step'], entry['eval_accuracy'], entry['eval_loss']) for entry in log_history if 'eval_accuracy' in entry]

# Create DataFrame
train_df = pd.DataFrame(train_loss, columns=['step', 'train_loss'])
eval_df = pd.DataFrame(eval_metrics, columns=['step', 'eval_accuracy', 'eval_loss'])

print("=== TRAINING SUMMARY (3-Class) ===\n")
print(f"Best Model Checkpoint: {data['best_model_checkpoint']}")
print(f"Best Validation Accuracy: {data['best_metric']:.4f} (at step {data['best_global_step']})")
print(f"Final Epoch: {data['epoch']}")
print(f"Total Training Steps: {data['global_step']}")
print(f"Early Stopping Patience Counter: {data['stateful_callbacks']['EarlyStoppingCallback']['attributes']['early_stopping_patience_counter']}")

print("\n=== VALIDATION METRICS PER EPOCH ===\n")
print(eval_df.to_string(index=False))

# Plot Training & Validation Loss
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Loss plot
ax1.plot(train_df['step'], train_df['train_loss'], label='Train Loss', alpha=0.7)
ax1.plot(eval_df['step'], eval_df['eval_loss'], label='Val Loss', marker='o', linewidth=2)
ax1.axvline(x=data['best_global_step'], color='red', linestyle='--', label=f'Best Model (step {data["best_global_step"]})')
ax1.set_xlabel('Steps')
ax1.set_ylabel('Loss')
ax1.set_title('Training & Validation Loss (3-Class)')
ax1.legend()
ax1.grid(True)

# Accuracy plot
ax2.plot(eval_df['step'], eval_df['eval_accuracy'], label='Val Accuracy', marker='o', linewidth=2, color='green')
ax2.axvline(x=data['best_global_step'], color='red', linestyle='--', label=f'Best Model (step {data["best_global_step"]})')
ax2.axhline(y=data['best_metric'], color='orange', linestyle=':', label=f'Best Acc: {data["best_metric"]:.4f}')
ax2.set_xlabel('Steps')
ax2.set_ylabel('Accuracy')
ax2.set_title('Validation Accuracy (3-Class)')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.savefig('training_history_3class.png', dpi=150)
plt.show()

print("\n✅ Training history plot saved as 'training_history_3class.png'")

"""
KONFIGURASI AGRESIF untuk meningkatkan akurasi dari 46% ke >80%
Masalah: Model underfitting - tidak belajar dengan baik
Solusi: Learning rate lebih tinggi, dropout lebih rendah, training lebih lama
"""

# ============================================================================
# CONFIGURATION 1: AGGRESSIVE LEARNING (Rekomendasi Utama)
# ============================================================================
training_args = TrainingArguments(
    output_dir=MODEL_SAVE_PATH,
    
    # Training parameters - AGGRESSIVE!
    num_train_epochs=30,  # Naikkan ke 30 epoch
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=2,
    
    # Learning rate - LEBIH TINGGI!
    learning_rate=5e-5,  # NAIKKAN dari 3e-5 ke 5e-5 (critical!)
    warmup_ratio=0.15,  # Naikkan warmup untuk stabilitas
    weight_decay=0.005,  # Kurangi regularization
    
    # Evaluation & saving
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=5,
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    
    # Logging
    logging_dir=f"{MODEL_SAVE_PATH}/logs",
    logging_strategy="epoch",
    report_to="tensorboard",
    
    # Optimization
    fp16=torch.cuda.is_available(),
    dataloader_num_workers=2,
    max_grad_norm=1.0,
    
    # Reproducibility
    seed=42,
    
    push_to_hub=False,
    disable_tqdm=False
)

# Model dengan dropout LEBIH RENDAH
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    problem_type="single_label_classification",
    hidden_dropout_prob=0.1,  # TURUNKAN dari 0.2 ke 0.1
    attention_probs_dropout_prob=0.1  # TURUNKAN dari 0.2 ke 0.1
)

# Early Stopping dengan patience LEBIH BESAR
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=8)]  # Naikkan ke 8
)

print("✓ AGGRESSIVE Configuration Loaded")
print("\n🚀 Key Changes:")
print("  • Learning Rate: 5e-5 (was 3e-5) - CRITICAL CHANGE!")
print("  • Dropout: 0.1 (was 0.2) - Less regularization")
print("  • Epochs: 30 (was 20)")
print("  • Early Stopping Patience: 8 (was 5)")
print("  • Weight Decay: 0.005 (was 0.01)")
print("  • Warmup: 15% (was 10%)")
print("\n🎯 Expected: Akurasi >70% (hopefully >80%)")

# ============================================================================
# ALTERNATIVE: FREEZE BOTTOM LAYERS (Jika masih gagal)
# ============================================================================
"""
# Freeze bottom 6 layers untuk fokus training di top layers
for i, layer in enumerate(model.bert.encoder.layer):
    if i < 6:  # Freeze layer 0-5
        for param in layer.parameters():
            param.requires_grad = False

print("✓ Bottom 6 layers frozen - only training top 6 layers")
"""

# ============================================================================
# ALTERNATIVE: INCREASE BATCH SIZE
# ============================================================================
"""
# Jika GPU memory cukup, coba:
training_args = TrainingArguments(
    ...
    per_device_train_batch_size=32,  # Naikkan dari 16
    gradient_accumulation_steps=1,   # Turunkan dari 2
    ...
)
"""

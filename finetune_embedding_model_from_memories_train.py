import pandas as pd
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


import os
os.environ["WANDB_DISABLED"] = "true"
# --- Load training data ---
csv_file = "embedding_training_data.csv"
df = pd.read_csv(csv_file)

# --- Convert CSV rows into InputExample objects ---
train_examples = []
for _, row in df.iterrows():
    text = str(row['text'])
    tags = str(row['tags'])
    train_examples.append(InputExample(texts=[text, tags], label=1.0))  # label=1.0 for positive pairs

# --- Create DataLoader ---
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)

# --- Load pretrained MiniLM model ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Define training loss ---
train_loss = losses.CosineSimilarityLoss(model)

# --- Fine-tune the model ---
model.fit(
    train_objectives=[(train_dataloader, train_loss)],
    epochs=3,              # Adjust epochs based on dataset size
    warmup_steps=100       # Optional warmup
)

# --- Save fine-tuned model ---
output_path = "custom-MiniLM"
model.save(output_path)
print(f"Fine-tuned model saved to {output_path}")

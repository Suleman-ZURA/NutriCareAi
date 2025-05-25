import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import random
import os
from tabulate import tabulate

# ===== Step 1: Setup =====
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# ===== Step 2: Load and Prepare Data =====
csv_path = "nutricare_dataset.csv"

if not os.path.exists(csv_path):
    print(f"❌ ERROR: '{csv_path}' not found in this folder.")
    exit()

# Reading CSV, skipping bad lines to avoid errors
df = pd.read_csv(csv_path, on_bad_lines='skip')

# Filter cause and treat flags to int (in case of strings)
df['is_cause'] = df['is_cause'].astype(int)
df['is_treat'] = df['is_treat'].astype(int)

# Remove rows with missing food or disease entities
df.dropna(subset=['food_entity', 'disease_entity'], inplace=True)

# Encode food and disease entities to integers
food_vocab = {name: idx for idx, name in enumerate(df['food_entity'].unique())}
disease_vocab = {name: idx for idx, name in enumerate(df['disease_entity'].unique())}
df['food_id'] = df['food_entity'].map(food_vocab)
df['disease_id'] = df['disease_entity'].map(disease_vocab)

# Positive samples (cause = 1)
positive_samples = [(f, d, 1) for f, d in zip(df.loc[df['is_cause'] == 1, 'food_id'], df.loc[df['is_cause'] == 1, 'disease_id'])]

# Negative samples (random food-disease pairs not in cause)
foods = list(food_vocab.values())
diseases = list(disease_vocab.values())
negative_samples = []

while len(negative_samples) < len(positive_samples):
    f = random.choice(foods)
    d = random.choice(diseases)
    if not ((df['food_id'] == f) & (df['disease_id'] == d) & (df['is_cause'] == 1)).any():
        negative_samples.append((f, d, 0))

# Combine and shuffle
all_samples = positive_samples + negative_samples
random.shuffle(all_samples)

# ===== Step 3: Train/Test Split =====
X_train, X_test = train_test_split(all_samples, test_size=0.2)

# ===== Step 4: Define Model =====
class NutriCareModel(nn.Module):
    def __init__(self, food_size, disease_size, emb_dim=32):
        super().__init__()
        self.food_emb = nn.Embedding(food_size, emb_dim)
        self.disease_emb = nn.Embedding(disease_size, emb_dim)
        self.fc = nn.Linear(emb_dim * 2, 1)

    def forward(self, food_ids, disease_ids):
        food_vec = self.food_emb(food_ids)
        disease_vec = self.disease_emb(disease_ids)
        x = torch.cat([food_vec, disease_vec], dim=1)
        return torch.sigmoid(self.fc(x)).squeeze(1)

# ===== Step 5: Training =====
model = NutriCareModel(len(food_vocab), len(disease_vocab)).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()

print("🚀 Training model...")
for epoch in range(50):
    total_loss = 0.0
    model.train()
    for f, d, label in X_train:
        f_tensor = torch.tensor([f], device=device)
        d_tensor = torch.tensor([d], device=device)
        label_tensor = torch.tensor([label], dtype=torch.float32, device=device)

        output = model(f_tensor, d_tensor)
        loss = loss_fn(output, label_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"🌀 Epoch {epoch+1}/10 - Loss: {total_loss:.4f}")

# ===== Step 6: Prepare sentence maps =====
cause_map = {}
treat_map = {}
for _, row in df.iterrows():
    f_id = row['food_id']
    d_id = row['disease_id']
    sent = row['sentence']
    if row['is_cause'] == 1:
        cause_map[(f_id, d_id)] = sent
    if row['is_treat'] == 1:
        treat_map[(f_id, d_id)] = sent

# ===== Step 7: Recommendation function =====
def recommend_foods(disease_name, top_k=5):
    if disease_name not in disease_vocab:
        print("❗ Disease not found in the dataset.")
        return [], []

    d_idx = disease_vocab[disease_name]
    f_idx = torch.tensor(list(food_vocab.values()), device=device)
    d_idx_tensor = torch.tensor([d_idx]*len(food_vocab), device=device)

    model.eval()
    with torch.no_grad():
        scores = model(f_idx, d_idx_tensor)  # shape: [num_foods]

    # Foods to avoid (cause) - highest scores
    top_cause_scores, top_cause_indices = torch.topk(scores, top_k)

    # Foods that treat disease (from treat_map)
    treat_foods = [f for f in food_vocab.values() if (f, d_idx) in treat_map]
    treat_scores = scores[torch.tensor(treat_foods, device=device)] if treat_foods else torch.tensor([])
    if len(treat_foods) > 0:
        top_treat_scores, top_treat_indices = torch.topk(treat_scores, min(top_k, len(treat_foods)))
        top_treat_foods = [treat_foods[i.item()] for i in top_treat_indices]
    else:
        top_treat_foods = []

    id_to_food = {v: k for k, v in food_vocab.items()}

    cause_table = []
    for idx in top_cause_indices:
        food_id = idx.item()
        food_name = id_to_food[food_id]
        sent = cause_map.get((food_id, d_idx), "No explanation available")
        cause_table.append([food_name, sent])

    treat_table = []
    for food_id in top_treat_foods:
        food_name = id_to_food[food_id]
        sent = treat_map.get((food_id, d_idx), "No explanation available")
        treat_table.append([food_name, sent])

    return cause_table, treat_table

# ===== Step 8: CLI Loop =====
print("\n✅ NutriCareAI is ready!")
print("Type the name of a disease (e.g., diabetes, hypertension) to see foods to avoid and foods to eat.")
print("Type 'exit' to quit.\n")

while True:
    disease = input("🩺 Enter disease: ").strip().lower()
    if disease == "exit":
        print("👋 Goodbye!")
        break
    cause_table, treat_table = recommend_foods(disease)
    if cause_table:
        print(f"\n⚠️ Foods TO AVOID for {disease}:")
        # Display only the food names without explanations
        print(tabulate([[food] for food, _ in cause_table], headers=["Food"], tablefmt="fancy_grid"))
    else:
        print("No foods to avoid found.")

    if treat_table:
        print(f"\n✅ Foods GOOD FOR {disease}:")
        # Display food names with explanations
        print(tabulate(treat_table, headers=["Food", "Explanation"], tablefmt="fancy_grid"))
    else:
        print("No good foods found.")
    print("\n")

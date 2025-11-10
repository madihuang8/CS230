# pip install -U sentence-transformers datasets torch pandas

from datasets import load_dataset
from sentence_transformers import SentenceTransformer, InputExample, losses, evaluation, util
from torch.utils.data import DataLoader
import pandas as pd

# 1) Load STSB (already split & score in [0,1])
ds = load_dataset("sentence-transformers/stsb")
train = ds["train"]; valid = ds["validation"]; test = ds["test"]

# 2) Build InputExamples (NO extra normalization)
train_ex = [InputExample(texts=[s1, s2], label=float(lbl))
            for s1, s2, lbl in zip(train["sentence1"], train["sentence2"], train["score"])]
valid_ex = [InputExample(texts=[s1, s2], label=float(lbl))
            for s1, s2, lbl in zip(valid["sentence1"], valid["sentence2"], valid["score"])]

# 3) Dataloader + loss
train_loader = DataLoader(train_ex, batch_size=32, shuffle=True)
model = SentenceTransformer("all-MiniLM-L6-v2")
loss_fn = losses.CosineSimilarityLoss(model)   # trains embeddings so cos_sim ~ gold score

# 4) Validation evaluator (expects lists, not Column objects)
val_eval = evaluation.EmbeddingSimilarityEvaluator(
    list(valid["sentence1"]), list(valid["sentence2"]), list(valid["score"])
)

# 5) Train
model.fit(
    train_objectives=[(train_loader, loss_fn)],
    evaluator=val_eval,
    epochs=4,
    warmup_steps=int(0.1 * len(train_loader)),
    output_path="models/sts-miniLM-stsb",
)

# 6) Reload best and evaluate on TEST
model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
test_eval = evaluation.EmbeddingSimilarityEvaluator(
    list(test["sentence1"]), list(test["sentence2"]), list(test["score"])
)
spearman = test_eval(model)  # Spearman correlation (higher is better)
print("Test Spearman:", spearman)

# 7) Create a table of gold vs predicted for each test row
rows = []
for s1, s2, gold in zip(test["sentence1"], test["sentence2"], test["score"]):
    emb = model.encode([s1, s2], convert_to_tensor=True)
    raw_cos = util.cos_sim(emb[0], emb[1]).item()   # Cosine ∈ [-1, 1]
    pred = (raw_cos + 1) / 2   # rescale to 0–1
    rows.append({
        "sentence1": s1,
        "sentence2": s2,
        "gold_similarity": float(gold),          # stays in [0,1]
        "predicted_similarity": float(pred)      # may be negative (cosine)
    })

df = pd.DataFrame(rows)
df.to_csv("test_similarity_output.csv", index=False)
print(df.head())

# 8) Single example usage (inference)
a = "The cat sat on the mat."
b = "A feline rested on the rug."
emb = model.encode([a, b], convert_to_tensor=True)
raw_cos = util.cos_sim(emb[0], emb[1])   # Cosine ∈ [-1, 1]
pred = (raw_cos + 1) / 2   # rescale to 0–1
print("Example cosine similarity:", pred)

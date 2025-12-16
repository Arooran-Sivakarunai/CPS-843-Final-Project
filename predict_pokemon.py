import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from numpy.linalg import norm


# =============================
#   Encoder definition
# =============================
class TinyEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 56 -> 28
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 28 -> 14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 14 -> 7
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


# =============================
#   Image transform
# =============================
transform = transforms.Compose([
    transforms.Resize((56, 56)),
    transforms.ToTensor(),
])


# =============================
#   Load Encoder
# =============================
def load_encoder(path="encoder.pth", latent_dim=64):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    enc = TinyEncoder(latent_dim)
    enc.load_state_dict(torch.load(path, map_location=device))
    enc.to(device)
    enc.eval()
    return enc, device


# =============================
#   Embed an image
# =============================
def embed_image(path, encoder, device):
    img = Image.open(path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        z = encoder(x).cpu().numpy()[0]

    return z


# =============================
#   Build embedding database
# =============================
def build_database(folder, encoder, device):
    db = {}
    for fname in os.listdir(folder):
        if fname.lower().endswith((".png", ".jpg", ".jpeg")):
            label = os.path.splitext(fname)[0]  # filename without extension
            full_path = os.path.join(folder, fname)
            db[label] = embed_image(full_path, encoder, device)
    return db


# =============================
#   Nearest-neighbor search
# =============================
def predict(test_img_path, database, encoder, device, metric="l2"):
    test_emb = embed_image(test_img_path, encoder, device)

    best_label = None
    best_score = float("inf")

    for label, emb in database.items():
        if metric == "cosine":
            score = 1 - np.dot(test_emb, emb) / (norm(test_emb) * norm(emb))
        else:
            score = norm(test_emb - emb)  # L2 default

        if score < best_score:
            best_score = score
            best_label = label

    return best_label, best_score


# =============================
#   Main
# =============================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="path to the test image (e.g. pikachu.png)")
    parser.add_argument("--db", default="sprites", help="folder containing known Pokémon sprites")
    parser.add_argument("--encoder", default="encoder.pth", help="trained encoder path")
    parser.add_argument("--metric", choices=["l2", "cosine"], default="l2")
    args = parser.parse_args()

    # Load model + build database
    print("Loading encoder...")
    encoder, device = load_encoder(args.encoder)

    print("Building Pokémon embedding database...")
    database = build_database(args.db, encoder, device)
    print(f"Loaded {len(database)} sprites.")

    # Predict
    print(f"\nPredicting for: {args.image}")
    
    label, score = predict(args.image, database, encoder, device, metric=args.metric)

    print("\n===============================")
    print(f" Predicted Pokémon:  {label}")
    print(f" Distance score:     {score:.4f}")
    print("===============================")

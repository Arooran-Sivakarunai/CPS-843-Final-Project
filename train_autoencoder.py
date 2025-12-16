import os
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class SpriteDataset(Dataset):
    def __init__(self, folder):
        self.paths = []
        for fname in os.listdir(folder):
            if fname.lower().endswith((".png", ".jpg", ".jpeg")):
                self.paths.append(os.path.join(folder, fname))

        self.transform = transforms.Compose([
            transforms.Resize((56, 56)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)

class TinyEncoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),  # 56 → 28
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), # 28 → 14
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), # 14 → 7
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(64 * 7 * 7, latent_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, 1)
        z = self.fc(x)
        return z


class TinyDecoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 64 * 7 * 7)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Output pixels in [0,1]
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, 64, 7, 7)
        x = self.decoder(x)
        return x


class Autoencoder(nn.Module):
    def __init__(self, latent_dim=64):
        super().__init__()
        self.enc = TinyEncoder(latent_dim)
        self.dec = TinyDecoder(latent_dim)

    def forward(self, x):
        z = self.enc(x)
        out = self.dec(z)
        return out, z


def train_autoencoder(
    sprite_folder="sprites",
    latent_dim=64,
    epochs=200,
    batch_size=32,
    lr=1e-3,
    save_path="encoder.pth"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = SpriteDataset(sprite_folder)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = Autoencoder(latent_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for imgs in loader:
            imgs = imgs.to(device)

            optimizer.zero_grad()
            recon, _ = model(imgs)
            loss = criterion(recon, imgs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{epochs}  Loss: {avg_loss:.6f}")

    torch.save(model.enc.state_dict(), save_path)
    print(f"Encoder saved to {save_path}")

if __name__ == "__main__":
    train_autoencoder(
        sprite_folder="sprites",
        latent_dim=64,
        epochs=200,
        batch_size=32,
        lr=1e-3,
        save_path="encoder.pth",
    )


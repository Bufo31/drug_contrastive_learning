import torch

from data import load_data
from model import DualEncoder
from loss import contrastive_loss


def train(epochs = 20, batch = 32, lr = 0.001):
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")

    print("device: ", device)

    train_loader, val_loader, test_loader = load_data(batch)

    x_mol_batch, x_ph_batch = next(iter(train_loader))

    mol_dim = x_mol_batch.shape[1]
    ph_dim = x_ph_batch.shape[1]

    print("mol_dim: ", mol_dim)
    print("ph_dim: ", ph_dim)

    model = DualEncoder(mol_dim, ph_dim)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr = lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for x_mol, x_ph in train_loader:
            x_mol = x_mol.to(device)
            x_ph = x_ph.to(device)

            optimizer.zero_grad()

            z_mol, z_ph = model(x_mol, x_ph)

            loss = contrastive_loss(z_mol, z_ph)

            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        print(f"epoch: {epoch+1}/{epochs}, avg_loss:", avg_loss)

    return model

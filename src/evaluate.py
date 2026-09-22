import torch

from data import load_data
from model import DualEncoder

def evaluate(batch_size=32):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, test_loader = load_data(batch_size)

    x_mol_batch, x_ph_batch = next(iter(test_loader))

    mol_dim = x_mol_batch.shape[1]
    ph_dim = x_ph_batch.shape[1]

    model = DualEncoder(mol_dim, ph_dim).to(device)

    model.load_state_dict(torch.load("best_model.pt",map_location=device))

    model.eval()

    all_z_mol = []
    all_z_ph = []

    with torch.no_grad():
        for x_mol, x_ph in test_loader:
            x_mol = x_mol.to(device)
            x_ph = x_ph.to(device)

            z_mol, z_ph = model(x_mol, x_ph)

            all_z_mol.append(z_mol)
            all_z_ph.append(z_ph)

        all_z_mol = torch.cat(all_z_mol, dim=0)
        all_z_ph = torch.cat(all_z_ph, dim=0)

        similarity = all_z_mol @ all_z_ph.T

        for k in [1, 5, 10]:
            correct = 0
            for i in range(similarity.shape[0]):
                scores = similarity[i]
                topk_indices = torch.topk(scores, k=k).indices
                if (topk_indices == i).any():
                    correct += 1
            recall = correct / similarity.shape[0]
            print(f"Mol → Ph Recall@{k}: {recall:.4f}")

        for k in [1, 5, 10]:
            correct = 0
            for j in range(similarity.shape[1]):
                scores = similarity[:, j]
                topk_indices = torch.topk(scores, k=k).indices
                if (topk_indices == j).any():
                    correct += 1
            recall = correct / similarity.shape[1]
            print(f"Ph → Mol Recall@{k}: {recall:.4f}")
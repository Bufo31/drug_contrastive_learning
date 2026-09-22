import torch.nn as nn
import torch.nn.functional as F

Embedding_dim = 128
Hidden_dim = 512

class MLPEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=Hidden_dim, output_dim=Embedding_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        z = self.net(x)
        z = F.normalize(z,dim=1)
        return z

class DualEncoder(nn.Module):
    def __init__(self, mol_dim, ph_dim, hidden_dim = Hidden_dim, output_dim=Embedding_dim):
        super().__init__()

        self.mol_encoder = MLPEncoder(mol_dim, hidden_dim, output_dim)
        self.ph_encoder = MLPEncoder(ph_dim, hidden_dim, output_dim)

    def forward(self, x_mol, x_ph):
        z_mol = self.mol_encoder(x_mol)
        z_ph = self.ph_encoder(x_ph)

        return z_mol, z_ph

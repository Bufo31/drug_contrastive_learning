import torch
import torch.nn.functional as F

def contrastive_loss(z_mol,z_ph,temperature = 0.07):

    similarity = z_mol @ z_ph.T
    similarity = similarity / temperature

    labels = torch.arange(z_mol.shape[0],device = z_mol.device)

    loss_mol = F.cross_entropy(similarity,labels)

    loss_ph = F.cross_entropy(similarity.T,labels)

    loss = (loss_mol + loss_ph)/2

    return loss

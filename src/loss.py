import torch
import torch.nn.functional as F

def contrastive_loss(z_mol,z_ph,temperature = 0.05):

    similarity = z_mol @ z_ph.T
    similarity = similarity / temperature

    labels = torch.arange(z_mol.shape[0],device = z_mol.device)

    loss_mol = F.cross_entropy(similarity,labels)

    loss_ph = F.cross_entropy(similarity.T,labels)

    loss = (loss_mol + loss_ph)/2

    return loss

def phenotype_aware_loss(
        z_mol,
        z_ph,
        x_ph,
        temperature=0.05,
        alpha=0.25,
        gamma=2):

    similarity = z_mol @ z_ph.T
    similarity = similarity / temperature

    batch_size = z_mol.shape[0]
    labels = torch.arange(batch_size, device=z_mol.device)

    with torch.no_grad():

        # 计算 phenotype similarity
        x_ph_norm = F.normalize(x_ph, dim=1)
        ph_similarity = x_ph_norm @ x_ph_norm.T

        # 只利用正的 phenotype similarity
        positive_similarity = torch.clamp(ph_similarity, min=0)

        # phenotype 越相似，negative weight 越小
        weights = 1 - alpha * positive_similarity ** gamma

        # 对角线是正确 positive pair，不进行降权
        weights.fill_diagonal_(1.0)

    weighted_similarity = similarity + torch.log(weights + 1e-8)

    loss_mol = F.cross_entropy(weighted_similarity, labels)
    loss_ph = F.cross_entropy(weighted_similarity.T, labels)

    loss = (loss_mol + loss_ph) / 2

    return loss


def tanimoto_similarity(x):

    intersection = x @ x.T

    bit_count = x.sum(dim=1, keepdim=True)

    union = bit_count + bit_count.T - intersection

    similarity = intersection / (union + 1e-8)

    return similarity

def structure_aware_loss(
        z_mol,
        z_ph,
        x_mol,
        temperature=0.05,
        alpha=0.25,
        gamma=2
):

    # 1. molecule embedding 和 phenotype embedding 的相似度
    similarity = z_mol @ z_ph.T
    similarity = similarity / temperature

    batch_size = z_mol.shape[0]

    labels = torch.arange(
        batch_size,
        device=z_mol.device
    )

    # 2. 根据 Morgan Fingerprint 计算结构相似度
    with torch.no_grad():

        structure_similarity = tanimoto_similarity(x_mol)

        # 结构越相似，negative 权重越小
        weights = 1 - alpha * structure_similarity ** gamma

        # 自己和自己的正确配对不能被降权
        weights.fill_diagonal_(1.0)

    # 3. 对 negative logits 进行降权
    weighted_similarity = (
        similarity
        + torch.log(weights + 1e-8)
    )

    # 4. 双向 contrastive loss
    loss_mol = F.cross_entropy(
        weighted_similarity,
        labels
    )

    loss_ph = F.cross_entropy(
        weighted_similarity.T,
        labels
    )

    loss = (loss_mol + loss_ph) / 2

    return loss
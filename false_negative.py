import numpy as np
import torch
import torch.nn.functional as F

from src.model import DualEncoder


def false_negative_analysis():

    data = np.load("data/processed/paired_data.npz")

    compound_ids = data["compound_ids"]
    fingerprints = data["fingerprints"]
    phenotypes = data["phenotypes"]

    np.random.seed(31)
    sample_size = 5000

    indices = np.random.choice(len(compound_ids), size=sample_size, replace=False)

    sampled_ids = compound_ids[indices]
    sampled_ph = torch.tensor(phenotypes[indices], dtype=torch.float32)


    # 计算不同 compound 之间的 phenotype similarity。检查是否存在 phenotype 很相似的 non-matching pairs
    sampled_ph = F.normalize(sampled_ph, dim=1)

    ph_similarity = sampled_ph @ sampled_ph.T

    mask = torch.triu(torch.ones(sample_size, sample_size, dtype=torch.bool), diagonal=1)

    pair_similarities = ph_similarity[mask]


    # 查看 phenotype similarity 的整体分布
    print("Phenotype similarity:")

    print("mean:", pair_similarities.mean().item())
    print("median:", pair_similarities.median().item())
    print("90% percentile:", torch.quantile(pair_similarities, 0.90).item())
    print("95% percentile:", torch.quantile(pair_similarities, 0.95).item())
    print("99% percentile:", torch.quantile(pair_similarities, 0.99).item())

    for threshold in [0.7, 0.8, 0.9]:
        count = (pair_similarities > threshold).sum().item()
        print(f"similarity > {threshold}:", count)


    # 找出 phenotype similarity 最高的 Top 10 pair
    pair_indices = torch.nonzero(mask)

    top_values, top_positions = torch.topk(pair_similarities, k=10)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DualEncoder(2048, 737).to(device)

    model.load_state_dict(torch.load("best_model.pt", map_location=device))

    model.eval()


    sampled_mol = torch.tensor(fingerprints[indices], dtype=torch.float32).to(device)
    sampled_ph_model = torch.tensor(phenotypes[indices], dtype=torch.float32).to(device)

    with torch.no_grad():
        z_mol = model.mol_encoder(sampled_mol)
        z_ph = model.ph_encoder(sampled_ph_model)

        model_similarity = z_mol @ z_ph.T

    model_similarity = model_similarity.to(device)


    # 输出 phenotype 最相似的 Top 10 pair，并将两个方向的 model similarity 取平均
    print("\nTop 10 pairs:")

    for k in range(10):
        i, j = pair_indices[top_positions[k]]

        i = i.item()
        j = j.item()

        model_sim = (model_similarity[i, j] + model_similarity[j, i]) / 2

        print(k + 1, sampled_ids[i], sampled_ids[j], "phenotype:", top_values[k].item(), "model:", model_sim.item())


    # 统计普通 negative 的 model similarity 作为参考
    negative_mask = ~torch.eye(sample_size, dtype=torch.bool)

    negative_similarities = model_similarity[negative_mask]

    print("\nNegative model similarity:")

    print("mean:", negative_similarities.mean().item())
    print("95% percentile:", torch.quantile(negative_similarities, 0.95).item())
    print("99% percentile:", torch.quantile(negative_similarities, 0.99).item())


    # 检查 phenotype 越相似时，model similarity 是否也越来越高
    model_pair_similarity = (model_similarity + model_similarity.T) / 2

    model_pair_similarities = model_pair_similarity[mask]

    print("\nPhenotype similarity vs model similarity:")

    for threshold in [0.3, 0.5, 0.7, 0.8, 0.9]:
        selected = pair_similarities > threshold

        print(f"phenotype > {threshold}:",
              "count =", selected.sum().item(),
              "mean model similarity =", model_pair_similarities[selected].mean().item())


false_negative_analysis()

#%%
"""
Minimal Tox21 Liver Toxicity Analysis (NR-AhR task)
- Dataset: ~7823 molecules, 12 toxicity tasks
- Focus: NR-AhR (Aryl hydrocarbon receptor, liver toxicity)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch_geometric.datasets import MoleculeNet

#%%
def load_tox21():
    dataset = MoleculeNet(root='./data', name='Tox21')
    labels = np.array([dataset[i].y.numpy().reshape(-1) for i in range(len(dataset))])
    tasks = [
        'NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase','NR-ER','NR-ER-LBD',
        'NR-PPAR-gamma','SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53'
    ]
    return dataset, labels, tasks

dataset, labels, tasks = load_tox21()

#%%
def liver_distribution(labels, tasks, save_fig=True):
    idx = tasks.index("NR-AhR")
    y = labels[:, idx]
    
    total = len(y)
    valid = y[~np.isnan(y)]
    toxic, nontoxic = np.sum(valid==1), np.sum(valid==0)
    missing = total - len(valid)

    # ---- Stats ----
    print("\n📊 Liver Toxicity (NR-AhR)")
    print(f"- Total: {total}")
    print(f"- Valid: {len(valid)} ({len(valid)/total*100:.1f}%)")
    print(f"- Toxic: {toxic} ({toxic/len(valid)*100:.1f}%)")
    print(f"- Non-toxic: {nontoxic}")
    print(f"- Missing: {missing}")
    print(f"- Imbalance: {nontoxic/max(toxic,1):.1f}:1 (non-toxic:toxic)")

    # ---- Plots ----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,5))
    fig.suptitle("NR-AhR (Liver Toxicity) Distribution", fontsize=14, fontweight="bold")

    # Bar chart
    ax1.bar(["Non-toxic","Toxic","Missing"], [nontoxic,toxic,missing],
            color=["lightgreen","red","lightgray"], alpha=0.8)
    ax1.set_ylabel("Molecules")
    ax1.set_title("Class Counts")

    # Pie chart (valid only)
    if toxic > 0 and nontoxic > 0:
        ax2.pie([nontoxic,toxic], labels=["Non-toxic","Toxic"],
                colors=["lightgreen","red"], autopct="%1.1f%%", startangle=90)
        ax2.set_title("Valid Labels (%)")

    plt.tight_layout()

    if save_fig:
        os.makedirs("./Figures", exist_ok=True)
        filepath = os.path.join("./Figures", "liver_toxicity_distribution.png")
        plt.savefig(filepath, dpi=300, bbox_inches="tight", facecolor="white")
        print(f"✓ Figure saved: {filepath}")

    plt.show()

    return pd.DataFrame([{
        "Task": "NR-AhR",
        "Total": total,
        "Valid": len(valid),
        "Toxic": toxic,
        "Non-toxic": nontoxic,
        "Missing": missing,
        "Toxic %": toxic/len(valid)*100
    }])

# Run analysis
liver_stats = liver_distribution(labels, tasks, save_fig=True)

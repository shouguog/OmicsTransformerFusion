# %% [markdown]
# **## Set up Environment ##**
# %%
import os, sys, json, time, warnings, requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import (
    silhouette_score, normalized_mutual_info_score, adjusted_rand_score,
    homogeneity_score, completeness_score, v_measure_score,
    calinski_harabasz_score, davies_bouldin_score, confusion_matrix
)
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, SimpleImputer

from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from lifelines import KaplanMeierFitter, CoxPHFitter
from lifelines.statistics import multivariate_logrank_test

from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

from tqdm.auto import tqdm
from utils.training import HiOmicsLoss, HiOmicsTrainer

warnings.filterwarnings('ignore')

# %% [markdown]
# **## plotting Parameters ##**
# %%
plt.rcParams.update({
    'figure.dpi': 120,
    'savefig.dpi': 300,
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'axes.grid': False,
})
PALETTE = sns.color_palette('Set2', 20)
CMAP_SEQ = 'YlOrRd'
CMAP_DIV = 'RdBu_r'
# %% [markdown]
# **## Reproducibility ##**
# %%
SEED = 42
SEEDS = [42, 123, 456, 789, 2024]
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True

device = torch.device('cuda' if torch.cuda.is_available()
                      else 'mps' if torch.backends.mps.is_available()
                      else 'cpu')
print(f"Device: {device}")
print(f"Seeds for stability check: {SEEDS}")
# %% [markdown]
# **## CONFIGURATION Architecture, batch size, loss weights, and learning rate ##**
# %%
from configs.HiOmicsConfig import HiOmicsConfig
config = HiOmicsConfig()
os.makedirs(config.results_path, exist_ok=True)
os.makedirs(config.figures_path, exist_ok=True)

print("=" * 70)
print("HiOmicsFormer Configuration (Matches Manuscript Table 2)")
print("=" * 70)
for k, v in vars(config).items():
    if not k.startswith('_'):
        print(f"  {k}: {v}")

# %% [markdown]
# **## DATA LOADING — ALL CANCER TYPES ##**
# %%
from dataset.loader import MLOmicsDataLoader
data_loader = MLOmicsDataLoader(config.data_path, data_variant=config.data_variant)

# %% [markdown]
# **## Discover and load ALL cancer types ##**
# %%
discovered_cancers = data_loader.discover_cancer_types()

omics_data, raw_omics_data, survival_data, cancer_labels, sample_ids, quality_report = \
    data_loader.load_pan_cancer(discovered_cancers, config.modalities)

# Update num_clusters to match actual number of loaded cancer types
n_loaded = len(quality_report)
if n_loaded != config.num_clusters:
    print(f"\n⚠ Updating config.num_clusters: {config.num_clusters} → {n_loaded} "
          f"(matching loaded cancer types)")
    config.num_clusters = n_loaded

print(f"\n{'='*70}")
print(f"DATASET SUMMARY")
print(f"{'='*70}")
print(f"Cancer types loaded: {len(quality_report)}")
print(f"Total patients: {len(cancer_labels)}")
print(f"Unique labels: {sorted(set(cancer_labels))}")
for cancer, n in sorted(quality_report.items(), key=lambda x: -x[1]):
    name = MLOmicsDataLoader.CANCER_FULL_NAMES.get(cancer, '')
    print(f"  {cancer:6s}: {n:4d} patients  {name}")

# %% [markdown]
# **## DATASET EXPLORATION ##**
# %%
print(f"\nDataset statistics:")
print(f"  Total patients: {len(cancer_labels):,}")
print(f"  Cancer types: {len(quality_report)}")
# %% [markdown]
# **## PREPROCESSING, Raw pre-imputation data is stored, and Feature selection ##**
# %%
print("Preprocessing omics data (with documented feature selection)...")
from dataset.preprocessor import MultiOmicsPreprocessor
preprocessor = MultiOmicsPreprocessor(config)
processed_data = preprocessor.fit_transform(omics_data)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(cancer_labels)
n_classes = len(label_encoder.classes_)
print(f"\nEncoded {n_classes} cancer types")
print(f"Processed feature dimensions: { {m: d.shape for m, d in processed_data.items()} }")

# %% [markdown]
# **## PyTorch Dataset ##**
# %%
from dataset.dataset import MultiOmicsDataset
full_dataset = MultiOmicsDataset(
    processed_data, labels_encoded,
    survival_data=survival_data, sample_ids=sample_ids
)
feature_dims = full_dataset.get_feature_dims()
print(f"Dataset: {len(full_dataset)} samples")
print(f"Feature dims: {feature_dims}")

# %% [markdown]
# **## Verify architecture ##**
# %%
from models.transformer import HiOmicsFormer
model = HiOmicsFormer(feature_dims, config).to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel Parameters: {total_params:,}")
print(f"Hidden dim: {config.hidden_dim} (manuscript: 256) ✓" if config.hidden_dim == 256 else "✗ MISMATCH")
print(f"Transformer layers: {config.num_encoder_layers} (manuscript: 4) ✓" if config.num_encoder_layers == 4 else "✗ MISMATCH")
print(f"Attention heads: {config.num_heads} (manuscript: 8) ✓" if config.num_heads == 8 else "✗ MISMATCH")
print(f"Clusters: {config.num_clusters} (manuscript: 9) ✓" if config.num_clusters == 9 else "✗ MISMATCH")

# %% [markdown]
# Check the model structure
# %%
print(model)
# %% [markdown]
# **## Loss weights ##**
# %%
print(f"Loss weights: L_recon={config.lambda_recon}, "
      f"λ_CL={config.lambda_CL}, λ_DEC={config.lambda_DEC}")
print(f"Matches Eq. 10: L = {config.lambda_recon}·L_recon + "
      f"{config.lambda_CL}·L_CL + {config.lambda_DEC}·L_DEC ✓")
# %% [markdown]
# ** FUnction for TRAINING UTILITIES ##**
# %%
def init_centroids(model, loader, config, device):
    """K-means centroid initialization."""
    model.eval()
    with torch.no_grad():
        all_z = []
        for batch in loader:
            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            out = model(batch)
            all_z.append(out['z'].cpu().numpy())
        z = np.concatenate(all_z)

    km = KMeans(n_clusters=config.num_clusters, random_state=SEED, n_init=20)
    km.fit(z)
    with torch.no_grad():
        model.cluster_centroids.data = torch.tensor(km.cluster_centers_, dtype=torch.float32).to(device)
    return km.labels_

print("Training utilities defined.")


# %% [markdown]
# **## 5-FOLD STRATIFIED CROSS-VALIDATION ##**
# %%
skf = StratifiedKFold(n_splits=config.n_folds, shuffle=True, random_state=SEED)

fold_results = []
fold_predictions = {}  # Store for survival analysis
fold_embeddings = {}
fold_histories = {}  # Store training curves for visualization

print("=" * 70)
print(f"5-FOLD STRATIFIED CROSS-VALIDATION")
print("=" * 70)

for fold_idx, (train_val_idx, test_idx) in enumerate(skf.split(
        np.zeros(len(labels_encoded)), labels_encoded)):

    print(f"\n{'─'*70}")
    print(f"FOLD {fold_idx + 1}/{config.n_folds}")
    print(f"{'─'*70}")
    print(f"  Train+Val: {len(train_val_idx)}, Test: {len(test_idx)}")

    # Split train into train/val (80/20 of train portion)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=0.2,
        stratify=labels_encoded[train_val_idx], random_state=SEED + fold_idx
    )

    # DataLoaders
    train_loader = DataLoader(full_dataset, batch_size=config.batch_size,
                              sampler=SubsetRandomSampler(train_idx),
                              num_workers=0, drop_last=False)
    val_loader = DataLoader(full_dataset, batch_size=config.batch_size,
                            sampler=SubsetRandomSampler(val_idx),
                            num_workers=0)
    test_loader = DataLoader(full_dataset, batch_size=config.batch_size,
                             sampler=SubsetRandomSampler(test_idx),
                             num_workers=0)

    # Fresh model each fold
    model_fold = HiOmicsFormer(feature_dims, config).to(device)

    # --- Phase 1: Pre-training (reconstruction only) ---
    print("  Phase 1: Pre-training encoder...")
    config_p1 = HiOmicsConfig()
    config_p1.lambda_DEC = 0.0  # No clustering
    config_p1.max_epochs = 30
    config_p1.patience = 15

    criterion_p1 = HiOmicsLoss(config_p1)
    optimizer_p1 = torch.optim.AdamW(model_fold.parameters(),
                                      lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
    scheduler_p1 = CosineAnnealingWarmRestarts(optimizer_p1, T_0=10, T_mult=2)

    trainer_p1 = HiOmicsTrainer(model_fold, optimizer_p1, criterion_p1,
                                 scheduler_p1, config_p1, device)
    history_p1 = trainer_p1.train_fold(train_loader, val_loader,
                          labels_encoded[train_idx], labels_encoded[val_idx])

    # Initialize centroids
    init_centroids(model_fold, train_loader, config, device)

    # --- Phase 2: Fine-tune with clustering ---
    print("  Phase 2: Fine-tuning with clustering...")
    criterion_p2 = HiOmicsLoss(config)
    optimizer_p2 = torch.optim.AdamW(model_fold.parameters(),
                                      lr=config.learning_rate,
                                      weight_decay=config.weight_decay)
    scheduler_p2 = CosineAnnealingWarmRestarts(optimizer_p2, T_0=20, T_mult=2)

    trainer_p2 = HiOmicsTrainer(model_fold, optimizer_p2, criterion_p2,
                                 scheduler_p2, config, device)
    history_p2 = trainer_p2.train_fold(train_loader, val_loader,
                          labels_encoded[train_idx], labels_encoded[val_idx])

    # --- Evaluate on test fold ---
    z_test, pred_test, test_metrics = trainer_p2.evaluate(
        test_loader, labels_encoded[test_idx]
    )

    fold_results.append(test_metrics)
    fold_predictions[fold_idx] = {'test_idx': test_idx, 'pred': pred_test, 'z': z_test}
    fold_histories[fold_idx] = {'p1': history_p1, 'p2': history_p2}
    fold_embeddings[fold_idx] = z_test

    print(f"  ▶ Fold {fold_idx+1} results: "
          f"Sil={test_metrics['silhouette']:.4f}, "
          f"NMI={test_metrics['nmi']:.4f}, "
          f"ARI={test_metrics['ari']:.4f}")

# --- Aggregate across folds ---
print(f"\n{'='*70}")
print("CROSS-VALIDATION RESULTS (mean ± SD)")
print(f"{'='*70}")

cv_summary = {}
for metric in ['silhouette', 'nmi', 'ari']:
    values = [r[metric] for r in fold_results]
    mean, std = np.mean(values), np.std(values)
    cv_summary[metric] = {'mean': mean, 'std': std, 'values': values}
    print(f"  {metric.upper():12s}: {mean:.3f} ± {std:.3f}  "
          f"(per-fold: {[f'{v:.3f}' for v in values]})")

# Keep last fold model for downstream analysis
model = model_fold

# %% [markdown]
# **## Visualize training progress, learned embeddings, and cluster quality ##**
# %%
# %matplotlib inline
fig = plt.figure(figsize=(22, 16))
gs = gridspec.GridSpec(3, 3, hspace=0.4, wspace=0.35)

# --- Fig 3a: Training loss curves per fold ---
ax = fig.add_subplot(gs[0, 0])
for fold_idx, hist in fold_histories.items():
    p1_loss = hist['p1']['train_loss']
    p2_loss = hist['p2']['train_loss']
    combined = p1_loss + p2_loss
    ax.plot(combined, alpha=0.7, label=f'Fold {fold_idx+1}', linewidth=1.5)
    # Mark phase transition
    ax.axvline(len(p1_loss), color='gray', linestyle=':', alpha=0.3)
ax.set_xlabel('Epoch')
ax.set_ylabel('Training Loss')
ax.set_title('(a) Training Loss per Fold', fontweight='bold')
ax.legend(fontsize=7)
ax.text(0.02, 0.95, 'Phase 1 | Phase 2', transform=ax.transAxes,
        fontsize=7, color='gray', va='top')

# --- Fig 3b: Per-fold metrics bar chart ---
ax = fig.add_subplot(gs[0, 1])
metrics_names = ['silhouette', 'nmi', 'ari']
x = np.arange(config.n_folds)
width = 0.25
for i, metric in enumerate(metrics_names):
    vals = [fold_results[f][metric] for f in range(len(fold_results))]
    ax.bar(x + i*width, vals, width, label=metric.upper(), color=PALETTE[i], edgecolor='white')
ax.set_xticks(x + width)
ax.set_xticklabels([f'Fold {i+1}' for i in range(config.n_folds)])
ax.set_ylabel('Score')
ax.set_title('(b) Per-Fold Metrics', fontweight='bold')
ax.legend()
ax.set_ylim(0, 1.05)

# --- Fig 3c: CV summary with error bars ---
ax = fig.add_subplot(gs[0, 2])
means = [cv_summary[m]['mean'] for m in metrics_names]
stds = [cv_summary[m]['std'] for m in metrics_names]
bars = ax.bar(range(len(metrics_names)), means, yerr=stds, capsize=8,
              color=[PALETTE[i] for i in range(len(metrics_names))], edgecolor='white',
              width=0.5, alpha=0.9)
ax.set_xticks(range(len(metrics_names)))
ax.set_xticklabels([m.upper() for m in metrics_names])
ax.set_ylabel('Score (mean ± SD)')
ax.set_title('(c) Cross-Validation Summary', fontweight='bold')
for bar, m, s in zip(bars, means, stds):
    ax.text(bar.get_x() + bar.get_width()/2, m + s + 0.02, f'{m:.3f}±{s:.3f}',
            ha='center', fontsize=8)
ax.set_ylim(0, 1.15)

# --- Fig 3d: t-SNE of learned embeddings (colored by cancer type) ---
ax = fig.add_subplot(gs[1, 0])
last_fold = max(fold_predictions.keys())
z_viz = fold_predictions[last_fold]['z']
test_idx_viz = fold_predictions[last_fold]['test_idx']
labels_viz = cancer_labels[test_idx_viz]

# t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=SEED, max_iter=1000)
z_2d = tsne.fit_transform(z_viz)

unique_labels = np.unique(labels_viz)
colors_map = {lab: PALETTE[i % len(PALETTE)] for i, lab in enumerate(unique_labels)}
for lab in unique_labels:
    mask = labels_viz == lab
    ax.scatter(z_2d[mask, 0], z_2d[mask, 1], c=[colors_map[lab]], s=8, alpha=0.6, label=lab)
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_title('(d) t-SNE: Colored by Cancer Type', fontweight='bold')
ax.legend(fontsize=6, ncol=2, markerscale=2, loc='best')

# --- Fig 3e: t-SNE colored by cluster assignment ---
ax = fig.add_subplot(gs[1, 1])
pred_viz = fold_predictions[last_fold]['pred']
unique_preds = np.unique(pred_viz)
for cl in unique_preds:
    mask = pred_viz == cl
    ax.scatter(z_2d[mask, 0], z_2d[mask, 1], s=8, alpha=0.6, label=f'C{cl}')
ax.set_xlabel('t-SNE 1')
ax.set_ylabel('t-SNE 2')
ax.set_title('(e) t-SNE: Colored by Predicted Cluster', fontweight='bold')
ax.legend(fontsize=6, ncol=2, markerscale=2, loc='best')

# --- Fig 3f: Confusion matrix (cancer type vs cluster) ---
ax = fig.add_subplot(gs[1, 2])
# Create a co-occurrence matrix
le_viz = LabelEncoder()
labels_num = le_viz.fit_transform(labels_viz)
cm = confusion_matrix(labels_num, pred_viz)
# Normalize by row (true label)
cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
sns.heatmap(cm_norm, ax=ax, cmap='YlOrRd', annot=False,
            xticklabels=[f'C{i}' for i in range(cm_norm.shape[1])],
            yticklabels=le_viz.classes_,
            cbar_kws={'label': 'Proportion'})
ax.set_xlabel('Predicted Cluster')
ax.set_ylabel('True Cancer Type')
ax.set_title('(f) Cancer Type vs Cluster Assignment', fontweight='bold')

# --- Fig 3g: Silhouette scores distribution ---
ax = fig.add_subplot(gs[2, 0])
from sklearn.metrics import silhouette_samples
sil_samples = silhouette_samples(z_viz, pred_viz)
y_lower = 0
for cl in sorted(unique_preds):
    mask = pred_viz == cl
    sil_cl = np.sort(sil_samples[mask])
    ax.barh(range(y_lower, y_lower + len(sil_cl)), sil_cl, height=1.0,
            color=PALETTE[cl % len(PALETTE)], edgecolor='none')
    ax.text(-0.05, y_lower + len(sil_cl)/2, f'C{cl}', fontsize=7, va='center')
    y_lower += len(sil_cl) + 5
ax.axvline(np.mean(sil_samples), color='red', linestyle='--', alpha=0.7,
           label=f'Mean={np.mean(sil_samples):.3f}')
ax.set_xlabel('Silhouette Coefficient')
ax.set_title('(g) Per-Sample Silhouette', fontweight='bold')
ax.legend(fontsize=8)

# --- Fig 3h: Cluster size distribution ---
ax = fig.add_subplot(gs[2, 1])
cluster_counts = pd.Series(pred_viz).value_counts().sort_index()
ax.bar(cluster_counts.index, cluster_counts.values,
       color=[PALETTE[i % len(PALETTE)] for i in cluster_counts.index],
       edgecolor='white')
ax.set_xlabel('Cluster ID')
ax.set_ylabel('Number of Samples')
ax.set_title('(h) Cluster Size Distribution', fontweight='bold')
for idx, val in zip(cluster_counts.index, cluster_counts.values):
    ax.text(idx, val + 2, str(val), ha='center', fontsize=8)

# --- Fig 3i: PCA of embeddings (2D) ---
ax = fig.add_subplot(gs[2, 2])
pca_emb = PCA(n_components=2)
z_pca = pca_emb.fit_transform(z_viz)
for lab in unique_labels:
    mask = labels_viz == lab
    ax.scatter(z_pca[mask, 0], z_pca[mask, 1], c=[colors_map[lab]], s=8, alpha=0.6, label=lab)
ax.set_xlabel(f'PC1 ({pca_emb.explained_variance_ratio_[0]*100:.1f}%)')
ax.set_ylabel(f'PC2 ({pca_emb.explained_variance_ratio_[1]*100:.1f}%)')
ax.set_title('(i) PCA of Learned Embeddings', fontweight='bold')
ax.legend(fontsize=6, ncol=2, markerscale=2, loc='best')

plt.suptitle('Figure 3: Training Results & Learned Representations', fontsize=16, fontweight='bold', y=1.01)
#plt.savefig(f'{config.figures_path}/fig3_training_embeddings.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nEmbedding visualization statistics:")
print(f"  Samples visualized: {len(z_viz)}")
print(f"  Unique cancer types: {len(unique_labels)}")
print(f"  Unique clusters: {len(unique_preds)}")
print(f"  Mean silhouette: {np.mean(sil_samples):.4f}")

# %% [markdown]
# **## Finished ##**
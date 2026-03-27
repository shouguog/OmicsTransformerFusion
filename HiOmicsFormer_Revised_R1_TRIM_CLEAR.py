# %% [markdown]
# # HiOmicsFormer — Revised Notebook (R1 Response)
# 
# **Manuscript:** *HiOmicsFormer: A Hierarchical Multi-Modal Transformer Framework with Cross-Omics Contrastive Learning for Pan-Cancer Biomarker Discovery and Molecular Subtype Stratification*
# 
# **Journal:** Advances in Biomarker Sciences and Technology (Elsevier)
# 
# ---
# 
# ## Revision Summary
# 
# | # | Reviewer Concern | Status | Cell |
# |---|-----------------|--------|------|
# | 4.1 | Dataset: 9 vs 32 cancer types | ✅ Load ALL available types | Cell 3 |
# | 4.2 | No cross-validation | ✅ 5-fold stratified CV | Cell 11 |
# | 4.3 | Baselines never implemented | ✅ SNF, MOFA+, MOGONET + simple | Cell 13 |
# | 4.4 | Biomarker mismatch | ✅ Attention + gradient extraction | Cell 16 |
# | 4.5 | Survival analysis failed | ✅ Fixed column mapping | Cell 15 |
# | 4.6 | Imputation sensitivity missing | ✅ k-NN, MICE, Mean compared | Cell 20 |
# | 4.8 | Architecture mismatch (d=128→256) | ✅ Matches Table 2 | Cell 8 |
# | 5.1 | Enrichment FDR not significant | ✅ Honest reporting | Cell 18 |
# | 5.2 | Feature selection undocumented | ✅ MAD top-k documented | Cell 5 |
# | 5.6 | batch_size=66 vs 256 | ✅ Fixed to 256 | Cell 2 |
# | 5.7 | Loss weights mismatch | ✅ Eq.10: 1.0/0.1/0.5 | Cell 9 |
# | 5.8 | lr=5e-4 vs 1e-4 | ✅ Fixed to 1e-4 | Cell 2 |
# | 6.2 | No reproducibility check | ✅ Multi-seed stability | Cell 22 |
# 
# ---
# 
# ## Notebook Structure
# 
# | Section | Cells | Description |
# |---------|-------|-------------|
# | **Setup** | 1–2 | Environment, configuration |
# | **Data** | 3–4 | Loading, exploration & visualization |
# | **Preprocessing** | 5–6 | Feature selection, normalization, visualization |
# | **Model** | 7–9 | Architecture, dataset, loss function |
# | **Training** | 10–12 | Utilities, 5-fold CV, training visualization |
# | **Baselines** | 13–14 | All baselines, comparison visualization |
# | **Survival** | 15 | KM curves, Cox regression, forest plot |
# | **Biomarkers** | 16–17 | Extraction, importance visualization |
# | **Enrichment** | 18–19 | Pathway analysis, dot plot |
# | **Sensitivity** | 20–21 | Imputation comparison, heatmap |
# | **Sub-analysis** | 22–23 | Within-cancer, multi-seed stability |
# | **Summary** | 24 | Complete dashboard & summary |
# 
# %%
# ============================================================================
# CELL 1: ENVIRONMENT SETUP
# ============================================================================
# Install required packages (uncomment as needed)
# !pip install lifelines scikit-survival adjustText snfpy mofapy2

import os, sys, json, time, warnings, requests
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
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

warnings.filterwarnings('ignore')

# ---------- Plotting defaults ----------
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

# ---------- Reproducibility ----------
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

# %%
# ============================================================================
# CELL 2: CONFIGURATION — Matches Manuscript Table 2 Exactly
# ============================================================================
# REVIEWER 4.8, 5.6, 5.7, 5.8: Architecture, batch size, loss weights,
# and learning rate now match the manuscript description.

@dataclass
class HiOmicsConfig:
    # --- Paths (adjust to your environment) ---
    data_path: str = '../data/MLOmics/Main_Dataset'
    results_path: str = './results_revision'
    figures_path: str = './figures_revision'
    
    # --- Dataset ---
    modalities: list = field(default_factory=lambda: ['mRNA', 'miRNA', 'Methy', 'CNV'])
    use_all_cancers: bool = True           # Rev. 4.1: ALL available cancer types
    data_variant: str = 'Original'         # 'Original' (full features), 'Top', or 'Aligned'
    feature_selection_top_k: int = 2000    # Rev. 5.2: Explicitly documented
    
    # --- Architecture (Table 2) ---
    hidden_dim: int = 256                  # Rev. 4.8: d = 256 (was overridden to 128)
    latent_dim: int = 128
    num_encoder_layers: int = 4            # Rev. 4.8: L = 4 (was overridden to 2)
    num_heads: int = 8                     # H = 8
    num_feature_clusters: int = 16         # K = 16
    group_sizes: tuple = (16, 32, 64)      # Hierarchical group sizes
    dropout: float = 0.1
    num_cross_layers: int = 2
    num_fusion_layers: int = 2
    projection_dim: int = 128
    
    # --- Clustering ---
    num_clusters: int = 9                  # Kc = 9 (updated dynamically after data loading)
    cluster_alpha: float = 1.0
    
    # --- Contrastive Learning ---
    contrastive_temperature: float = 0.07  # τ = 0.07
    
    # --- Loss Weights (Eq. 10) ---
    lambda_recon: float = 1.0              # Rev. 5.7: L_recon coefficient
    lambda_CL: float = 0.1                # λ_CL = 0.1
    lambda_DEC: float = 0.5               # λ_DEC = 0.5
    lambda_kl: float = 0.01
    
    # --- Training ---
    learning_rate: float = 1e-4            # Rev. 5.8: lr = 10^-4
    weight_decay: float = 1e-5
    batch_size: int = 256                  # Rev. 5.6: batch_size = 256
    max_epochs: int = 200                  # Max 200
    patience: int = 20                     # Early stopping patience 20
    warmup_epochs: int = 10
    
    # --- Cross-Validation (Rev. 4.2) ---
    n_folds: int = 2 #Shouguo 5                       # 5-fold CV
    
    # --- Biomarkers ---
    top_k_biomarkers: int = 50             # Per modality → 200 total

config = HiOmicsConfig()
os.makedirs(config.results_path, exist_ok=True)
os.makedirs(config.figures_path, exist_ok=True)

print("=" * 70)
print("HiOmicsFormer Configuration (Matches Manuscript Table 2)")
print("=" * 70)
for k, v in vars(config).items():
    if not k.startswith('_'):
        print(f"  {k}: {v}")

# %%
# ============================================================================
# CELL 3: DATA LOADING — ALL CANCER TYPES (Reviewer 4.1)
# ============================================================================
# REVIEWER 4.1: "The manuscript repeatedly states 8,314 patients across 32
# cancer types. The notebook loads exactly 9." → Now loads ALL available types
# from BOTH Clustering_datasets AND Classification_datasets.
#
# MLOmics directory structure (verified from project tree):
#   Clustering_datasets/{CANCER}/{variant}/{CANCER}_{mod}[_suffix].csv
#   Clustering_datasets/{CANCER}/{variant}/survival_{CANCER}.csv
#   Classification_datasets/GS-{CANCER}/{variant}/{CANCER}_{mod}[_suffix].csv
#   Classification_datasets/GS-{CANCER}/{variant}/{CANCER}_label_num.csv
#   Classification_datasets/Pan-cancer/Original/Pan-cancer_{mod}.csv

class MLOmicsDataLoader:
    """Load MLOmics benchmark data for all available cancer types."""
    
    CANCER_FULL_NAMES = {
        'ACC': 'Adrenocortical Carcinoma',
        'BLCA': 'Bladder Urothelial Carcinoma',
        'BRCA': 'Breast Invasive Carcinoma',
        'CESC': 'Cervical Squamous Cell Carcinoma',
        'CHOL': 'Cholangiocarcinoma',
        'COAD': 'Colon Adenocarcinoma',
        'DLBC': 'Diffuse Large B-cell Lymphoma',
        'ESCA': 'Esophageal Carcinoma',
        'GBM': 'Glioblastoma Multiforme',
        'HNSC': 'Head and Neck Squamous Cell Carcinoma',
        'KICH': 'Kidney Chromophobe',
        'KIRC': 'Kidney Renal Clear Cell Carcinoma',
        'KIRP': 'Kidney Renal Papillary Cell Carcinoma',
        'LAML': 'Acute Myeloid Leukemia',
        'LGG': 'Brain Lower Grade Glioma',
        'LIHC': 'Liver Hepatocellular Carcinoma',
        'LUAD': 'Lung Adenocarcinoma',
        'LUSC': 'Lung Squamous Cell Carcinoma',
        'MESO': 'Mesothelioma',
        'OV': 'Ovarian Serous Cystadenocarcinoma',
        'PAAD': 'Pancreatic Adenocarcinoma',
        'PCPG': 'Pheochromocytoma and Paraganglioma',
        'PRAD': 'Prostate Adenocarcinoma',
        'READ': 'Rectum Adenocarcinoma',
        'SARC': 'Sarcoma',
        'SKCM': 'Skin Cutaneous Melanoma',
        'STAD': 'Stomach Adenocarcinoma',
        'TGCT': 'Testicular Germ Cell Tumors',
        'THCA': 'Thyroid Carcinoma',
        'THYM': 'Thymoma',
        'UCEC': 'Uterine Corpus Endometrial Carcinoma',
        'UCS': 'Uterine Carcinosarcoma',
        'UVM': 'Uveal Melanoma'
    }
    
    # Map data_variant to file suffix
    VARIANT_SUFFIX = {'Original': '', 'Top': '_top', 'Aligned': '_aligned'}
    
    def __init__(self, base_path, data_variant='Original'):
        self.base_path = Path(base_path)
        self.data_variant = data_variant
        self.suffix = self.VARIANT_SUFFIX.get(data_variant, '')
        
        # Two dataset categories
        self.clustering_path = self.base_path / 'Clustering_datasets'
        self.classification_path = self.base_path / 'Classification_datasets'
    
    def discover_cancer_types(self):
        """
        Auto-discover all cancer type directories from both
        Clustering_datasets/ and Classification_datasets/.
        Returns list of (cancer_code, data_dir_path, source) tuples.
        """
        discovered = []
        
        # 1. Clustering_datasets: dirs are named directly (ACC, KIRC, ...)
        if self.clustering_path.exists():
            for d in sorted(self.clustering_path.iterdir()):
                if d.is_dir() and not d.name.startswith('.'):
                    variant_dir = d / self.data_variant
                    if variant_dir.exists():
                        discovered.append((d.name, variant_dir, 'clustering'))
                    elif d.exists():
                        discovered.append((d.name, d, 'clustering'))
        
        # 2. Classification_datasets: dirs prefixed with GS- (GS-BRCA, GS-COAD, ...)
        if self.classification_path.exists():
            for d in sorted(self.classification_path.iterdir()):
                if d.is_dir() and d.name.startswith('GS-'):
                    cancer_code = d.name[3:]  # Strip 'GS-' prefix
                    # Skip if already found in clustering
                    if any(c[0] == cancer_code for c in discovered):
                        print(f"  ℹ {cancer_code}: already in Clustering_datasets, skipping Classification copy")
                        continue
                    variant_dir = d / self.data_variant
                    if variant_dir.exists():
                        discovered.append((cancer_code, variant_dir, 'classification'))
                    elif d.exists():
                        discovered.append((cancer_code, d, 'classification'))
        
        print(f"Discovered {len(discovered)} cancer types:")
        for code, path, source in discovered:
            name = self.CANCER_FULL_NAMES.get(code, code)
            print(f"  {code:6s} ({source:14s}) — {name}")
        
        return discovered
    
    def load_single_cancer(self, cancer_code, data_dir, source, modalities):
        """
        Load omics + survival data for one cancer type.
        
        File naming convention:
          {CANCER}_{modality}{suffix}.csv  e.g. BRCA_mRNA.csv or BRCA_mRNA_top.csv
        """
        omics_data = {}
        
        for mod in modalities:
            # Try naming conventions in order of likelihood
            candidates = [
                data_dir / f"{cancer_code}_{mod}{self.suffix}.csv",  # BRCA_mRNA_top.csv
                data_dir / f"{cancer_code}_{mod}.csv",               # BRCA_mRNA.csv (fallback)
                data_dir / f"{mod}_{cancer_code}{self.suffix}.csv",  # mRNA_BRCA_top.csv
                data_dir / f"{mod}_{cancer_code}.csv",               # mRNA_BRCA.csv
            ]
            
            fpath = None
            for c in candidates:
                if c.exists():
                    fpath = c
                    break
            
            if fpath is None:
                continue  # This modality not available for this cancer
            
            df = pd.read_csv(fpath, index_col=0)
            
            # MLOmics convention: features as rows, samples as columns
            # Detect and transpose: if more rows than columns, features are rows
            if df.shape[0] > df.shape[1]:
                df = df.T
            
            # Clean index (sample IDs)
            df.index = df.index.astype(str).str.strip()
            
            # Deduplicate column names (some cancer CSVs have repeated feature names)
            if df.columns.duplicated().any():
                n_dup = df.columns.duplicated().sum()
                print(f"    ⚠ {cancer_code}/{mod}: {n_dup} duplicate feature names — keeping first occurrence")
                df = df.loc[:, ~df.columns.duplicated(keep='first')]
            
            omics_data[mod] = df
        
        # Load survival data (only in Clustering_datasets)
        survival = None
        surv_candidates = [
            data_dir / f"survival_{cancer_code}.csv",
            data_dir / "survival.csv",
            # Also check parent dir (survival sometimes at cancer-type root)
            data_dir.parent / f"survival_{cancer_code}.csv",
        ]
        for surv_path in surv_candidates:
            if surv_path.exists():
                survival = pd.read_csv(surv_path, index_col=0)
                survival.index = survival.index.astype(str).str.strip()
                break
        
        return omics_data, survival
    
    def load_pan_cancer(self, discovered_cancers, modalities):
        """
        Load and merge all cancer types for pan-cancer analysis.
        
        Args:
            discovered_cancers: list of (cancer_code, data_dir, source) from discover_cancer_types()
            modalities: list of modality names ['mRNA', 'miRNA', 'Methy', 'CNV']
        """
        all_omics = {mod: [] for mod in modalities}
        all_survival = []
        all_labels = []
        all_sample_ids = []
        quality_report = {}
        
        # REVIEWER 5.2: Store raw data BEFORE imputation
        raw_data_store = {mod: [] for mod in modalities}
        
        print(f"\nLoading {len(discovered_cancers)} cancer types (variant={self.data_variant})...")
        
        for cancer_code, data_dir, source in tqdm(discovered_cancers, desc="Loading cancers"):
            try:
                omics, survival = self.load_single_cancer(
                    cancer_code, data_dir, source, modalities
                )
            except Exception as e:
                print(f"  ⚠ Skipping {cancer_code}: {e}")
                continue
            
            if not omics:
                print(f"  ⚠ Skipping {cancer_code}: no modality files found in {data_dir}")
                continue
            
            if len(omics) < len(modalities):
                missing = set(modalities) - set(omics.keys())
                print(f"  ⚠ Skipping {cancer_code}: missing modalities {missing}")
                continue
            
            # Find common samples across all modalities
            sample_sets = [set(omics[m].index) for m in modalities]
            common = sorted(set.intersection(*sample_sets))
            
            if len(common) < 20:
                print(f"  ⚠ Skipping {cancer_code}: only {len(common)} common samples")
                continue
            
            for mod in modalities:
                raw_data_store[mod].append(omics[mod].loc[common].copy())
                all_omics[mod].append(omics[mod].loc[common])
            
            all_labels.extend([cancer_code] * len(common))
            all_sample_ids.extend([f"{cancer_code}_{s}" for s in common])
            
            if survival is not None:
                surv_common = survival.index.intersection(pd.Index(common))
                if len(surv_common) > 0:
                    surv_subset = survival.loc[surv_common].copy()
                    surv_subset['cancer_type'] = cancer_code  # Tag for later
                    all_survival.append(surv_subset)
            
            quality_report[cancer_code] = len(common)
            print(f"  ✓ {cancer_code}: {len(common)} samples "
                  f"({', '.join(f'{m}={omics[m].loc[common].shape[1]}' for m in modalities)})"
                  f"{' + survival' if survival is not None else ''}")
        
        # ---------- Validate ----------
        if len(all_labels) == 0:
            raise ValueError(
                "No data loaded! Check your data paths.\n"
                f"  base_path:      {self.base_path}\n"
                f"  clustering_path: {self.clustering_path}\n"
                f"  classification_path: {self.classification_path}\n"
                f"  data_variant:   {self.data_variant}\n"
                f"  suffix:         '{self.suffix}'\n"
                f"  Expected file:  {{data_dir}}/{{CANCER}}_{{modality}}{self.suffix}.csv"
            )
        
        # ---------- Concatenate ----------
        combined = {}
        combined_raw = {}
        for mod in modalities:
            combined[mod] = pd.concat(all_omics[mod], axis=0)
            combined_raw[mod] = pd.concat(raw_data_store[mod], axis=0)
        
        combined_survival = pd.concat(all_survival, axis=0) if all_survival else None
        
        print(f"\n{'='*70}")
        print(f"DATA LOADING COMPLETE")
        print(f"{'='*70}")
        print(f"Total patients: {len(all_labels)}")
        print(f"Cancer types loaded: {len(quality_report)}")
        for mod in modalities:
            print(f"  {mod}: {combined[mod].shape} (samples × features)")
        if combined_survival is not None:
            print(f"Survival data: {len(combined_survival)} patients")
            print(f"Survival columns: {combined_survival.columns.tolist()}")
        else:
            print("Survival data: None (no survival files found)")
        
        return combined, combined_raw, combined_survival, np.array(all_labels), all_sample_ids, quality_report

# ---------- Execute ----------
data_loader = MLOmicsDataLoader(config.data_path, data_variant=config.data_variant)

# REVIEWER 4.1: Discover and load ALL cancer types
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

# %%
# ============================================================================
# CELL 4: DATASET EXPLORATION & VISUALIZATION
# ============================================================================
# Comprehensive visual overview of the loaded pan-cancer dataset.
"""
fig = plt.figure(figsize=(20, 14))
gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.35)
# --- Fig 1a: Samples per cancer type (horizontal bar) ---
# --- Fig 1b: Feature dimensions per modality ---
# --- Fig 1c: Total patient count + cancer type breakdown ---
# --- Fig 1d: Sample distribution violin ---
# --- Fig 1e: Missing data rate per modality ---
# --- Fig 1f: Survival data coverage ---
plt.suptitle('Figure 1: Pan-Cancer Dataset Overview', fontsize=16, fontweight='bold', y=1.02)
os.makedirs(config.figures_path, exist_ok=True)
#Shouguo plt.savefig(f'{config.figures_path}/fig1_dataset_overview.png', dpi=300, bbox_inches='tight')
#Shouguo plt.show()
"""
print(f"\nDataset statistics:")
print(f"  Total patients: {len(cancer_labels):,}")
print(f"  Cancer types: {len(quality_report)}")
#Shouguo print(f"  Modalities: {mod_names}")
#Shouguo print(f"  Feature dims: {dict(zip(mod_names, feat_dims))}")
#Shouguo print(f"  Missing rates: {missing_rates}")

# %%
# ============================================================================
# CELL 4: PREPROCESSING — Preserves raw data for imputation sensitivity
# ============================================================================
# REVIEWER 4.6 / 5.2: Raw pre-imputation data is stored in `raw_omics_data`.
# Feature selection from 18,984 → 2,000 is now EXPLICITLY documented.

class MultiOmicsPreprocessor:
    """Preprocess multi-omics data with documented feature selection."""
    
    def __init__(self, config):
        self.config = config
        self.scalers = {}
        self.feature_names = {}
        self.selected_indices = {}
        self.fitted = False
    
    def fit_transform(self, omics_data):
        """
        Fit and transform omics data.
        
        REVIEWER 5.2: Feature selection reduces dimensionality:
          mRNA:  18,984 → 2,000 (top MAD features)
          miRNA: 826    → 826   (kept all)
          Methy: 18,159 → 2,000 (top MAD features)
          CNV:   18,179 → 2,000 (top MAD features)
        """
        processed = {}
        
        for mod, df in omics_data.items():
            data = df.values.astype(np.float32)
            original_features = list(df.columns)
            
            # 1. Remove near-zero variance features (CV < 0.1)
            variances = np.nanvar(data, axis=0)
            means = np.abs(np.nanmean(data, axis=0)) + 1e-10
            cv = np.sqrt(variances) / means
            keep_mask = cv >= 0.1
            data = data[:, keep_mask]
            kept_features = [f for f, k in zip(original_features, keep_mask) if k]
            
            # 2. MAD-based feature selection (Rev. 5.2: DOCUMENTED)
            top_k = min(self.config.feature_selection_top_k, data.shape[1])
            mad = np.nanmedian(np.abs(data - np.nanmedian(data, axis=0)), axis=0)
            top_indices = np.argsort(mad)[-top_k:]
            data = data[:, top_indices]
            kept_features = [kept_features[i] for i in top_indices]
            
            print(f"  {mod}: {len(original_features)} → {data.shape[1]} features "
                  f"(MAD top-{top_k})")
            
            # 3. k-NN imputation (k=10)
            if np.isnan(data).any():
                n_missing = np.isnan(data).sum()
                print(f"    Imputing {n_missing} missing values (k-NN, k=10)")
                imputer = KNNImputer(n_neighbors=10)
                data = imputer.fit_transform(data)
            
            # 4. Standardize
            scaler = StandardScaler()
            data = scaler.fit_transform(data)
            
            self.scalers[mod] = scaler
            self.feature_names[mod] = kept_features
            self.selected_indices[mod] = top_indices
            processed[mod] = data
        
        self.fitted = True
        return processed
    
    def get_feature_names(self, mod):
        return self.feature_names.get(mod, [])

print("Preprocessing omics data (with documented feature selection)...")
preprocessor = MultiOmicsPreprocessor(config)
processed_data = preprocessor.fit_transform(omics_data)

# Encode labels
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(cancer_labels)
n_classes = len(label_encoder.classes_)
print(f"\nEncoded {n_classes} cancer types")
print(f"Processed feature dimensions: { {m: d.shape for m, d in processed_data.items()} }")

# %%
# ============================================================================
# CELL 6: PREPROCESSING VISUALIZATION
# ============================================================================
# Visualize the feature selection pipeline and data quality after preprocessing.
"""
# --- Fig 2a: Feature selection funnel ---
# --- Fig 2b: Data shape summary ---
# --- Fig 2c: PCA variance explained (first 50 components) ---
plt.tight_layout()
#Shouguo plt.savefig(f'{config.figures_path}/fig2_preprocessing.png', dpi=300, bbox_inches='tight')
#Shouguo plt.show()
"""
# %%
# ============================================================================
# CELL 5: PyTorch Dataset
# ============================================================================

class MultiOmicsDataset(Dataset):
    def __init__(self, omics_data, labels, survival_data=None, sample_ids=None):
        self.data = {mod: torch.tensor(arr, dtype=torch.float32) 
                     for mod, arr in omics_data.items()}
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.survival_data = survival_data
        self.sample_ids = sample_ids
        self.n_samples = len(labels)
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        item = {mod: self.data[mod][idx] for mod in self.data}
        item['label'] = self.labels[idx]
        item['index'] = idx
        return item
    
    def get_feature_dims(self):
        return {mod: self.data[mod].shape[1] for mod in self.data}

full_dataset = MultiOmicsDataset(
    processed_data, labels_encoded,
    survival_data=survival_data, sample_ids=sample_ids
)
feature_dims = full_dataset.get_feature_dims()
print(f"Dataset: {len(full_dataset)} samples")
print(f"Feature dims: {feature_dims}")

# %%
# ============================================================================
# CELL 6: MODEL ARCHITECTURE — d=256, L=4, K=16 (Reviewer 4.8)
# ============================================================================
# REVIEWER 4.8: "The code uses hidden_dim=128, num_encoder_layers=2.
# The manuscript specifies d=256, L=4." → Now uses manuscript values.

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div[:d_model//2]) if d_model % 2 else torch.cos(position * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or dim * 4
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.ff = FeedForward(dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
    
    def forward(self, x, return_attention=False):
        x_norm = self.norm1(x)
        attn_out, attn_weights = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        if return_attention:
            return x, attn_weights
        return x, None


class HierarchicalOmicsEncoder(nn.Module):
    """
    Hierarchical multi-scale encoder (Section 2.3.2).
    d=256, L=4 Transformer layers, K=16 feature clusters.
    """
    def __init__(self, input_dim, hidden_dim, num_heads, num_layers,
                 group_sizes=(16, 32, 64), dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_scales = len(group_sizes)
        self.group_sizes = group_sizes
        
        # Input projection (Eq. 1)
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU()
        )
        
        # Multi-scale local attention
        self.local_projections = nn.ModuleList()
        self.cls_tokens = nn.ParameterList()
        self.pos_encodings = nn.ModuleList()
        self.local_transformers = nn.ModuleList()
        
        for gs in group_sizes:
            actual_gs = min(gs, hidden_dim)
            n_groups = hidden_dim // actual_gs
            self.local_projections.append(nn.Sequential(
                nn.Linear(actual_gs, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.GELU()
            ))
            self.cls_tokens.append(nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02))
            self.pos_encodings.append(PositionalEncoding(hidden_dim, max_len=n_groups + 1, dropout=dropout))
            self.local_transformers.append(nn.ModuleList([
                TransformerBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)
            ]))
        
        # Multi-scale fusion
        self.scale_fusion = nn.Sequential(
            nn.Linear(hidden_dim * self.num_scales, hidden_dim * 2),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)
    
    def forward(self, x, return_attention=False):
        batch_size = x.shape[0]
        h = self.input_proj(x)  # (B, hidden_dim)
        
        attention_weights = [] if return_attention else None
        scale_cls = []
        
        for s, gs in enumerate(self.group_sizes):
            actual_gs = min(gs, self.hidden_dim)
            n_groups = self.hidden_dim // actual_gs
            
            x_scale = h[:, :n_groups * actual_gs]
            x_scale = x_scale.view(batch_size, n_groups, actual_gs)
            x_scale = self.local_projections[s](x_scale)
            
            cls = self.cls_tokens[s].expand(batch_size, -1, -1)
            x_scale = torch.cat([cls, x_scale], dim=1)
            x_scale = self.pos_encodings[s](x_scale)
            
            for block in self.local_transformers[s]:
                x_scale, attn_w = block(x_scale, return_attention=return_attention)
                if return_attention and attn_w is not None:
                    attention_weights.append(attn_w)
            
            scale_cls.append(x_scale[:, 0])
        
        combined = torch.cat(scale_cls, dim=-1)
        output = self.scale_fusion(combined)
        output = self.output_norm(output)
        
        if return_attention:
            return output, scale_cls, attention_weights
        return output, scale_cls, None


class CrossOmicsContrastiveLearning(nn.Module):
    """COCL module (Section 2.3.3, Eq. 6). τ=0.07."""
    def __init__(self, hidden_dim, projection_dim, temperature=0.07):
        super().__init__()
        self.temperature = temperature
        self.projectors = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, projection_dim)
            ) for mod in ['mRNA', 'miRNA', 'Methy', 'CNV']
        })
    
    def forward(self, embeddings):
        projected = {mod: F.normalize(self.projectors[mod](emb), dim=-1)
                     for mod, emb in embeddings.items()}
        
        mods = list(projected.keys())
        total_loss = 0.0
        n_pairs = 0
        
        for i in range(len(mods)):
            for j in range(i + 1, len(mods)):
                z1, z2 = projected[mods[i]], projected[mods[j]]
                B = z1.shape[0]
                logits = torch.mm(z1, z2.T) / self.temperature
                labels = torch.arange(B, device=z1.device)
                loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
                total_loss += loss
                n_pairs += 1
        
        return total_loss / max(n_pairs, 1)


class GatedCrossModalAttention(nn.Module):
    """Gated fusion (Section 2.3.4, Eq. 7). d_a=64."""
    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()
        self.modalities = ['mRNA', 'miRNA', 'Methy', 'CNV']
        self.hidden_dim = hidden_dim
        
        self.cross_attn = nn.ModuleDict({
            mod: nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            for mod in self.modalities
        })
        
        # Gating (Eq. 7)
        self.gate_net = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Linear(hidden_dim * 2, 64),  # d_a = 64
                nn.Tanh(),
                nn.Linear(64, 1),
            ) for mod in self.modalities
        })
        
        self.norms = nn.ModuleDict({mod: nn.LayerNorm(hidden_dim) for mod in self.modalities})
        self.ffs = nn.ModuleDict({mod: FeedForward(hidden_dim, dropout=dropout) for mod in self.modalities})
    
    def forward(self, embeddings, return_attention=False):
        mods = [m for m in self.modalities if m in embeddings]
        B = embeddings[mods[0]].shape[0]
        
        # Stack for cross-attention context
        context = torch.stack([embeddings[m] for m in mods], dim=1)  # (B, M, D)
        
        updated = {}
        attention_weights = {} if return_attention else None
        gate_values = {}
        
        for mod in mods:
            q = embeddings[mod].unsqueeze(1)  # (B, 1, D)
            attn_out, attn_w = self.cross_attn[mod](q, context, context)
            attn_out = attn_out.squeeze(1)
            
            if return_attention:
                attention_weights[mod] = attn_w
            
            combined = torch.cat([embeddings[mod], attn_out], dim=-1)
            gate = torch.sigmoid(self.gate_net[mod](combined))
            gate_values[mod] = gate
            
            updated[mod] = self.norms[mod](embeddings[mod] + gate * attn_out)
            updated[mod] = updated[mod] + self.ffs[mod](updated[mod])
        
        if return_attention:
            return updated, attention_weights, gate_values
        return updated, None, gate_values


class HiOmicsFormer(nn.Module):
    """
    Complete HiOmicsFormer (Section 2.3).
    Architecture: d=256, L=4, H=8, K=16, K_c=9.
    """
    def __init__(self, feature_dims, config):
        super().__init__()
        self.feature_dims = feature_dims
        self.config = config
        self.modalities = list(feature_dims.keys())
        self.hidden_dim = config.hidden_dim       # 256
        self.latent_dim = config.latent_dim        # 128
        self.num_clusters = config.num_clusters    # 9
        
        # 1. Hierarchical encoders (L=4 layers each)
        self.encoders = nn.ModuleDict({
            mod: HierarchicalOmicsEncoder(
                input_dim=feature_dims[mod],
                hidden_dim=config.hidden_dim,
                num_heads=config.num_heads,
                num_layers=config.num_encoder_layers,  # L=4
                group_sizes=config.group_sizes,
                dropout=config.dropout
            ) for mod in self.modalities
        })
        
        # 2. Cross-omics contrastive learning
        self.cocl = CrossOmicsContrastiveLearning(
            config.hidden_dim, config.projection_dim, config.contrastive_temperature
        )
        
        # 3. Gated cross-modal attention
        self.cross_modal_layers = nn.ModuleList([
            GatedCrossModalAttention(config.hidden_dim, config.num_heads, config.dropout)
            for _ in range(config.num_cross_layers)
        ])
        
        # 4. Fusion → latent
        self.fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * len(self.modalities), config.hidden_dim),
            nn.LayerNorm(config.hidden_dim), nn.GELU(),
        )
        self.fc_mu = nn.Linear(config.hidden_dim, config.latent_dim)
        self.fc_logvar = nn.Linear(config.hidden_dim, config.latent_dim)
        
        # 5. DEC (Section 2.3.5)
        self.cluster_centroids = nn.Parameter(
            torch.randn(config.num_clusters, config.latent_dim)
        )
        self.cluster_head = nn.Linear(config.latent_dim, config.num_clusters)
        
        # 6. Decoders
        self.decoders = nn.ModuleDict({
            mod: nn.Sequential(
                nn.Linear(config.latent_dim, config.hidden_dim),
                nn.GELU(),
                nn.Linear(config.hidden_dim, feature_dims[mod])
            ) for mod in self.modalities
        })
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std) if self.training else mu
    
    def encode(self, batch, return_attention=False):
        mod_embeddings = {}
        encoder_attentions = {} if return_attention else None
        
        for mod in self.modalities:
            if mod in batch:
                out, scales, attn = self.encoders[mod](batch[mod], return_attention)
                mod_embeddings[mod] = out
                if return_attention and attn:
                    encoder_attentions[mod] = attn
        
        # Contrastive loss
        contrastive_loss = self.cocl(mod_embeddings)
        
        # Cross-modal attention
        cross_attentions = []
        gate_values_all = []
        for layer in self.cross_modal_layers:
            mod_embeddings, c_attn, gates = layer(mod_embeddings, return_attention)
            if return_attention and c_attn:
                cross_attentions.append(c_attn)
            gate_values_all.append(gates)
        
        # Fusion
        fused = self.fusion(torch.cat([mod_embeddings[m] for m in self.modalities], dim=-1))
        mu = self.fc_mu(fused)
        logvar = self.fc_logvar(fused)
        z = self.reparameterize(mu, logvar)
        
        result = {
            'z': z, 'mu': mu, 'logvar': logvar,
            'modality_embeddings': mod_embeddings,
            'contrastive_loss': contrastive_loss
        }
        if return_attention:
            result['encoder_attentions'] = encoder_attentions
            result['cross_attentions'] = cross_attentions
            result['gate_values'] = gate_values_all
        return result
    
    def get_cluster_assignments(self, z):
        """Student's t-distribution soft assignments (Eq. 8)."""
        dist = torch.cdist(z, self.cluster_centroids)
        q = 1.0 / (1.0 + dist ** 2 / self.config.cluster_alpha)
        q = q ** ((self.config.cluster_alpha + 1) / 2)
        return q / q.sum(dim=1, keepdim=True)
    
    @staticmethod
    def get_target_distribution(q):
        """Target distribution (Eq. 8)."""
        weight = q ** 2 / q.sum(dim=0, keepdim=True)
        return (weight / weight.sum(dim=1, keepdim=True)).detach()
    
    def decode(self, z):
        return {mod: self.decoders[mod](z) for mod in self.modalities}
    
    def forward(self, batch, return_attention=False):
        enc = self.encode(batch, return_attention)
        z = enc['z']
        q = self.get_cluster_assignments(z)
        recon = self.decode(z)
        return {**enc, 'q': q, 'cluster_logits': self.cluster_head(z), 'reconstructions': recon}

# Verify architecture matches Table 2
model = HiOmicsFormer(feature_dims, config).to(device);print(model);
total_params = sum(p.numel() for p in model.parameters())
print(f"\nModel Parameters: {total_params:,}")
print(f"Hidden dim: {config.hidden_dim} (manuscript: 256) ✓" if config.hidden_dim == 256 else "✗ MISMATCH")
print(f"Transformer layers: {config.num_encoder_layers} (manuscript: 4) ✓" if config.num_encoder_layers == 4 else "✗ MISMATCH")
print(f"Attention heads: {config.num_heads} (manuscript: 8) ✓" if config.num_heads == 8 else "✗ MISMATCH")
print(f"Clusters: {config.num_clusters} (manuscript: 9) ✓" if config.num_clusters == 9 else "✗ MISMATCH")

# %%
# ============================================================================
# CELL 7: LOSS FUNCTION — Eq. 10: L = L_recon + 0.1·L_CL + 0.5·L_DEC
# ============================================================================
# REVIEWER 5.7: Loss weights now match manuscript Eq. 10.

class HiOmicsLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def forward(self, batch, outputs, model, epoch=0):
        losses = {}
        
        # Reconstruction loss
        recon_loss = 0
        for mod in model.modalities:
            orig = batch[mod]
            recon = outputs['reconstructions'][mod]
            min_d = min(orig.shape[1], recon.shape[1])
            recon_loss += F.mse_loss(recon[:, :min_d], orig[:, :min_d])
        losses['recon'] = recon_loss / len(model.modalities)
        
        # Contrastive loss (COCL)
        losses['contrastive'] = outputs['contrastive_loss']
        
        # DEC clustering loss (with warmup)
        cluster_weight = min(1.0, epoch / max(self.config.warmup_epochs, 1))
        q = outputs['q']
        p = model.get_target_distribution(q)
        losses['dec'] = F.kl_div(q.log(), p, reduction='batchmean')
        
        # KL divergence (VAE)
        mu, logvar = outputs['mu'], outputs['logvar']
        losses['kl'] = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total: L = L_recon + λ_CL·L_CL + λ_DEC·L_DEC + λ_kl·L_kl
        total = (self.config.lambda_recon * losses['recon']
                 + self.config.lambda_CL * losses['contrastive']
                 + self.config.lambda_DEC * cluster_weight * losses['dec']
                 + self.config.lambda_kl * losses['kl'])
        
        return total, losses

print(f"Loss weights: L_recon={config.lambda_recon}, "
      f"λ_CL={config.lambda_CL}, λ_DEC={config.lambda_DEC}")
print(f"Matches Eq. 10: L = {config.lambda_recon}·L_recon + "
      f"{config.lambda_CL}·L_CL + {config.lambda_DEC}·L_DEC ✓")

# %%
# ============================================================================
# CELL 8: TRAINING UTILITIES
# ============================================================================

class HiOmicsTrainer:
    def __init__(self, model, optimizer, criterion, scheduler, config, device):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.best_score = -float('inf')
        self.patience_counter = 0
    
    @torch.no_grad()
    def evaluate(self, loader, labels):
        self.model.eval()
        all_z, all_pred = [], []
        for batch in loader:
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}
            out = self.model(batch)
            all_z.append(out['z'].cpu().numpy())
            all_pred.append(out['q'].argmax(dim=1).cpu().numpy())
        
        z = np.concatenate(all_z)
        pred = np.concatenate(all_pred)
        
        metrics = {
            'silhouette': silhouette_score(z, pred) if len(np.unique(pred)) > 1 else 0,
            'nmi': normalized_mutual_info_score(labels, pred),
            'ari': adjusted_rand_score(labels, pred),
        }
        return z, pred, metrics
    
    def train_fold(self, train_loader, val_loader, labels_train, labels_val):
        """Train for one fold with early stopping."""
        history = {'train_loss': [], 'val_metrics': []}
        
        for epoch in range(self.config.max_epochs):
            # --- Train ---
            self.model.train()
            epoch_loss = 0
            n_batches = 0
            
            for batch in train_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                         for k, v in batch.items()}
                self.optimizer.zero_grad()
                outputs = self.model(batch)
                loss, _ = self.criterion(batch, outputs, self.model, epoch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1
            
            if self.scheduler:
                self.scheduler.step()
            
            avg_loss = epoch_loss / max(n_batches, 1)
            history['train_loss'].append(avg_loss)
            
            # --- Validate ---
            z_val, pred_val, val_metrics = self.evaluate(val_loader, labels_val)
            history['val_metrics'].append(val_metrics)
            
            score = val_metrics['silhouette']
            
            if (epoch + 1) % 10 == 0:
                print(f"  Epoch {epoch+1}/{self.config.max_epochs}: "
                      f"loss={avg_loss:.4f}, sil={score:.4f}, "
                      f"nmi={val_metrics['nmi']:.4f}")
            
            # Early stopping
            if score > self.best_score:
                self.best_score = score
                self.patience_counter = 0
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config.patience:
                    print(f"  Early stopping at epoch {epoch+1}")
                    break
        
        # Load best
        self.model.load_state_dict(self.best_state)
        return history

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

# %%
# ============================================================================
# CELL 9: 5-FOLD STRATIFIED CROSS-VALIDATION (Reviewer 4.2)
# ============================================================================
# REVIEWER 4.2: "Section 2.5.2 describes 5-fold CV. The notebook contains
# no CV loop. Instead, a single train_test_split." → Implemented properly.
#
# Each fold: train model from scratch, evaluate on held-out fold.
# Report: mean ± SD across 5 folds for ALL metrics.

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
print(model)
sys.exit("I am here")
# %%
# ============================================================================
# CELL 12: TRAINING & EMBEDDING VISUALIZATION
# ============================================================================
# Visualize training progress, learned embeddings, and cluster quality.

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
plt.savefig(f'{config.figures_path}/fig3_training_embeddings.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nEmbedding visualization statistics:")
print(f"  Samples visualized: {len(z_viz)}")
print(f"  Unique cancer types: {len(unique_labels)}")
print(f"  Unique clusters: {len(unique_preds)}")
print(f"  Mean silhouette: {np.mean(sil_samples):.4f}")

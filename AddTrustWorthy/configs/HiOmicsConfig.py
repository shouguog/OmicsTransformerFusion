from dataclasses import dataclass, field
@dataclass
class HiOmicsConfig:
    # --- Paths (adjust to your environment) ---
    data_path: str = './data/MLOmics/Main_Dataset'
    results_path: str = './results_revision'
    figures_path: str = './figures_revision'

    # --- Dataset ---
    modalities: list = field(default_factory=lambda: ['mRNA', 'miRNA', 'Methy', 'CNV'])
    use_all_cancers: bool = True  # Rev. 4.1: ALL available cancer types
    data_variant: str = 'Original'  # 'Original' (full features), 'Top', or 'Aligned'
    feature_selection_top_k: int = 2000  # Rev. 5.2: Explicitly documented

    # --- Architecture (Table 2) ---
    hidden_dim: int = 256  # Rev. 4.8: d = 256 (was overridden to 128)
    latent_dim: int = 128
    num_encoder_layers: int = 4  # Rev. 4.8: L = 4 (was overridden to 2)
    num_heads: int = 8  # H = 8
    num_feature_clusters: int = 16  # K = 16
    group_sizes: tuple = (16, 32, 64)  # Hierarchical group sizes
    dropout: float = 0.1
    num_cross_layers: int = 2
    num_fusion_layers: int = 2
    projection_dim: int = 128

    # --- Clustering ---
    num_clusters: int = 9  # Kc = 9 (updated dynamically after data loading)
    cluster_alpha: float = 1.0

    # --- Contrastive Learning ---
    contrastive_temperature: float = 0.07  # τ = 0.07

    # --- Loss Weights (Eq. 10) ---
    lambda_recon: float = 1.0  # Rev. 5.7: L_recon coefficient
    lambda_CL: float = 0.1  # λ_CL = 0.1
    lambda_DEC: float = 0.5  # λ_DEC = 0.5
    lambda_kl: float = 0.01

    # --- Training ---
    learning_rate: float = 1e-4  # Rev. 5.8: lr = 10^-4
    weight_decay: float = 1e-5
    batch_size: int = 256  # Rev. 5.6: batch_size = 256
    max_epochs: int = 200  # Max 200
    patience: int = 20  # Early stopping patience 20
    warmup_epochs: int = 10

    # --- Cross-Validation (Rev. 4.2) ---
    n_folds: int = 5  # 5-fold CV

    # --- Biomarkers ---
    top_k_biomarkers: int = 50  # Per modality → 200 total

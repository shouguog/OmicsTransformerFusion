import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
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
                data_dir / f"{cancer_code}_{mod}.csv",  # BRCA_mRNA.csv (fallback)
                data_dir / f"{mod}_{cancer_code}{self.suffix}.csv",  # mRNA_BRCA_top.csv
                data_dir / f"{mod}_{cancer_code}.csv",  # mRNA_BRCA.csv
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

        print(f"\n{'=' * 70}")
        print(f"DATA LOADING COMPLETE")
        print(f"{'=' * 70}")
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


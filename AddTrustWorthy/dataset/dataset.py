import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
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

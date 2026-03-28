import torch.nn as nn
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    silhouette_score, normalized_mutual_info_score, adjusted_rand_score
)
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
                print(f"  Epoch {epoch + 1}/{self.config.max_epochs}: "
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
                    print(f"  Early stopping at epoch {epoch + 1}")
                    break

        # Load best
        self.model.load_state_dict(self.best_state)
        return history

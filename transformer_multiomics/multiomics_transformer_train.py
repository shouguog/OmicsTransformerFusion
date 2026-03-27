# %% [markdown]
# # Predictive Modelling of Proteomics using a Transformer Model
# 
# ## Task Type and Model Decision
# I selected a Transformer model because the multi-head attention can learn different types of relationships between omics data features and capturing various correlations that might exist. The self-attention mechanism allows the model to identify important feature interactions, which I think are important in multi-omics data integration. While VAEs would also be suitable, particularly for handling noise and missing values, the potential complex interactions between omics data features made Transformers my preferred choice.
# 
# ## Data
# The aim is to use multiple input omics datasets to predict proteomics data. Input omics data was selected by iteratively testing combinations to determine which best predicted proteomics. This indicated that transcriptomics data only as the optimal input combination for proteomics prediction.
# 
# ## Plan
# The approach follows a systematic comparison between a baseline MLP model and the Transformer architecture:
# 
# #### In multiomics_data_analysis.ipynb
# 1. [Inspect data](#inspect-data);
# 
# #### In multiomics_transformer_train.ipynb
# 2. [Create a simple MLP model](#baseline-mlp) as a baseline;
# 3. [Develop Transformer models](#transformer-models) with different fusion strategies and activation functions;
# 4. [Perform hyperparameter tuning](#hyperparameter-tuning);
# 5. [Conduct progressive input omics selection](#omics-selection);
# 6. [Create comprehensive loop](#evaluation-loop) to iterate through steps 3-5, obtaining metrics to determine the best Transformer model for comparison with the baseline MLP;
# 
# #### In multiomics_transformer_performance_analysis.ipynb
# 7. [Select best-performing model](#best-performing);
# 8. [Analyse best-predicted proteomic features](#feature-analysis);
# 
# %%
import torch
import torch.nn as nn
import torch.optim as optim
# %%
# for using VSCode
import sys
from pathlib import Path

# Add the parent directory of transformer_multiomics to sys.path
sys.path.append(str(Path.cwd().parent))
# %%
from config import DATA_PATH, MODEL_PATH, RESULT_PATH
from data_loader import (
    ensure_directories_exist,
    load_datasets,
)
from data_preparation import prepare_data_loaders
from models.mlp import MultiOmicsMLP
from models.transformer import MultiOmicsTransformer
from training.evaluater import evaluate_model
from training.trainer import train_model
from utils.plots import plot_error_distribution, plot_loss_curves
# %%
# CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# %%
# Ensure directories exist
ensure_directories_exist(MODEL_PATH, RESULT_PATH, DATA_PATH)

# Load datasets
datasets = load_datasets()
print("Available datasets:", datasets.keys())
# %%
input_datasets = ["transcriptomics", "methylation", "metabolomics", "cnv"]
target_dataset = "proteomics"

train_loader, val_loader, test_loader, input_dims_dict, output_dim = prepare_data_loaders(
    omics_set=input_datasets,
    datasets=datasets,
    target_dataset=target_dataset,
    batch_size=32,
    test_size=0.2,
)
# %%
# Initialize model with multi-omics input dims
mlp_model = MultiOmicsMLP(input_dims_dict=input_dims_dict, output_dim=output_dim, hidden_dims=[256, 128, 64]).to(device)

# Loss function and optimiser
mlp_criterion = nn.MSELoss()
learning_rate = 0.001
weight_decay = 0
mlp_optimiser = optim.Adam(mlp_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Early stopping setup
patience = 15
epochs = 100

mlp_model = MultiOmicsMLP(input_dims_dict=input_dims_dict, output_dim=output_dim, hidden_dims=[256, 128, 64]).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(mlp_model.parameters(), lr=0.001, weight_decay=0)

trained_model, train_losses, test_losses, best_loss, early_stop_epoch, final_attention_weights = train_model(
    model=mlp_model,
    train_loader=train_loader,
    test_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=epochs,
    patience=patience,
    model_name="multiomics_mlp",
    save_path=MODEL_PATH,
    return_attention_weights=False,
)
# %%
# Plot training and validation loss curves
early_stopping_epoch = len(test_losses) - patience
plot_loss_curves(
    train_losses,
    test_losses,
    early_stopping_epoch=early_stopping_epoch,
    title="Transformer: Training and Validation Loss",
)
# %%
# Evaluate model performance
metrics_mlp = evaluate_model(mlp_model, test_loader, device, is_transformer=False, model_name="MLP")
# %%
# Plot error distribution
all_predictions = metrics_mlp["predictions"]
all_targets = metrics_mlp["targets"]

plot_error_distribution(all_predictions, all_targets, model_name="MLP")
# %% [markdown]
# # Baseline MLP Results <a id="results-MLP"></a>
# 
# The MLP is performing quite well already, with R^2 of ~0.83 and the error distribution Normally distributed, without systematically overestimating/underestimating the target values.
# %% [markdown]
# # Transformer <a id="transformer-models"></a> 
# 
# This part is the development of the Transformer model. The Transformer model is designed so that it can be used to iteratively scan through different combinations of input omics data, fusion strategies, activation function (only ReLU and GeLU, as I'm working with PyTorch), and for hyperparameter optimisation.
# %% [markdown]
# # Training
# %%
# Initialize model with multi-omics input dims
transformer_model = MultiOmicsTransformer(
    input_dims=input_dims_dict,
    output_dim=output_dim,
    hidden_dim=256,  # 256, 512, -> Larger embedding space
    num_heads=4,  # 4, 8, 16 -> More attention heads
    num_layers=4,  # 4, 8 -> Deeper transformer
    dropout=0.01,  # Regularization
    use_batch_norm=False,
    use_input_norm=False,
    activation_fn="gelu",  # "gelu", "relu", "identity"
    use_modality_embedding=False,
    pooling_type="attention",  # "attention", "mean", "max", "concat"
).to(device)

# Loss function and optimiser
transformer_criterion = nn.MSELoss()
learning_rate = 0.001
weight_decay = 0
transformer_optimiser = optim.AdamW(transformer_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Early stopping setup
patience = 20
epochs = 100

transformer_model = MultiOmicsTransformer(
    input_dims=input_dims_dict,
    output_dim=output_dim,
    hidden_dim=256,
    num_heads=4,
    num_layers=4,
    dropout=0.01,
    use_batch_norm=False,
    use_input_norm=False,
    activation_fn="gelu",
    use_modality_embedding=False,
    pooling_type="attention",
).to(device)

criterion = nn.MSELoss()
optimizer = optim.AdamW(transformer_model.parameters(), lr=0.001, weight_decay=0)

result = train_model(
    model=transformer_model,
    train_loader=train_loader,
    test_loader=test_loader,
    criterion=criterion,
    optimizer=optimizer,
    device=device,
    epochs=100,
    patience=20,  # Different patience for transformer
    model_name="multiomics_transformer",
    save_path=MODEL_PATH,
    return_attention_weights=True,  # Transformer returns attention weights
)

# Unpack results - attention weights are returned as the 5th element
trained_model, train_losses, test_losses, best_loss, early_stop_epoch, attention_weights = result
# %%
# Plot training and validation loss curves
early_stopping_epoch = len(test_losses) - patience
plot_loss_curves(
    train_losses,
    test_losses,
    early_stopping_epoch=early_stopping_epoch,
    title="Transformer: Training and Validation Loss",
)
# %%
# Evaluate model performance
metrics_transformer = evaluate_model(
    transformer_model,
    test_loader,
    device,
    is_transformer=True,
    model_name="Transformer",
)
# %%
# Plot error distribution
all_predictions = metrics_transformer["predictions"]
all_targets = metrics_transformer["targets"]

plot_error_distribution(all_predictions, all_targets, model_name="Transformer")
# %%
# global importance of each omics type
# Concatenate and average across samples
attn_tensor = torch.cat(attention_weights, dim=0)  # shape: [num_samples, num_modalities]
avg_attention = attn_tensor.mean(dim=0)

print("Average attention per modality:", avg_attention)
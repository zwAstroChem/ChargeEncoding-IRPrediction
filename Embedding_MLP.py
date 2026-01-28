import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
import json
import pickle
import time
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.model_selection import KFold
from rdkit import rdBase

# ---------------- Fix Random Seed ----------------
# Set random seeds for reproducibility across all libraries
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
rdBase.DisableLog('rdApp.warning')  # Suppress RDKit warnings


# ---------------- Dataset Class ----------------
class PAHDataset(Dataset):
    """Custom PyTorch Dataset for handling molecular data.
    Stores SMILES fingerprints, target spectra, and charge states."""
    
    def __init__(self, smiles, labels, charges):
        self.smiles = smiles  # ECFP fingerprints
        self.labels = labels  # Target spectra (IR spectra)
        self.charges = charges  # Molecular charge states

    def __getitem__(self, index):
        """Returns a single data sample as a dictionary."""
        return {
            'smi': torch.tensor(self.smiles[index]),      # Molecular fingerprint
            'label': torch.tensor(self.labels[index]),    # Target spectrum
            'charge': torch.tensor(self.charges[index])   # Charge state
        }

    def __len__(self):
        """Returns total number of samples in dataset."""
        return len(self.smiles)


# ---------------- Loss Function ----------------
class EMDLoss(nn.Module):
    """Earth Mover's Distance (EMD) loss for comparing probability distributions.
    Specifically designed for comparing normalized spectral predictions."""
    
    def __init__(self):
        super(EMDLoss, self).__init__()

    @staticmethod
    def forward(pred, label):
        """Calculates EMD between predicted and target spectra.
        
        Args:
            pred: Predicted spectrum tensor
            label: Ground truth spectrum tensor
            
        Returns:
            EMD loss value (scalar)
        """
        diff = pred - label  # Element-wise difference
        cumsum_diff = torch.cumsum(diff, dim=1)  # Cumulative sum along wavelength dimension
        loss = torch.sum(torch.abs(cumsum_diff))  # Sum absolute values for batch
        return loss


# ---------------- Model Definition ----------------
class Predictor(nn.Module):
    """Main neural network model for spectral prediction.
    Combines molecular fingerprints with charge embeddings via learnable embedding layer."""
    
    def __init__(self, input_dim, layers_dim, output_dim, charge_vocab=None, charge_dim=16):
        """
        Args:
            input_dim: Dimension of ECFP fingerprint input
            layers_dim: List of hidden layer dimensions (e.g., [1500, 1000, 800, 600])
            output_dim: Dimension of output spectrum (number of wavelength points)
            charge_vocab: List of possible charge values (default: [-1, 0, 1, 2])
            charge_dim: Embedding dimension for charge states
        """
        super(Predictor, self).__init__()
        self.layers = nn.ModuleList()

        # Define charge vocabulary and embedding layer
        if charge_vocab is not None:
            self.charge_vocab = charge_vocab
        else:
            self.charge_vocab = [-1, 0, 1, 2]  # Default: negative, neutral, +1, +2
            print("Using default charge vocabulary:", self.charge_vocab)
        self.charge_embedding = nn.Embedding(len(self.charge_vocab), charge_dim)

        # Calculate total input dimension after concatenation
        total_dim = input_dim + charge_dim

        # Build neural network layers
        # First layer: concatenated fingerprint + charge embedding ¡ú first hidden layer
        self.layers.append(nn.Linear(total_dim, layers_dim[0]))
        self.layers.append(nn.ReLU())

        # Hidden layers
        for i in range(0, len(layers_dim) - 1):
            self.layers.append(nn.Linear(layers_dim[i], layers_dim[i + 1]))
            self.layers.append(nn.ReLU())

        # Output layer: final hidden layer ¡ú spectrum prediction
        self.layers.append(nn.Linear(layers_dim[-1], output_dim))

        # Initialize EMD loss function
        self.loss_1 = EMDLoss()

    @staticmethod
    def normalize(A, B):
        """Normalizes tensors to probability distributions (sum to 1).
        
        Args:
            A, B: Input tensors (predictions and labels)
            
        Returns:
            Normalized tensors where each row sums to 1
        """
        # Ensure tensors are float type
        if isinstance(A, torch.Tensor):
            if A.dtype != torch.float:
                A = A.type(torch.float)
        else:
            A = torch.tensor(A, dtype=torch.float)
            
        if isinstance(B, torch.Tensor):
            if B.dtype != torch.float:
                B = B.type(torch.float)
        else:
            B = torch.tensor(B, dtype=torch.float)
            
        # Add batch dimension if needed
        if A.dim() == 1:
            A = A.unsqueeze(0)
        if B.dim() == 1:
            B = B.unsqueeze(0)
            
        # Normalize to probability distributions
        normed_A = A / A.sum(dim=1, keepdim=True)
        normed_B = B / B.sum(dim=1, keepdim=True)
        return normed_A, normed_B

    def forward(self, X, Y, charges):
        """Forward pass through the network.
        
        Args:
            X: ECFP fingerprint tensor
            Y: Target spectrum tensor
            charges: List of charge values (e.g., [-1, 0, 1, 2])
            
        Returns:
            pred: Predicted spectrum
            emd_loss: EMD loss between normalized prediction and target
        """
        # Convert charge values to indices for embedding lookup
        charge_indices = torch.tensor([self.charge_vocab.index(i) for i in charges]).to(X.device)

        # Get charge embeddings (learnable representation)
        charge_emb = self.charge_embedding(charge_indices)

        # Ensure correct data type
        X = X.float()
        charge_emb = charge_emb.float()

        # Concatenate molecular fingerprint with charge embedding
        X = torch.cat((X, charge_emb), dim=-1)

        # Pass through all layers
        for layer in self.layers:
            X = layer(X)
        pred = torch.abs(X)  # Ensure non-negative predictions (spectra)

        # Calculate EMD loss on normalized spectra
        normed_pred, normed_Y = self.normalize(pred, Y)
        emd_loss = self.loss_1(normed_pred, normed_Y)

        return pred, emd_loss


class EarlyStopping:
    """Early stopping callback to prevent overfitting.
    Monitors validation loss and stops training when no improvement is seen."""
    
    def __init__(self, patience=50, min_delta=0.005, verbose=False):
        """
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change in loss to qualify as improvement
            verbose: Whether to print progress messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.best_weights = None  # Store best model weights

    def __call__(self, model, current_loss):
        """Check if training should stop.
        
        Returns:
            True if early stopping condition met, False otherwise
        """
        if current_loss < self.best_loss - self.min_delta:
            # Improvement detected
            self.best_loss = current_loss
            self.patience_counter = 0
            self.best_weights = model.state_dict()  # Save best weights
            if self.verbose:
                print(f'Improvement detected: best_loss updated to {self.best_loss:.4f}')
        else:
            # No improvement
            self.patience_counter += 1
            if self.verbose:
                print(f'No improvement: patience_counter = {self.patience_counter}')

        # Stop if patience exceeded
        if self.patience_counter >= self.patience:
            if self.verbose:
                print(f"Early stopping triggered after {self.patience_counter} epochs without improvement.")
            return True
        return False


# Hyperparameters for fingerprint generation
radius = 11  # Radius for Morgan fingerprint (larger = more substructural information)
Nmin = 4     # Minimum occurrence count for fingerprint bits to be included


# ---------------- Data Generation Function ----------------
def generate_data(smiles, labels, charges, uids, train_indexes, radius=radius, Nmin=Nmin):
    """Generates ECFP fingerprints from SMILES strings.
    
    Args:
        smiles: List of SMILES strings
        labels: List of target spectra
        charges: List of charge states
        uids: List of unique molecule IDs
        train_indexes: Indices of training samples (used for feature selection)
        radius: Morgan fingerprint radius
        Nmin: Minimum frequency for fingerprint bits
        
    Returns:
        smiles2fp: Matrix of fingerprints (samples ¡Á features)
        morganfp_features: List of selected fingerprint bit indices
    """
    num_samples = len(smiles)
    morganfps = np.empty(num_samples, dtype=object)
    features_counter = defaultdict(int)
    
    # Generate fingerprints for all molecules
    for idx in range(0, num_samples):
        curr_smiles = smiles[idx]
        mol = Chem.MolFromSmiles(curr_smiles)
        fp = AllChem.GetMorganFingerprint(mol, radius)  # Generate Morgan fingerprint
        morganfps[idx] = fp
        
        # Count feature occurrences in training set only
        if idx in train_indexes:
            for feature in [*fp.GetNonzeroElements()]:
                features_counter[feature] += 1
    
    # Output fragments count before Nmin filtering
    print(f"Molecular fragments before Nmin filtering: {len(features_counter)}")

    # Select features that appear at least Nmin times in training set
    morganfp_features = np.empty(len(features_counter.keys()), dtype=object)
    cnt = 0
    for feature in features_counter.keys():
        if features_counter[feature] >= Nmin:
            morganfp_features[cnt] = feature
            cnt += 1
            
    # Output fragments count after Nmin filtering
    print(f"Molecular fragments after Nmin filtering (¡Ý{Nmin} occurrences): {cnt}")

    morganfp_features = list(morganfp_features[:cnt])

    # Create fingerprint matrix
    num_features = len(morganfp_features)
    smiles2fp = np.zeros((num_samples, num_features))
    for idx in range(num_samples):
        for feature in [*morganfps[idx].GetNonzeroElements()]:
            if feature in morganfp_features:
                pos = morganfp_features.index(feature)
                smiles2fp[idx][pos] = morganfps[idx][feature]
    return smiles2fp, morganfp_features


# ---------------- Data Loading ----------------
# Load combined dataset containing SMILES, spectra, charges, and UIDs
data_path = r"./torch_spectrum_12599.json"
combined_data = json.load(open(data_path, 'r'))
print(f"Number of molecules in dataset: {len(combined_data)}")

# Extract data from JSON structure
UID_list = list(combined_data.keys())
smiles = []
labels = []
charges = []
uids = []
for item in UID_list:
    smiles.append(combined_data[item]["smi"])
    labels.append(combined_data[item]["spectra"])
    charges.append(int(combined_data[item]["charge"]))
    uids.append(int(combined_data[item]["UID"]))
    
# Convert to numpy arrays and select spectral region (245:300 = specific wavelength range)
smiles = np.array(smiles)
labels = np.array(labels)[:, 245:300]  # Select wavelength range 245-300
charges = np.array(charges)
uids = np.array(uids)


# ================== Start 5-Fold Split ==================
# Create non-random 5-fold split for cross-validation
# Note: This is NOT random shuffle - indices are split in order

n_samples = len(smiles)
indices = np.arange(n_samples)

# Step 1: Shuffle molecules randomly
shuffled_indices = np.random.permutation(indices)

# Step 2: Split into 5 folds (A, B, C, D, E)
fold_size = n_samples // 5
remainder = n_samples % 5

splits = []
start = 0
for i in range(5):
    # Distribute remainder samples evenly across first few folds
    size = fold_size + (1 if i < remainder else 0)
    splits.append(shuffled_indices[start:start + size])
    start += size

print("Fold sizes:", [len(block) for block in splits])
# ================== Split End ==================


# ---------------- 5-Fold Cross Validation ----------------
all_fold_results = {}  # Store results from all folds

for fold in range(1, 6):  # fold = 1,2,3,4,5
    print(f"\n{'='*10} Fold {fold} {'='*10}")

    # Define indexes for current fold using rotating scheme:
    # Test: current block (fold-1)
    # Validation: next block (fold % 5)
    # Train: remaining three blocks
    test_indexes = splits[fold - 1]  # Current block for test
    val_indexes = splits[fold % 5]   # Next block for validation

    # Training set: All blocks except test and validation
    train_blocks = []
    for i in range(5):
        if i != (fold - 1) and i != (fold % 5):
            train_blocks.append(splits[i])
    train_indexes = np.concatenate(train_blocks)

    # Print split statistics
    print(f"Train set size: {len(train_indexes)}")
    print(f"Val set size: {len(val_indexes)}")
    print(f"Test set size: {len(test_indexes)}")
    
    # Verify no data leakage between sets
    train_set = set(train_indexes)
    test_set = set(test_indexes)
    val_set = set(val_indexes)

    print(f"Train/Test overlap: {len(train_set & test_set)}")
    print(f"Train/Val overlap: {len(train_set & val_set)}")
    print(f"Val/Test overlap: {len(val_set & test_set)}")

    # Generate fingerprint features using training set for feature selection
    smiles2fp, morganfp_features = generate_data(smiles, labels, charges, uids, train_indexes, radius=radius, Nmin=Nmin)

    # Organize all data into dictionary
    DATA = {
        "smiles_train": smiles[train_indexes],
        "smiles_val": smiles[val_indexes],
        "smiles_test": smiles[test_indexes],
        "X_train": smiles2fp[train_indexes],
        "X_valid": smiles2fp[val_indexes],
        "X_test": smiles2fp[test_indexes],
        "Y_train": labels[train_indexes],
        "Y_valid": labels[val_indexes],
        "Y_test": labels[test_indexes],
        "charges_train": charges[train_indexes],
        "charges_valid": charges[val_indexes],
        "charges_test": charges[test_indexes],
        "morganfp_features": morganfp_features,
        "radius": radius,
        "Nmin": Nmin,
        "UID_train": uids[train_indexes],
        "UID_valid": uids[val_indexes],
        "UID_test": uids[test_indexes]
    }

    # Save preprocessed data for this fold
    with open(f"DATA_fold{fold}.pck", "wb") as f:
        pickle.dump(DATA, f)

    # Debug: Check charge distribution in each set
    print(f"[Fold {fold}] Train first 10 charges: {DATA['charges_train'][:10]}")
    print(f"[Fold {fold}] Val first 10 charges: {DATA['charges_valid'][:10]}")
    print(f"[Fold {fold}] Test first 10 charges: {DATA['charges_test'][:10]}")

    print(f"[Fold {fold}] Train X size: {len(DATA['X_train'])}")
    print(f"[Fold {fold}] Val X size: {len(DATA['X_valid'])}")
    print(f"[Fold {fold}] Test X size: {len(DATA['X_test'])}")

    # ---------------- Data Loader ----------------
    batch = 32  # Batch size for training
    train_dataset = PAHDataset(DATA["X_train"], DATA["Y_train"], DATA["charges_train"])
    valid_dataset = PAHDataset(DATA["X_valid"], DATA["Y_valid"], DATA["charges_valid"])
    test_dataset = PAHDataset(DATA["X_test"], DATA["Y_test"], DATA["charges_test"])

    train_dataloader = DataLoader(train_dataset, batch_size=batch, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch)
    test_dataloader = DataLoader(test_dataset, batch_size=batch)

    # Initialize model with appropriate dimensions
    input_size, layers, output_size = DATA["X_train"].shape[-1], [1500, 1000, 800, 600], DATA["Y_train"].shape[-1]
    predictor = Predictor(input_size, layers, output_size)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=0.0001, betas=(0.9, 0.999), eps=1e-8)

    # Set device (GPU if available, otherwise CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device being used: {device}")
    predictor.to(device)

    # ---------------- Record Loss ----------------
    train_emd_losses = []  # Track training EMD loss per epoch
    valid_emd_losses = []  # Track validation EMD loss per epoch


    # ---------------- Train/Eval/Test Function ----------------
    def iteration(epo, data_loader, mode):
        """Single iteration through data loader for train/val/test.
        
        Args:
            epo: Current epoch number
            data_loader: PyTorch DataLoader
            mode: 'train', 'val', or 'test'
            
        Returns:
            all_preds_tensor: All predictions concatenated
            current_emd: Average EMD loss (0 for test mode)
        """
        emd_sum = 0.0
        all_preds = []

        for i, data in enumerate(data_loader):
            # Forward pass
            preds, emd = predictor(
                data["smi"].to(device),
                data["label"].to(device),
                data["charge"].tolist()
            )

            # Accumulate loss (skip for test set)
            if mode != "test":
                emd_sum += emd.item()
            
            # Backpropagation for training only
            if mode == "train":
                optimizer.zero_grad()
                emd.backward()
                optimizer.step()

            # Collect predictions for later analysis
            all_preds.append(preds)

        # Concatenate all batch predictions
        all_preds_tensor = torch.cat(all_preds, dim=0) if all_preds else torch.empty(0, device=device)

        # Calculate average EMD (not for test set)
        current_emd = emd_sum / (i + 1) if i >= 0 and mode != "test" else 0.0

        # Record losses for plotting
        if mode == "train":
            train_emd_losses.append(current_emd)
        elif mode == "val":
            valid_emd_losses.append(current_emd)

        # Print progress
        if mode == "train" or mode == "val":
            print(f"[Fold {fold}] | {mode.upper()} | Epoch: {epo} | EMD: {current_emd:.4f}")
        elif mode == "test":
            print(f"[Fold {fold}] | {mode.upper()} | Epoch: {epo} | Predictions Collected")

        return all_preds_tensor, current_emd


    # ---------------- Early Stopping ----------------
    early_stopping = EarlyStopping(patience=50, min_delta=0.001, verbose=True)

    # ---------------- Training Loop ----------------
    epochs = 999999  # Large number - training stops via early stopping
    for epoch in range(epochs):
        # Training phase
        predictor.train()
        _, train_emd = iteration(epoch, train_dataloader, "train")

        # Validation phase
        predictor.eval()
        with torch.no_grad():
            _, val_emd = iteration(epoch, valid_dataloader, "val")

        # Check early stopping condition
        if early_stopping(predictor, val_emd):
            # Load best model weights before breaking
            predictor.load_state_dict(early_stopping.best_weights)
            print(f"[Fold {fold}] Early stopping at epoch {epoch}")
            break

    # ---------------- Test and Save Results ----------------
    predictor.eval()
    with torch.no_grad():
        test_preds, _ = iteration(0, test_dataloader, "test")

    # Process test labels and predictions for evaluation
    test_labels = torch.tensor(DATA["Y_test"], dtype=torch.float).to(device)
    normed_preds, normed_labels = predictor.normalize(test_preds.to(device), test_labels)
    
    # Calculate EMD for each test sample
    diff = normed_preds - normed_labels
    cumsum_diff = torch.cumsum(diff, dim=1)
    emds = torch.sum(torch.abs(cumsum_diff), dim=1).cpu().detach().numpy()

    # Convert predictions and EMDs to dictionaries keyed by UID
    pred_dict = {int(uid): pred.tolist() for uid, pred in zip(DATA["UID_test"], test_preds.cpu().detach().numpy())}
    emd_dict = {int(uid): float(emd) for uid, emd in zip(DATA["UID_test"], emds)}

    # Save enhanced DATA with predictions
    DATA['Y_pred_test'] = pred_dict
    DATA['EMD_test'] = emd_dict
    with open(f"DATA_fold{fold}.pck", "wb") as f:
        pickle.dump(DATA, f)

    # Save trained model weights
    torch.save(predictor.state_dict(), f"model_fold{fold}.pth")

    # Plot EMD distribution for test set
    plt.figure(figsize=(6, 4), dpi=300)
    plt.hist(emds, bins=20, color='skyblue', edgecolor='black')
    plt.title(f"Fold {fold} | Mean EMD: {emds.mean():.4f}", fontsize=12)
    plt.xlabel("EMD")
    plt.ylabel("Frequency")
    plt.savefig(f"test_emd_fold{fold}.png", bbox_inches='tight', facecolor='white')
    plt.close()

    # Plot training/validation loss curves
    plt.figure(figsize=(8, 5), dpi=200)
    plt.plot(train_emd_losses, label='Train EMD Loss', color='blue')
    plt.plot(valid_emd_losses, label='Valid EMD Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold} Training Curve')
    plt.legend()
    plt.grid(False)
    ax = plt.gca()
    ax.set_facecolor('white')
    plt.gcf().set_facecolor('white')
    plt.savefig(f"loss_curve_fold{fold}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    # Save fold results for final summary
    all_fold_results[f'fold_{fold}'] = {
        'mean_emd': emds.mean(),
        'std_emd': emds.std(),
        'train_losses': train_emd_losses,
        'valid_losses': valid_emd_losses
    }

# ---------------- Final Summary ----------------
print("\n? 5-Fold Cross Validation Completed!")
print("=" * 50)
for k, v in all_fold_results.items():
    print(f"{k}: Mean EMD = {v['mean_emd']:.4f} ¡À {v['std_emd']:.4f}")

# Save all results to file
with open("all_fold_results.pck", "wb") as f:
    pickle.dump(all_fold_results, f)
    
print("\nResults saved to 'all_fold_results.pck'")
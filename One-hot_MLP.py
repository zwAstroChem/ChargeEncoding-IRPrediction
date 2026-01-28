import os
import warnings
import pickle
import json
import datetime

# Suppress TensorFlow INFO and WARNING messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=all, 1=INFO, 2=INFO+WARNING, 3=ERROR only
warnings.filterwarnings("ignore")  # Suppress Python warnings

# Import necessary libraries
try:
    import numpy as np
    from collections import defaultdict
    import pandas as pd
    from rdkit import Chem
    from rdkit.Chem import AllChem
    import matplotlib.pyplot as plt
    import tensorflow as tf
    import keras
    import keras.backend as K
    from keras.models import Sequential
    from keras.layers import Dense, Lambda
    from keras.optimizers import Adam
except ModuleNotFoundError:
    raise Exception("Please make sure rdkit, tensorflow and keras are installed!")

# Configure GPU settings for TensorFlow
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to avoid allocating all GPU memory at once
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def GetChargeOneHot(charge_val):
    """
    Convert charge value to One-Hot encoding vector.
    
    Args:
        charge_val: Integer or string representing molecular charge
        
    Returns:
        List of 4 elements representing One-Hot encoding:
        -1 -> [1, 0, 0, 0]  # Negative charge
        +1 -> [0, 1, 0, 0]  # Positive charge (+1)
         0 -> [0, 0, 1, 0]  # Neutral charge
        +2 -> [0, 0, 0, 1]  # Positive charge (+2)
        Other -> [0, 0, 0, 0]  # Unknown charge state (all zeros)
    """
    try:
        c = int(charge_val)  # Convert to integer
    except (ValueError, TypeError):
        c = 0  # Default to neutral if conversion fails
    
    # Map integer charge to One-Hot vector
    if c == -1:
        return [1, 0, 0, 0]
    elif c == 1:
        return [0, 1, 0, 0]
    elif c == 0:
        return [0, 0, 1, 0]
    elif c == 2:
        return [0, 0, 0, 1]
    else:
        return [0, 0, 0, 0]  # Default for unknown charges


def GenerateDATA(smiles, targets, molecule_ids, charges, train_indexes, val_indexes, test_indexes, radius=11, Nmin=4):
    """
    Generate Morgan fingerprints and create dataset for training.
    
    Args:
        smiles: Array of SMILES strings
        targets: Array of target spectra
        molecule_ids: Array of molecule identifiers
        charges: Array of One-Hot encoded charge vectors
        train_indexes: Indices for training set
        val_indexes: Indices for validation set
        test_indexes: Indices for test set
        radius: Radius for Morgan fingerprint generation
        Nmin: Minimum frequency for fingerprint feature selection
        
    Returns:
        Dictionary containing all processed data for training/validation/testing
    """
    allIDs = defaultdict(int)  # Counter for fingerprint bits
    morganfps = np.empty(len(smiles), dtype=object)  # Store fingerprints
    
    # Step 1: Generate Morgan fingerprints for all molecules
    for smiles_ind in range(len(smiles)):
        curr_smiles = smiles[smiles_ind]
        mol = Chem.MolFromSmiles(curr_smiles)
        fp = AllChem.GetMorganFingerprint(mol, radius)  # Generate fingerprint
        morganfps[smiles_ind] = fp
        
        # Count feature occurrences only in training set (for feature selection)
        if smiles_ind in train_indexes:
            for ID in [*fp.GetNonzeroElements()]:
                allIDs[ID] += 1

    print(f"Number of fragments before Nmin filtering: {len(allIDs)}")

    # Step 2: Filter features based on minimum occurrence (Nmin)
    IDlist = []
    for ID in allIDs.keys():
        if allIDs[ID] >= Nmin:
            IDlist.append(ID)
    N_fragments = len(IDlist)

    print(f"Number of fragments after Nmin filtering: {len(IDlist)}")

    # Step 3: Build feature matrix combining fingerprints and charge encodings
    # Matrix dimensions: [n_samples, (N_fragments + 4)]
    # First N_fragments columns: ECFP fingerprints
    # Last 4 columns: Charge One-Hot encoding
    Xs = np.zeros((len(smiles), N_fragments + 4))
    
    # ---------------------------------------------------------
    # Fill Fingerprint Features (Columns 0 to N_fragments-1)
    # ---------------------------------------------------------
    for i in range(len(smiles)):
        non_zero = morganfps[i].GetNonzeroElements()  # Get non-zero fingerprint bits
        for fragmentID in non_zero:
            if fragmentID in IDlist:
                location = IDlist.index(fragmentID)  # Find position in feature list
                Xs[i][location] = morganfps[i][fragmentID]  # Set fingerprint value

    # ---------------------------------------------------------
    # Fill Charge One-Hot Features (Last 4 columns)
    # ---------------------------------------------------------
    for i in range(len(smiles)):
        Xs[i][-4:] = charges[i]  # Assign One-Hot vector to last 4 columns

    # Package all data into result dictionary
    result = {
        "smiles_train": smiles[train_indexes],
        "smiles_val": smiles[val_indexes],
        "smiles_test": smiles[test_indexes],
        "ids_train": molecule_ids[train_indexes],
        "ids_val": molecule_ids[val_indexes],
        "ids_test": molecule_ids[test_indexes],
        "X_train": Xs[train_indexes],
        "X_val": Xs[val_indexes],
        "X_test": Xs[test_indexes],
        "Y_train": targets[train_indexes],
        "Y_val": targets[val_indexes],
        "Y_test": targets[test_indexes],
        "IDs": IDlist,
        "radius": radius,
        "Nmin": Nmin,
        "charge_train": charges[train_indexes],
        "charge_val": charges[val_indexes],
        "charge_test": charges[test_indexes],
        "molecule_ids": molecule_ids
    }
    return result


def EMDloss(Y1, Y2):
    """
    Earth Mover's Distance (EMD) loss function for comparing probability distributions.
    
    Args:
        Y1: Predicted spectra (batch_size ¡Á n_wavelengths)
        Y2: True spectra (batch_size ¡Á n_wavelengths)
        
    Returns:
        EMD loss value
    """
    # Normalize spectra to probability distributions (sum to 1)
    # Add small epsilon to prevent division by zero
    normed_Y1 = (Y1 / (K.sum(Y1, axis=1)[:, None] + 1e-9))
    normed_Y2 = (Y2 / (K.sum(Y2, axis=1)[:, None] + 1e-9))
    
    # Calculate difference between normalized distributions
    diff = normed_Y1 - normed_Y2
    
    # Compute EMD: sum of absolute cumulative differences
    return K.sum(K.abs(K.cumsum(diff, axis=1)))


def plot_training_loss(history, fold):
    """
    Plot training and validation loss curves.
    
    Args:
        history: Keras training history object
        fold: Current fold number (for title and filename)
    """
    train_losses = history.history['loss']
    val_losses = history.history['val_loss']
    
    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(range(len(train_losses)), train_losses, label='Train EMD Loss', color='blue')
    plt.plot(range(len(val_losses)), val_losses, label='Validation EMD Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title(f'Fold {fold} Training and Validation Loss')
    plt.legend()
    plt.grid(False)
    
    # Set background colors for clean visualization
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    plt.gcf().set_facecolor('white')
    
    plt.savefig(f"loss_curve_fold{fold}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_test_hist(emds, fold, mean_emd):
    """
    Plot histogram of EMD values for test set.
    
    Args:
        emds: Array of EMD values for test samples
        fold: Current fold number (for title)
        mean_emd: Mean EMD value (displayed in title)
    """
    plt.figure(figsize=(8, 6), dpi=300)
    plt.hist(emds, bins=50, alpha=0.75, edgecolor='black', color='skyblue')
    plt.title(f"Fold {fold} | Mean EMD: {mean_emd:.4f}")
    plt.xlabel("EMD")
    plt.ylabel("Count")
    plt.grid(False)
    
    # Set background colors
    ax = plt.gca()
    ax.set_facecolor('white')
    ax.spines['left'].set_color('gray')
    ax.spines['bottom'].set_color('gray')
    plt.gcf().set_facecolor('white')
    
    plt.savefig(f"test_EMD_hist_fold{fold}.png", dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()


# -------------------------------------------------------------------------
# Model Training Function
# -------------------------------------------------------------------------

def TrainModel(data, layers, lossfunc, fold):
    """
    Build and train a Keras neural network model.
    
    Args:
        data: Dictionary containing training/validation data
        layers: List of hidden layer sizes (e.g., [1500, 1000, 800, 650])
        lossfunc: Loss function to use (EMDloss)
        fold: Current fold number (for logging)
        
    Returns:
        Trained Keras model
    """
    print(f"\n>> Building model for Fold {fold}...")
    
    # Create sequential model
    model = Sequential()
    input_dim = data["X_train"].shape[1]  # Total input dimension (ECFP + charge)
    output_dim = data["Y_train"].shape[1]  # Output dimension (spectrum length)
    
    print(f"Input Dim: {input_dim}, Output Dim: {output_dim}")

    # Add input layer
    model.add(Dense(layers[0], input_dim=input_dim, activation='relu'))
    
    # Add hidden layers
    for i in range(len(layers) - 1):
        model.add(Dense(layers[i + 1], activation='relu'))
    
    # Add output layer (linear activation)
    model.add(Dense(output_dim, activation='linear'))
    
    # Ensure non-negative predictions (spectra cannot be negative)
    model.add(Lambda(lambda x: K.abs(x)))

    # Compile model with Adam optimizer
    opt = Adam(lr=0.0001)
    model.compile(loss=lossfunc, optimizer=opt)

    # Early stopping callback to prevent overfitting
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', 
        min_delta=0.005,  # Minimum change to qualify as improvement
        patience=50,      # Number of epochs to wait for improvement
        verbose=1, 
        mode='auto', 
        restore_best_weights=True  # Restore best weights when stopping
    )
    
    print(f"Starting training for Fold {fold}...")
    
    # Train model
    history = model.fit(
        data["X_train"], 
        data["Y_train"], 
        epochs=1000,      # Maximum epochs (early stopping will stop earlier)
        batch_size=32,
        callbacks=[early_stop], 
        validation_data=(data["X_val"], data["Y_val"]),
        verbose=2  # Show progress bar
    )

    # Plot training curves
    plot_training_loss(history, fold)
    return model


# -------------------------------------------------------------------------
# Main Program
# -------------------------------------------------------------------------

def main():
    # Configuration
    json_filename = "./torch_spectrum_12599.json" 
    
    print(f"Loading data from {json_filename}...")
    if not os.path.exists(json_filename):
        raise FileNotFoundError(f"Could not find {json_filename}")

    # Load JSON data
    with open(json_filename, "r") as f:
        data_dict = json.load(f)

    # Initialize data containers
    smiles_list = []
    targets_list = []
    charges_list = []
    id_list = []
    
    # Sort keys for deterministic processing
    sorted_ids = sorted(data_dict.keys())
    
    print("Parsing JSON data...")
    for uid in sorted_ids:
        item = data_dict[uid]
        
        # Extract data fields
        smi = item.get("smi")
        spectra = item.get("spectra")
        charge_str = item.get("charge")
        
        # Skip incomplete entries
        if smi is None or spectra is None:
            continue
            
        # Validate SMILES
        if Chem.MolFromSmiles(smi) is None:
            print(f"Invalid SMILES for ID {uid}, skipping.")
            continue
            
        # Append valid data
        smiles_list.append(smi)
        targets_list.append(spectra)
        charges_list.append(GetChargeOneHot(charge_str))  # Convert charge to One-Hot
        id_list.append(uid)

    # Convert to numpy arrays
    smiles = np.array(smiles_list)
    charges = np.array(charges_list)
    uids = np.array(id_list)

    # Apply spectral windowing: select wavelength range [0:147]
    # Note: This differs from Embedding_MLP.py which used [245:300]
    targets = np.array(targets_list)[:, :147]
    
    print(f"Total valid molecules: {len(uids)}")
    print(f"Spectra length (Target dim after slicing): {targets.shape[1]}")

    # Validate target dimensions
    if targets.shape[1] == 0:
        raise ValueError("Target dimension is 0! Please check if your spectra data length is > 147.")

    # ================== 5-Fold Cross Validation Setup ==================
    
    n_samples = len(smiles)
    if n_samples < 5:
        raise ValueError("Data too small for 5-fold CV (need at least 5 samples).")

    # Create indices and shuffle for randomness
    indices = np.arange(n_samples)
    np.random.seed(1)  # Fixed seed for reproducibility
    shuffled_indices = np.random.permutation(indices)
    
    # Split into 5 approximately equal folds
    fold_size = n_samples // 5
    remainder = n_samples % 5
    splits = []
    start = 0
    for i in range(5):
        size = fold_size + (1 if i < remainder else 0)
        splits.append(shuffled_indices[start:start + size])
        start += size

    # Store statistics for all folds
    all_fold_stats = {}

    # ================== Perform 5-Fold Cross Validation ==================
    
    for fold in range(1, 6):
        print("\n" + "="*50)
        print(f" PROCESSING FOLD {fold} / 5 ")
        print("="*50)

        # Define fold assignments using rotating scheme:
        # Test: current block (fold-1)
        # Validation: next block (fold % 5)
        # Training: remaining three blocks
        test_indexes = splits[fold - 1]
        val_indexes = splits[fold % 5]
        
        train_blocks = []
        for i in range(5):
            if i != (fold - 1) and i != (fold % 5):
                train_blocks.append(splits[i])
        train_indexes = np.concatenate(train_blocks)

        # Generate dataset for current fold
        DATA = GenerateDATA(
            smiles, targets, uids, charges,
            train_indexes, val_indexes, test_indexes, 
            radius=11, Nmin=4
        )

        # ---------------------------------------------------------
        # DEBUG: Inspect feature vectors to verify concatenation
        # ---------------------------------------------------------
        print(f"\n[DEBUG] Inspecting first 5 training samples in Fold {fold}:")
        limit = min(5, len(DATA["ids_train"]))
        for i in range(limit):
            uid = DATA["ids_train"][i]
            x_vector = DATA["X_train"][i]
            
            # First 5 elements are ECFP fingerprint bits
            ecfp_first_5 = x_vector[:5]
            
            # Last 4 elements are Charge One-Hot encoding
            charge_one_hot = x_vector[-4:]
            
            print(f"Sample {i} (ID: {uid}):")
            print(f"  -> ECFP First 5 bits: {ecfp_first_5}")
            print(f"  -> Charge One-Hot (Last 4): {charge_one_hot}")
        print("="*50)
        # ---------------------------------------------------------

        # Train model for current fold
        model = TrainModel(DATA, [1500, 1000, 800, 650], EMDloss, fold)
        
        # Save model
        model.save(f"MODEL_fold{fold}.h5")

        # ================== Evaluate Model ==================
        print(f"Evaluating Fold {fold}...")
        
        # Make predictions on test set
        preds = model.predict(DATA["X_test"], batch_size=32)
        
        # Normalize predictions and true values for EMD calculation
        preds_norm = preds / (np.sum(preds, axis=1)[:, np.newaxis] + 1e-9)
        Y_test_norm = DATA["Y_test"] / (np.sum(DATA["Y_test"], axis=1)[:, np.newaxis] + 1e-9)
        
        # Calculate EMD for each test sample
        diff = preds_norm - Y_test_norm
        emds = np.sum(np.abs(np.cumsum(diff, axis=1)), axis=1)
        mean_emd = np.mean(emds)
        std_emd = np.std(emds)

        print(f"Fold {fold} Result -> Mean EMD: {mean_emd:.4f}, Std: {std_emd:.4f}")

        # Save predictions and EMD values to DATA dictionary
        DATA["Y_pred_test"] = preds
        DATA["EMD_test"] = emds
        pickle.dump(DATA, open(f"DATA_fold{fold}.pck", "wb"))

        # Visualize results
        plot_test_hist(emds, fold, mean_emd)

        # Store fold statistics
        all_fold_stats[f"Fold_{fold}"] = {"mean": mean_emd, "std": std_emd}
        
        # Clear Keras session and free memory
        K.clear_session()
        del model
        del DATA

    # ================== Final Summary ==================
    print("\n" + "="*50)
    print(" 5-FOLD CROSS VALIDATION COMPLETED ")
    print("="*50)
    
    # Calculate average performance across folds
    avg_mean_emd = np.mean([v["mean"] for v in all_fold_stats.values()])
    print(f"Average Mean EMD across 5 folds: {avg_mean_emd:.4f}")
    
    # Print results for each fold
    for k, v in all_fold_stats.items():
        print(f"{k}: {v['mean']:.4f} (+/- {v['std']:.4f})")


if __name__ == "__main__":
    main()
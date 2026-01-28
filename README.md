# PAH infrared Spectrum Predictor (Embedding-MLP)

## Description
This repository contains a machine learning model for predicting infrared (IR) spectra of Polycyclic Aromatic Hydrocarbons (PAHs) across four charge states: -1, 0, +1, and +2. The model uses Extended-Connectivity Fingerprints (ECFP) for molecular structure representation and a learnable embedding layer to encode charge information. It is trained on a dataset of 12,599 PAHs using a multi-layer perceptron (MLP) optimized with Earth Mover's Distance (EMD) loss.

## Features
- Predicts normalized IR spectra for PAHs of varying sizes (up to ~150 carbon atoms)
- Incorporates charge state information via learnable embeddings or one-hot encoding (within two different python files, Embedding_MLP.py and One-hot_MLP.py)
- Provides fast inference compared to DFT calculations
- Includes 5-fold cross-validation scripts and evaluation tools

## Requirements
- Python 3.8+
- PyTorch 2.0+
- RDKit
- NumPy
- Matplotlib
- scikit-learn
- CUDA-capable GPU (recommended for training)

## Install dependencies 
via:
pip install torch rdkit-pypi numpy matplotlib scikit-learn

## Usage
1. Prepare Data
Place your JSON data file in the root directory. The file should contain SMILES strings, spectra, and charge information in the expected format.

2. Run Training and Evaluation
Execute the main script to perform 5-fold cross-validation:
python Embedding_MLP.py

The script will:
a. Preprocess molecular fingerprints
b. Train five separate models (one per fold)
c. Save each model as model_fold{fold}.pth
d. Save processed data as DATA_fold{fold}.pck
e. Generate loss curves (loss_curve_fold{fold}.png) and EMD histograms (test_emd_fold{fold}.png)
f. Output mean EMD results for each fold

3. Load a Pretrained Model
import torch
from Embedding_MLP import Predictor
model = Predictor(input_dim=num_features, layers_dim=[1500, 1000, 800, 600], output_dim=55)
model.load_state_dict(torch.load("model_fold1.pth"))
model.eval()

4. Predict on New Molecules
# Prepare input (ECFP fingerprint + charge)
with torch.no_grad():
    prediction, _ = model(fp_tensor, label_tensor, charge_list)
Output Files
DATA_fold{fold}.pck: Processed data for each fold
model_fold{fold}.pth: Trained model weights
loss_curve_fold{fold}.png: Training/validation loss curves
test_emd_fold{fold}.png: Distribution of EMD errors on test set
all_fold_results.pck: Summary of all fold results

## Limitations
Molecular Size: Model performance may degrade for PAHs with >150 carbon atoms due to limited training data in that size range.
Charge States: Only supports charge states -1, 0, +1, +2. Tri-anionic or other exotic charge states are not supported.
Fingerprint Limitations: Uses ECFP fingerprints with radius=11. Unusual structural motifs not represented in the training data may be poorly predicted.
Normalized Output: Predicts normalized spectra only, not absolute intensities.
Computational Dependency: Requires RDKit for fingerprint generation, which may not handle all SMILES formats or stereochemistry perfectly.
Hardware: Training is GPU-intensive; inference is faster but still requires moderate computational resources.

## Citation
If you use this code in your research, please cite the associated paper:
He, J., et al. (2026). "Charge-aware machine learning prediction of PAH infrared spectra."

## License
This project is provided under the MIT License. 

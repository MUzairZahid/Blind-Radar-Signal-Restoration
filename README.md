# BRSR-OpGAN: Blind Radar Signal Restoration using Operational GAN

This repository contains the official implementation of BRSR-OpGAN (Blind Radar Signal Restoration using Operational Generative Adversarial Network), a novel approach for restoring radar signals corrupted by various artifacts without making prior assumptions about the corruption type or severity.

## Project Overview

BRSR-OpGAN ([paper link](https://arxiv.org/abs/2407.13949)) is a novel approach for restoring radar signals corrupted by various artifacts without making prior assumptions about the corruption type or severity.

BRSR-OpGAN can restore radar signals corrupted by:
- Additive White Gaussian Noise (AWGN)
- Echo
- Co-Channel Interference (CCI)
- Any blend of these artifacts

This makes it ideal for real-world radar applications where corruption types and severity are often unknown in advance.

For more details, please see our papers:
- [BRSR-OpGAN: Blind Radar Signal Restoration using Operational Generative Adversarial Network](https://arxiv.org/abs/2407.13949)
- [CoRe-Net: Co-Operational Regressor Network with Progressive Transfer Learning for Blind Radar Signal Restoration](https://arxiv.org/abs/2501.17125)

## Repository Structure

```
BRSR-OpGAN/
├── data_generation/          # Data generation and preparation scripts
│   ├── README.md             # Documentation for data generation
│   ├── DataGeneration_*.m    # MATLAB scripts for generating datasets
│   ├── DataPreparation_*.py  # Python scripts for processing datasets
│   └── ...
├── models.py                 # Model implementations
├── models_SelfONN.py         # Simple U-Net architecture with SelfONN
├── models_SelfONN_residual.py  # Residual U-Net architecture with SelfONN
├── utils.py                  # Utility functions for training and evaluation
├── train.py                  # Training script with various configuration options
├── test_BRSR-OpGAN.py        # Testing script for evaluating multiple models
└── ...
```

## Setup

### Requirements

#### MATLAB (for data generation)
- MATLAB R2019b or later
- Signal Processing Toolbox
- Time-Frequency Toolbox (tftb-0.2)

#### Python (for training and testing)
- Python 3.7+
- PyTorch 1.7+
- FastONN (SelfONN implementation)
- NumPy
- SciPy
- Matplotlib
- h5py
- scikit-learn

### Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/BRSR-OpGAN.git
cd BRSR-OpGAN
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Install FastONN (for SelfONN layer implementation):
```bash
pip install fastonn
```

## Dataset Generation and Preparation

The dataset consists of radar signals with 12 different waveform types corrupted by various artifacts. For detailed instructions on data generation, please refer to the [data generation README](data_generation/README.md).

### Quick Data Generation Steps

1. Generate the dataset using MATLAB:
```matlab
cd data_generation
addpath 'waveform-types'
run DataGeneration_Baseline.m   % For AWGN-only dataset
run DataGeneration_Extended.m    % For mixed artifacts dataset
```

2. Process the dataset using Python:
```bash
cd data_generation
python DataPreparation_Baseline.py
python DataPreparation_Extended.py
```

**Note:** The complete datasets and pre-trained models are available upon request. Please contact the authors.

## Training

Use `train.py` to train the BRSR-OpGAN model with various configuration options:

```bash
python train.py --model residual --Q 3 --mode train --epochs 1000 --lambda_recon 100 --lambda_freq 2 --batch_size 64 --device cuda --dataset extended
```

### Command-line Arguments

- `--model`: Choose model architecture (`simple` or `residual`)
- `--Q`: SelfONN parameter (set to 1 for conventional CNN)
- `--mode`: Operation mode (`train`, `evaluate`, or `both`)
- `--epochs`: Number of training epochs
- `--lambda_recon`: Reconstruction loss weight
- `--lambda_freq`: Frequency domain loss weight (set to 0 to disable)
- `--batch_size`: Training batch size
- `--device`: Device to use (`cuda` or `cpu`)
- `--out_folder`: Output folder for saved weights and logs
- `--data_folder`: Folder containing the dataset
- `--dataset`: Dataset type (`base` or `extended`)
- `--normalize`: Enable data normalization

## Testing

Use `test_BRSR-OpGAN.py` to evaluate trained models:

```bash
python test_BRSR-OpGAN.py --data_folder Prepared_Dataset --dataset extended --batch_size 32 --output_dir results/
```

This script can evaluate multiple models simultaneously for comparison. To test a single model, you can modify the `model_configs_info` dictionary in the script to include only your model of interest.

### Modifying for a Single Model

For testing a single model, modify the `model_configs_info` dictionary:

```python
model_configs_info = {
    'BRSR-OpGAN (Q=3)': {
        'paths': ['path/to/your/model.pth'],
        'q': 3
    }
}
```

## Citation

If you use this code or find it helpful for your research, please cite our papers:

```bibtex
@article{zahid2024brsr,
  title={BRSR-OpGAN: Blind Radar Signal Restoration using Operational Generative Adversarial Network},
  author={Zahid, Muhammad Uzair and Kiranyaz, Serkan and Yildirim, Alper and Gabbouj, Moncef},
  journal={arXiv preprint arXiv:2407.13949},
  year={2024}
}
```

```bibtex
@article{zahid2025core,
  title={CoRe-Net: Co-Operational Regressor Network with Progressive Transfer Learning for Blind Radar Signal Restoration},
  author={Zahid, Muhammad Uzair and Kiranyaz, Serkan and Yildirim, Alper and Gabbouj, Moncef},
  journal={arXiv preprint arXiv:2501.17125},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.


## Contact

For any questions or to request datasets and pre-trained models, please contact:
- Muhammad Uzair Zahid - [muhammaduzair.zahid@tuni.fi]

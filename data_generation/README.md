# Radar Signal Dataset Generation

This directory contains MATLAB scripts for generating baseline and extended radar signal datasets used in BRSR-OpGAN research.

## Files

### Generation Scripts
- **DataGeneration_Baseline.m** - Generates signals corrupted with AWGN at fixed SNR levels
- **DataGeneration_Extended.m** - Generates signals with randomized blends of multiple artifacts
- **add_distortion.m** - Helper function that applies AWGN, echo, and interference artifacts

### Visualization Tools
- **plot_signals_awgn.m** - Visualizes baseline signals (clean vs. AWGN-corrupted)
- **plot_signals_blind.m** - Visualizes extended dataset signals with multiple artifact components

### Required Data
- **mixing_signals.mat** - Contains signal patterns used for interference simulation
- **waveform-types/** - Directory containing radar waveform generation functions

## Generated Datasets

### Baseline Dataset
- **Corruption type:** AWGN only (Additive White Gaussian Noise)
- **SNR levels:** -14 dB to 10 dB (2 dB steps)
- **Training set:** 400 samples × 12 waveforms × 13 SNR levels = 62,400 signals
- **Test set:** 150 samples × 12 waveforms × 13 SNR levels = 23,400 signals

### Extended BRSR Dataset
- **Corruption types:** Random combination of AWGN, Echo, and Interference
- **SNR levels:** Random values between -14 dB and 10 dB
- **Training set:** 400 samples × 12 waveforms × 13 SNR levels = 62,400 signals
- **Test set:** 150 samples × 12 waveforms × 13 SNR levels = 23,400 signals

## Radar Waveform Types (12)

| Type  | Description |
|-------|-------------|
| LFM   | Linear Frequency Modulation |
| Costas | Frequency hopping |
| BPSK  | Binary Phase Shift Keying (Barker codes) |
| Frank | Polyphase code |
| P1-P4 | Polyphase codes |
| T1-T4 | Frequency shift keying variants |

## Output Format

Each dataset contains the following MATLAB data files:

### Baseline Dataset

```
Dataset_baseline_train/
├── X_clean_signals_train.mat  - Original radar signals [N×2×1024]
├── X_distorted_signals_train.mat - AWGN-corrupted signals [N×2×1024]
├── Y_label_train.mat - Waveform class labels [N×1]
└── Y_SNR_train.mat - SNR values [N×1]

Dataset_baseline_test/
├── X_clean_signals_test.mat
├── X_distorted_signals_test.mat
├── Y_label_test.mat
└── Y_SNR_test.mat
```

### Extended Dataset

```
Dataset_extended_train/
├── X_clean_signals_train.mat - Clean signals [N×2×1024]
├── X_distorted_signals_train.mat - Combined corrupted signals [N×2×1024]
├── X_distortion_awgn_train.mat - AWGN component [N×2×1024]
├── X_distortion_echo_train.mat - Echo component [N×2×1024]
├── X_distortion_cci_train.mat - Interference component [N×2×1024]
├── Y_label_train.mat - Waveform class labels [N×1]
└── Y_SNR_train.mat - SNR values [N×1]

Dataset_extended_test/
└── [Same structure as train]
```

## Usage

1. **Setup**
   ```matlab
   addpath 'waveform-types'
   ```

2. **Generate baseline dataset**
   ```matlab
   run DataGeneration_Baseline.m
   ```

3. **Generate extended dataset**
   ```matlab
   run DataGeneration_Extended.m
   ```

## Requirements

- MATLAB R2019b or later
- Signal Processing Toolbox
- Time-Frequency Toolbox (tftb-0.2)

## Citation

If you use these datasets in your research, please cite our papers:

```bibtex
@article{zahid2024brsr,
  title={Brsr-opgan: Blind radar signal restoration using operational generative adversarial network},
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

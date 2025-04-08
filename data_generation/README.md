I'll update the README.md file to include information about the plotting functions and create a downloadable file section. Here's the revised README.md for your data generation folder:

```markdown
# Radar Signal Dataset Generation

This directory contains MATLAB scripts for generating radar signal datasets used in the BRSR-OpGAN research. The code generates paired clean/corrupted radar signals for training and evaluating blind radar signal restoration models.

## Contents

### Main Generation Scripts
- `DataGeneration_Baseline.m`: Generates radar signals corrupted with AWGN only
- `DataGeneration_Extended.m`: Generates radar signals corrupted with multiple artifacts
- `add_distortion.m`: Helper function for adding different types of artifacts

### Visualization Functions
- `plot_signals_awgn.m`: Visualization function for AWGN-corrupted signals (baseline dataset)
- `plot_signals_blind.m`: Visualization function for multi-artifact signals (extended dataset)

### Required Data
- `mixing_signals.mat`: Contains interference signals for the extended dataset generation

## Prerequisites

- MATLAB R2019b or later
- Signal Processing Toolbox
- Time-Frequency Toolbox (tftb-0.2)
- A folder containing waveform generation functions in `waveform-types/`

## Generated Datasets

### Baseline Dataset
- Contains signals corrupted only with AWGN at fixed SNR levels
- SNR range: -14 dB to 10 dB in 2 dB steps
- 400 samples per waveform per SNR level (training)
- 150 samples per waveform per SNR level (testing)
- Output saved to `Dataset_baseline_train/` and `Dataset_baseline_test/`

### Extended BRSR Dataset
- Contains signals corrupted with a randomized blend of artifacts:
  - AWGN (Additive White Gaussian Noise)
  - Echo (Delayed version of the signal)
  - CCI (Co-Channel Interference)
- Random SNR values between -14 dB and 10 dB
- 400 samples per waveform type per SNR level (training)
- 150 samples per waveform type per SNR level (testing)
- Output saved to `Dataset_extended_train/` and `Dataset_extended_test/`

## Radar Waveform Types

Both datasets include 12 types of radar waveforms:
1. LFM (Linear Frequency Modulation)
2. Costas
3. BPSK (Barker codes)
4. Frank
5. P1
6. P2
7. P3
8. P4
9. T1
10. T2
11. T3
12. T4

## Data Format

Each signal consists of complex-valued samples stored as a 2×1024 array (real and imaginary parts).

The generated datasets include the following variables:
- `X_clean_signals_*`: Original clean radar signals
- `X_distorted_signals_*`: Corrupted radar signals (with all artifacts)
- `X_distortion_awgn_*`: AWGN component (Extended dataset only)
- `X_distortion_echo_*`: Echo component (Extended dataset only)
- `X_distortion_cci_*`: Interference component (Extended dataset only)
- `Y_label_*`: Waveform class labels (1-12)
- `Y_SNR_*`: SNR values in dB

## Usage

1. Download the required data file:
   - [mixing_signals.mat](https://drive.google.com/uc?export=download&id=YOUR_FILE_ID) - Replace with actual download link

2. Make sure the required dependencies are available in your MATLAB path:
   ```matlab
   addpath 'waveform-types'
   ```

3. Generate the baseline dataset:
   ```matlab
   run DataGeneration_Baseline.m
   ```

4. Generate the extended dataset:
   ```matlab
   run DataGeneration_Extended.m
   ```

5. To visualize signals (optional):
   ```matlab
   % For baseline dataset (AWGN only)
   plot_signals_awgn(clean_signal, noisy_signal, SNR_value);
   
   % For extended dataset (multiple artifacts)
   plot_signals_blind(clean_signal, distorted_signal, awgn_component, echo_component, cci_component, SNR_value);
   ```

## Parameters

Key parameters that can be modified:
- `fps_train` and `fps_test`: Number of signals per SNR per waveform type
- `N`: Signal length (1024 samples by default)
- `SNR`: SNR range, default is -14:2:10 dB
- Waveform-specific parameters in each case section

## Generated Data Structure

```
Dataset_baseline_train/
├── X_clean_signals_train.mat
├── X_distorted_signals_train.mat
├── Y_label_train.mat
└── Y_SNR_train.mat

Dataset_baseline_test/
├── X_clean_signals_test.mat
├── X_distorted_signals_test.mat
├── Y_label_test.mat
└── Y_SNR_test.mat

Dataset_extended_train/
├── X_clean_signals_train.mat
├── X_distorted_signals_train.mat
├── X_distortion_awgn_train.mat
├── X_distortion_echo_train.mat
├── X_distortion_cci_train.mat
├── Y_label_train.mat
└── Y_SNR_train.mat

Dataset_extended_test/
├── X_clean_signals_test.mat
├── X_distorted_signals_test.mat
├── X_distortion_awgn_test.mat
├── X_distortion_echo_test.mat
├── X_distortion_cci_test.mat
├── Y_label_test.mat
└── Y_SNR_test.mat
```

## Citation

If you use these datasets in your research, please cite:

```bibtex
@article{zahid2024brsr,
  title={BRSR-OpGAN: Blind Radar Signal Restoration using Operational Generative Adversarial Network},
  author={Zahid, Muhammad Uzair and Kiranyaz, Serkan and Yildirim, Alper and Gabbouj, Moncef},
  journal={},
  year={2024}
}
```
```

For the downloadable file information, you'll need to host the `mixing_signals.mat` file somewhere and then update the link in the README. Options include:

1. Google Drive
2. Dropbox
3. OneDrive
4. A university file server
5. Zenodo (for research data)

Once you've uploaded the file to one of these services, you can replace the placeholder link in the README with the actual download link.

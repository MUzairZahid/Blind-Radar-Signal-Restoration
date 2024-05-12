# BRSR-OpGAN-Blind-Radar-Signal-Restoration-using-Operational-Generative-Adversarial-Network

This repository contains the code and data associated with the paper [Title of Your Paper]. It includes implementations for generating baseline data as well as an extended data generation framework and methodologies for signal restoration using Operational Generative Adversarial Networks (OpGANs).

## Repository Contents
- `baseline_data_generator/`: Code adapted for generating baseline dataset.
- `extended_data_generator/`: Our own framework for generating more complex, real-world-like radar signal datasets.
- `signal_restoration/`: Implementations of BRSR-OpGAN for signal restoration on the generated datasets.

## References and Acknowledgments
Portions of the baseline dataset generation code are adapted from the following research paper:

- Jiang, Mengting, et al. "Multilayer Decomposition Denoising Empowered CNN for Radar Signal Modulation Recognition." IEEE Access (2024). [GitHub Link](https://github.com/stu-cjlu-sp/rsrc-for-pub/tree/main/VMD-LMD-WT)

We thank the authors for their contributions to the community and acknowledge their pioneering work, which served as a foundation for the initial stages of our dataset preparation process.

## Our Contributions
This work extends the concepts from the referenced paper by introducing:
1. **Extended Data Generation Framework**: A novel approach to generate datasets that more accurately reflect the complexity and variability of real-world radar signals.
2. **Radar Signal Restoration**: Development and implementation of a new Operational Generative Adversarial Network model designed specifically for the nuanced demands of radar signal denoising and restoration.

The extended dataset and our BRSR-OpGAN model are designed to tackle more complex noise and interference conditions, moving beyond the limitations of traditional methods.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation
If you find our work useful in your research, please consider citing:

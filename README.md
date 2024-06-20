# BRSR-OpGAN-Blind-Radar-Signal-Restoration-using-Operational-Generative-Adversarial-Network

This repository contains the code and data associated with the paper. It includes implementations for generating baseline data as well as an extended data generation framework and methodologies for signal restoration using Operational Generative Adversarial Networks (OpGANs).

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


Creating a well-documented README file for your GitHub repository is crucial for making your project accessible and understandable to other developers or users. Below is a template for a README file for your CycleGAN model project, which you can further customize based on the specifics of your repository:

### README.md Template

```markdown
# CycleGAN for Audio Denoising

This repository contains the implementation of a CycleGAN model used for audio denoising, with support for both simple and residual architectures. The model is designed to handle datasets with different types of noise and is built with SelfONN layers that can be configured for complexity using the parameter `q`.

## Features

- Support for both simple and residual model architectures.
- Customizable SelfONN layer complexity through the `q` parameter.
- Configurable for training and evaluation on different datasets.
- Implementation in PyTorch with support for CUDA-enabled devices for efficient training.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.6+
- PyTorch 1.7+
- NumPy
- An environment that supports CUDA (if using GPU acceleration)

## Installation

Clone the repository to your local machine:

```bash
git clone https://github.com/yourusername/cyclegan-audio-denoising.git
cd cyclegan-audio-denoising
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Usage

Here's how to run the model on your datasets:

### Training

To train the model, use the following command:

```bash
python main.py --model simple --Q 3 --mode train --epochs 1000 --batch_size 64 --device cuda --data_folder ./data --dataset base
```

### Evaluation

To evaluate a trained model:

```bash
python main.py --model simple --Q 3 --mode evaluate --batch_size 64 --device cuda --data_folder ./data --dataset base
```

Replace `simple` with `residual` to use the residual architecture, and adjust other parameters as needed.

## Configuration

The script `main.py` supports several command-line arguments to adjust the model's training and evaluation:

- `--model`: Choose `simple` or `residual` architecture.
- `--Q`: Set the complexity level of SelfONN layers.
- `--mode`: Select `train`, `evaluate`, or `both`.
- `--epochs`: Specify the number of training epochs.
- `--batch_size`: Define the batch size for training.
- `--device`: Specify the computing device (`cuda` or `cpu`).
- Additional parameters can be viewed using `python main.py -h`.

## Contributing

Contributions to the project are welcome! To contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/AmazingFeature`).
3. Make your changes.
4. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
5. Push to the branch (`git push origin feature/AmazingFeature`).
6. Open a pull request.

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Your Name – [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/yourusername/cyclegan-audio-denoising](https://github.com/yourusername/cyclegan-audio-denoising)

## Acknowledgments

- Thanks to all contributors who have helped in shaping this project.
- Special thanks to XYZ for providing the datasets.
```

### Explanation of Sections

1. **Features**: Highlight the unique aspects of your project.
2. **Prerequisites**: List any requirements needed to run the project.
3. **Installation**: Step-by-step instructions to get the project set up locally.
4. **Usage**: Instructions on how to run the project.
5. **Configuration**: Details on how to configure the software or hardware environment.
6. **Contributing**: Guide for potential contributors.
7. **License**: Information about the project's license.
8. **Contact**: Your contact information or additional resources.
9. **Acknowledgments**: Credits to those who contributed to the project.

This README template is structured to provide clear and comprehensive information about your project, enhancing its visibility and usability. Adjust the content as necessary to match the specifics of your project and personal or organizational style.

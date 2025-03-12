import argparse
import os
import pickle
from collections import defaultdict
import numpy as np
import torch
from utils import *
from models import *
from dataset import *

def load_and_prepare_dataset(batch_size, dataset_path):
    """
    Load the dataset and prepare it for model evaluation
    """
    # Load the dataset
    with open(dataset_path, 'rb') as file:
        dataset = pickle.load(file)

    dataloaders = {}
    for split in ['train', 'validation', 'test']:
        # Check if 'distortions' key is present
        distortions = dataset[split].get('distortions', None)

        radar_dataset = RadarSignalDataset(
            dataset[split]['clean'],
            dataset[split]['noisy'],
            dataset[split]['label'],
            dataset[split]['SNR'],
            distortions,
            normalize='True'
        )
        
        dataloaders[split] = DataLoader(
            radar_dataset, 
            batch_size=batch_size,
            shuffle=False
        )

    return dataloaders

def calculate_snr(restored, clean):
    """Calculate Signal-to-Noise Ratio using PyTorch functions"""
    # Calculate signal power
    signal_power = torch.mean(clean**2, dim=[1, 2])
    
    # Calculate noise power
    noise_power = torch.mean((restored - clean)**2, dim=[1, 2])
    
    # Calculate SNR
    snr = 10 * torch.log10(signal_power / noise_power)
    return snr

def calculate_psnr(restored, clean):
    """Calculate Peak Signal-to-Noise Ratio for denormalized signals"""
    # Determine the actual max value from the clean signal (across both I/Q)
    max_value = torch.max(torch.abs(clean))
    
    # Calculate MSE across all dimensions
    mse = torch.mean((restored - clean)**2, dim=[1, 2])
    
    # Use the actual max value for PSNR calculation
    psnr = 10 * torch.log10(max_value**2 / mse)
    return psnr

def calculate_mse(restored, clean):
    """Calculate Mean Squared Error using PyTorch's F.mse_loss"""
    import torch.nn.functional as F
    
    # Calculate MSE across all dimensions (both I/Q and time)
    mse = F.mse_loss(restored, clean, reduction='none').mean(dim=[1, 2])
    return mse

def calculate_metrics(dataloader, model_configs, device):
    """
    Calculate SNR, PSNR and MSE for each model configuration and corrupted signals
    
    Returns:
    Dictionary of metrics for each model configuration and corrupted signals
    """
    results = {}
    
    # Add an entry for corrupted signals
    results["Corrupted Signal"] = defaultdict(list)
    
    # Initialize metrics for each model configuration
    for config_name in model_configs.keys():
        results[config_name] = defaultdict(list)
    
    # Set all models to evaluation mode
    for config_name, models in model_configs.items():
        for model in models:
            model.eval()
    
    with torch.no_grad():
        for data in dataloader['test']:
            clean, noisy, labels, snr_distorted, _ = data
            clean_signal, clean_min, clean_max = [x.to(device) for x in clean]
            distorted_signal, distorted_min, distorted_max = [x.to(device) for x in noisy]
            
            # Denormalize signals for accurate metrics
            clean_denorm = denormalize_signal(clean_signal, clean_min, clean_max)
            distorted_denorm = denormalize_signal(distorted_signal, distorted_min, distorted_max)
            
            # Calculate metrics for corrupted signals
            corrupted_snr = calculate_snr(distorted_denorm, clean_denorm).cpu().numpy()
            corrupted_psnr = calculate_psnr(distorted_denorm, clean_denorm).cpu().numpy()
            corrupted_mse = calculate_mse(distorted_denorm, clean_denorm).cpu().numpy()
            
            for idx, label in enumerate(labels):
                label_val = int(label.item())
                results["Corrupted Signal"]["snr"].append(corrupted_snr[idx])
                results["Corrupted Signal"]["psnr"].append(corrupted_psnr[idx])
                results["Corrupted Signal"]["mse"].append(corrupted_mse[idx])
                results["Corrupted Signal"]["label"].append(label_val)
            
            # Process through each model configuration
            for config_name, models in model_configs.items():
                restored = distorted_signal
                for model in models:
                    restored = model(restored)
                
                # Denormalize restored signal
                restored_denorm = denormalize_signal(restored, clean_min, clean_max)
                
                # Calculate metrics
                restored_snr = calculate_snr(restored_denorm, clean_denorm).cpu().numpy()
                restored_psnr = calculate_psnr(restored_denorm, clean_denorm).cpu().numpy()
                restored_mse = calculate_mse(restored_denorm, clean_denorm).cpu().numpy()
                
                for idx, label in enumerate(labels):
                    label_val = int(label.item())
                    results[config_name]["snr"].append(restored_snr[idx])
                    results[config_name]["psnr"].append(restored_psnr[idx])
                    results[config_name]["mse"].append(restored_mse[idx])
                    results[config_name]["label"].append(label_val)
    
    # Calculate averages
    avg_results = {}
    for config_name, metrics in results.items():
        avg_results[config_name] = {
            "avg_snr": np.mean(metrics["snr"]),
            "avg_psnr": np.mean(metrics["psnr"]),
            "avg_mse": np.mean(metrics["mse"]),
            # Calculate per-label averages
            "per_label_snr": {}, 
            "per_label_psnr": {},
            "per_label_mse": {}
        }
        
        # Group by label
        label_metrics = defaultdict(lambda: defaultdict(list))
        for i, label in enumerate(metrics["label"]):
            label_metrics[label]["snr"].append(metrics["snr"][i])
            label_metrics[label]["psnr"].append(metrics["psnr"][i])
            label_metrics[label]["mse"].append(metrics["mse"][i])
        
        # Calculate per-label averages
        for label, label_data in label_metrics.items():
            avg_results[config_name]["per_label_snr"][label] = np.mean(label_data["snr"])
            avg_results[config_name]["per_label_psnr"][label] = np.mean(label_data["psnr"])
            avg_results[config_name]["per_label_mse"][label] = np.mean(label_data["mse"])
    
    return avg_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, default='Prepared_Dataset')
    parser.add_argument('--dataset', type=str, default='extended')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--output_dir', type=str, default='visualization_models/Metrics_Analysis')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the dataset
    dataset_path = os.path.join(args.data_folder, f"{args.dataset}_dataset_k0.pickle")
    dataloaders = load_and_prepare_dataset(batch_size=args.batch_size, dataset_path=dataset_path)
    
    # Define model configurations with paths to model weights and their Q values
    model_configs_info = {
        'CNN-GAN (Time Domain)': {
            'paths': ['visualization_models/CNN_GAN_Time_Domain.pth'],
            'q': 1
        },
        'CNN-GAN (Dual Domain)': {
            'paths': ['visualization_models/CNN_GAN_Dual_Domain.pth'],
            'q': 1
        },
        'BRSR-OpGAN (Q=3, Time Domain)': {
            'paths': ['visualization_models/BRSR_OpGAN_Q3_Time_Domain.pth'],
            'q': 3
        },
        'BRSR-OpGAN (Q=3, Dual Domain)': {
            'paths': ['visualization_models/BRSR_OpGAN_Q3_Dual_Domain.pth'],
            'q': 3
        },
        'BRSR-OpGAN (Q=3, Dual Domain, 2nd Pass)': {
            'paths': [
                'visualization_models/BRSR_OpGAN_Q3_Dual_Domain.pth',
                'visualization_models/BRSR_OpGAN_Q3_Dual_Domain_2ndPass.pth'
            ],
            'q': 3
        }
    }
    
    # Initialize model configurations
    model_configs = {}
    
    # Create and load models with appropriate Q values
    for config_name, config_info in model_configs_info.items():
        models = []
        paths = config_info['paths']
        q_value = config_info['q']
        
        for path in paths:
            model = ResidualGenerator(q=q_value).to(device)
            try:
                # Handle different save formats
                checkpoint = torch.load(path, map_location=device)
                if isinstance(checkpoint, dict) and 'A_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['A_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                models.append(model)
                print(f"Successfully loaded model from {path} with Q={q_value}")
            except Exception as e:
                print(f"Error loading model from {path}: {e}")
        model_configs[config_name] = models
    
    # Calculate metrics
    results = calculate_metrics(dataloaders, model_configs, device)
    
    # Print results in a table format
    print("\n" + "="*80)
    print("{:<35} | {:<15} | {:<15} | {:<15}".format(
        "Model", "Avg SNR (dB)", "Avg PSNR (dB)", "Avg MSE"))
    print("-"*80)
    
    # Print corrupted signal metrics first
    print("{:<35} | {:<15.2f} | {:<15.2f} | {:<15.4f}".format(
        "Corrupted Signal (Reference)",
        results["Corrupted Signal"]["avg_snr"],
        results["Corrupted Signal"]["avg_psnr"],
        results["Corrupted Signal"]["avg_mse"]))
    
    # Print metrics for each model configuration
    for config_name in model_configs_info.keys():
        print("{:<35} | {:<15.2f} | {:<15.2f} | {:<15.4f}".format(
            config_name,
            results[config_name]["avg_snr"],
            results[config_name]["avg_psnr"],
            results[config_name]["avg_mse"]))
    print("="*80)
    
    # Generate LaTeX table
    with open(os.path.join(args.output_dir, 'metrics_table.tex'), 'w') as f:
        f.write("\\begin{table*}[t!]\n")
        f.write("\\centering\n")
        f.write("\\caption{Performance comparison of different restoration methods over the BRSR dataset.}\n")
        f.write("\\label{tab:results}\n")
        f.write("\\resizebox{\\linewidth}{!}{\n")
        f.write("\\begin{tabular}{lccc}\n")
        f.write("    \\toprule\n")
        f.write("    \\textbf{Restoration Algorithms} & \\textbf{Average SNR (dB)} & \\textbf{Average PSNR (dB)} & \\textbf{Average MSE} \\\\\n")
        f.write("    \\midrule\n")
        
        # Write corrupted signal metrics
        f.write(f"    \\cellcolor[gray]{{.93}}Corrupted Signal (Reference) & \\cellcolor[gray]{{.93}}{results['Corrupted Signal']['avg_snr']:.2f} & \\cellcolor[gray]{{.93}}{results['Corrupted Signal']['avg_psnr']:.2f} & \\cellcolor[gray]{{.93}}{results['Corrupted Signal']['avg_mse']:.2f} \\\\\n")
        
        # Write model metrics with alternating gray backgrounds
        row_color = False
        for config_name in model_configs_info.keys():
            if row_color:
                f.write(f"    \\cellcolor[gray]{{.93}}{config_name} & \\cellcolor[gray]{{.93}}{results[config_name]['avg_snr']:.2f} & \\cellcolor[gray]{{.93}}{results[config_name]['avg_psnr']:.2f} & \\cellcolor[gray]{{.93}}{results[config_name]['avg_mse']:.2f} \\\\\n")
            else:
                f.write(f"    {config_name} & {results[config_name]['avg_snr']:.2f} & {results[config_name]['avg_psnr']:.2f} & {results[config_name]['avg_mse']:.2f} \\\\\n")
            row_color = not row_color
        
        # Bold the best results (two-pass approach)
        best_model = 'BRSR-OpGAN (Q=3, Dual Domain, 2nd Pass)'
        if row_color:
            f.write(f"    \\cellcolor[gray]{{.93}}\\textbf{{{best_model}}} & \\cellcolor[gray]{{.93}}\\textbf{{{results[best_model]['avg_snr']:.2f}}} & \\cellcolor[gray]{{.93}}\\textbf{{{results[best_model]['avg_psnr']:.2f}}} & \\cellcolor[gray]{{.93}}\\textbf{{{results[best_model]['avg_mse']:.2f}}} \\\\\n")
        else:
            f.write(f"    \\textbf{{{best_model}}} & \\textbf{{{results[best_model]['avg_snr']:.2f}}} & \\textbf{{{results[best_model]['avg_psnr']:.2f}}} & \\textbf{{{results[best_model]['avg_mse']:.2f}}} \\\\\n")
        
        f.write("    \\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("}\n")
        f.write("\\end{table*}\n")
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main()
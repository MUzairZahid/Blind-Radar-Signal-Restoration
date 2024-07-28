import argparse
import os
import torch
import torch.nn as nn
from utils import *


def main():
    # Setup command-line arguments for configuring the training and evaluation process
    parser = argparse.ArgumentParser(description='Train and/or Evaluate a CycleGAN model on a dataset')
    parser.add_argument('--model', type=str, choices=['simple', 'residual'], default='residual', help='Choose the model architecture: Residual or simple')
    parser.add_argument('--Q', type=int, default=3, help='Set q value for SelfONN layers, with 1 representing a conventional CNN.')
    parser.add_argument('--mode', type=str, choices=['train', 'evaluate', 'both'], default='both', help='Define operation mode: train, evaluate, or both.')
    parser.add_argument('--epochs', type=int, default=1000, help='Specify the number of training epochs.')
    parser.add_argument('--lambda_recon', type=float, default=100, help='Reconstruction loss weight.')
    parser.add_argument('--lambda_freq', type=int, default=2, help='Frequency loss weight.')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size.')
    parser.add_argument('--device', type=str, default='cuda', help='Select the computation device: cpu or cuda.')
    parser.add_argument('--out_folder', type=str, default='saved_weights_hanning', help='Output folder for saving model weights and logs.')
    parser.add_argument('--data_folder', type=str, default='Prepared_Dataset', help='Folder containing the dataset.')
    parser.add_argument('--dataset', type=str, choices=['base', 'extended'], default='extended', help='Choose the dataset: base or extended.')
    parser.add_argument('--normalize', type=bool, default=True, help='Enable data normalization.')
    
    args = parser.parse_args()

    # Construct the path for the dataset based on the user's choices.
    dataset_filename = f"{args.dataset}_dataset.pickle"
    dataset_path = os.path.join(args.data_folder, dataset_filename)
    print(f"Selected dataset filename: {dataset_filename}")

    # Update and log the output directory name to reflect the configuration specifics
    args.out_folder = f"{args.out_folder}_BS{args.batch_size}_F{args.lambda_freq}_q{args.Q}_{args.model}_{args.dataset}"
    print(f"Configured output folder: {args.out_folder}")
    os.makedirs(args.out_folder, exist_ok=True)  # Ensure the output directory exists

    # Initialize models based on the provided configuration
    G, D = initialize_models(args.model, args.Q, args.device)

    # Train or evaluate the model based on the specified mode
    if args.mode in ['train', 'both']:
        # Train the model
        train_dual_loss(G, D, args.epochs, args.lambda_recon, args.batch_size, args.device, args.out_folder, args.normalize, dataset_path, args.lambda_freq)

    if args.mode in ['evaluate', 'both']:
        # Prepare the dataset for evaluation
        dataloaders = load_and_prepare_dataset(batch_size=64, dataset_path=dataset_path, normalize=args.normalize, num_workers=1, pin_memory=True)
        logging_file_path = os.path.join(args.out_folder, 'average_snr_values.txt')
        best_model_path = load_textfile_and_find_best_model(logging_file_path, args.out_folder)
        
        G.load_state_dict(torch.load(best_model_path, map_location=args.device))

        # Write results to a text file in the specified output folder
        results_file_path = os.path.join(args.out_folder, 'evaluation_results.txt')

        # Evaluate the model on the specified dataset
        if args.dataset == "base":
            results = evaluate_model_on_test_data_awgn(G, dataloaders, args.device)
            with open(results_file_path, 'w') as file:
                for snr_level, values in results.items():
                    mean_snr = np.mean(values)
                    file.write(f'True SNR {snr_level} dB: Mean Restored SNR: {mean_snr:.2f} dB\n')
        elif args.dataset == "extended":
            results = evaluate_model_on_test_data_blind(G, dataloaders, args.device)
            with open(results_file_path, 'w') as file:
                for key, value in results.items():
                    file.write(f'{key}: {value:.6f}\n')




        print(f'Results saved to {results_file_path}')

if __name__ == "__main__":
    main()

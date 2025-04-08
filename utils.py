import torch
from torch.utils.data import DataLoader, Dataset, Subset
import os
import pickle
from models_SelfONN_residual import *
import numpy as np
from models import *
import time
import json


class PSNRLoss(nn.Module):
    def __init__(self):
        super(PSNRLoss, self).__init__()

    def forward(self, signal, reference):
        """
        Calculate PSNR loss using the maximum absolute value from the reference signal
        
        Args:
            signal: The restored/predicted signal with shape [batch_size, 2, signal_length]
            reference: The clean/reference signal with shape [batch_size, 2, signal_length]
            
        Returns:
            Negative mean PSNR (as a loss to be minimized)
        """
        # Determine the actual max value from the reference signal (across both I/Q)
        max_value = torch.max(torch.abs(reference), dim=2)[0]
        max_value = torch.max(max_value, dim=1)[0]
        
        # Calculate MSE across all dimensions (separately for I and Q components)
        mse = torch.mean((signal - reference)**2, dim=2)
        # Average MSE across I/Q components
        mse = torch.mean(mse, dim=1)
        
        # Calculate PSNR for each batch sample using its specific max value
        psnr = 10 * torch.log10(max_value**2 / (mse + 1e-8))  # Adding small epsilon to avoid division by zero
        
        # Return negative mean PSNR as loss (since we minimize the loss)
        return -psnr.mean()

def train_dual_loss(G, D, num_epochs, lambda_recon, batch_size, device, out_folder, normalize, dataset_path, lambda_freq):
    """
    Train a GAN model on the specified dataset.

    Parameters:
    dataset_num (int): Number of the dataset to use for training.
    num_epochs (int): Number of epochs for training.
    lambda_recon (float): Weight for the reconstruction loss in the total loss calculation.
    batch_size (int): Batch size for training.
    device (str): Device to use for training ('cpu' or 'cuda').
    """

    # Load and prepare dataset based on the selected dataset
    dataloaders = load_and_prepare_dataset(batch_size=batch_size, dataset_path=dataset_path, normalize=normalize, 
                                           num_workers=4, pin_memory=True)

    # Lists to track training and validation losses and SNR values
    train_losses_g = []
    train_losses_d = []

    betas=(0.5, 0.999)
    G_params = list(G.parameters())
    D_params = list(D.parameters())
    # Initialize optimizers for Generator and Discriminator
    optimizer_g = torch.optim.Adam(G_params, lr=0.0005, betas=betas)
    optimizer_d = torch.optim.Adam(D_params, lr=0.0005, betas=betas)
    criterion_mae = nn.L1Loss()
    #criterion_mae = nn.MSELoss()
    criterion_mse = nn.MSELoss()

    if lambda_freq == 0: division=1 
    else: division =2

    for epoch in range(num_epochs):
        G.train()
        D.train()

        train_loss_g_freq_epoch = 0.0
        train_loss_g_time_epoch = 0.0
        train_loss_g_epoch = 0.0
        train_loss_d_epoch = 0.0

        for (clean_signal, clean_min, clean_max), \
               (distorted_signal, distorted_min, distorted_max), \
               label, snr_distorted, distortions in dataloaders['train']:
            batch_size = clean_signal.size(0)
            clean_signal = clean_signal.to(device)
            distorted_signal = distorted_signal.to(device)

            real_labels = torch.ones(batch_size, 1, device=device)
            fake_labels = torch.zeros(batch_size, 1, device=device)

            # Generator forward pass
            clean_signal_fake = G(distorted_signal)
            outputs_d_fake = D(clean_signal_fake)

            # Compute spectrograms directly on GPU and calculate the additional loss term
            spectrogram_clean = calculate_spectrogram_torch(clean_signal)
            spectrogram_fake = calculate_spectrogram_torch(clean_signal_fake)

            loss_g_mae_freq = criterion_mae(spectrogram_fake, spectrogram_clean) # Freq domain Rec Loss.
            loss_g_mae_time = criterion_mae(clean_signal_fake, clean_signal) # Time domain Rec Loss.

            loss_g_mae_freq = lambda_freq*loss_g_mae_freq

            loss_g_mae = (loss_g_mae_time +loss_g_mae_freq)/division

            loss_g_bce = criterion_mse(outputs_d_fake, real_labels)
            loss_g = loss_g_bce + lambda_recon * loss_g_mae

            # Generator backward and optimize
            optimizer_g.zero_grad()
            loss_g.backward()
            optimizer_g.step()

            # Discriminator forward pass and loss for real and fake
            out_real = D(clean_signal)
            loss_d_real = criterion_mse(out_real, real_labels)
            out_fake = D(clean_signal_fake.detach())
            loss_d_fake = criterion_mse(out_fake, fake_labels)
            loss_d = loss_d_real + loss_d_fake

            # Discriminator backward and optimize
            optimizer_d.zero_grad()
            loss_d.backward()
            optimizer_d.step()

            train_loss_g_epoch += loss_g.item()
            train_loss_d_epoch += loss_d.item()
            train_loss_g_freq_epoch += loss_g_mae_freq.item()
            train_loss_g_time_epoch += loss_g_mae_time.item()

        # Calculate average losses for the epoch
        avg_train_loss_g = train_loss_g_epoch / len(dataloaders['train'])
        avg_train_loss_d = train_loss_d_epoch / len(dataloaders['train'])
        avg_train_loss_g_freq = train_loss_g_freq_epoch / len(dataloaders['train'])
        avg_train_loss_g_time = train_loss_g_time_epoch / len(dataloaders['train'])

        # Log the losses
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss G: {avg_train_loss_g:.4f}, Loss D: {avg_train_loss_d:.4f}, '
            f'Loss G Freq: {avg_train_loss_g_freq:.4f}, Loss G Time: {avg_train_loss_g_time:.4f}')

        if epoch % 10 == 0:
            evaluate_and_save_model(epoch, G, D, dataloaders, device, out_folder)

class RadarSignalDataset(Dataset):
    def __init__(self, clean_signals, distorted_signals, labels, snr_distorted, distortions = None, normalize=True):
        """
        Custom Dataset for radar signals with optional normalization.
        """
        # Convert numpy arrays to torch tensors once rather than converting them every time __getitem__ is called
        self.clean_signals = torch.tensor(clean_signals, dtype=torch.float32)
        self.distorted_signals = torch.tensor(distorted_signals, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.snr_distorted = torch.tensor(snr_distorted, dtype=torch.float32) 
        self.distortions = torch.tensor(distortions, dtype=torch.float32) if distortions is not None else None
        self.normalize = normalize

        # Store min and max values for each channel of each signal
        self.clean_min_max = self.calculate_min_max(self.clean_signals)
        self.distorted_min_max = self.calculate_min_max(self.distorted_signals)
                

    def __len__(self):
        return len(self.clean_signals)

    def __getitem__(self, idx):
        clean_signal = self.clean_signals[idx]
        distorted_signal = self.distorted_signals[idx]
        label = self.labels[idx]
        snr_distorted = self.snr_distorted[idx]
        distortions = self.distortions[idx] if self.distortions is not None else torch.tensor([])

        clean_min, clean_max = self.clean_min_max[idx,:,0].view(-1,1), self.clean_min_max[idx,:,1].view(-1,1) # torch.Size([2, 1])
        distorted_min, distorted_max = self.distorted_min_max[idx,:,0].view(-1,1), self.distorted_min_max[idx,:,1].view(-1,1)
        
        if self.normalize:
            clean_signal = self.normalize_signal(clean_signal, clean_min, clean_max)
            distorted_signal = self.normalize_signal(distorted_signal, distorted_min, distorted_max)

        return (clean_signal, clean_min, clean_max), \
               (distorted_signal, distorted_min, distorted_max), \
               label, snr_distorted, distortions

    @staticmethod
    def normalize_signal(signal, signal_min, signal_max):
        """
        Normalize the multi-channel signal to the range [-1, 1].

        Parameters:
        signal (numpy array): The signal to be normalized.
        signal_min (numpy array): Min values for each channel.
        signal_max (numpy array): Max values for each channel.

        Returns:
        numpy array: Normalized signal.
        """
        return 2 * ((signal - signal_min) / (signal_max - signal_min)) - 1
    
    @staticmethod
    def calculate_min_max(signals):
        # Calculate the min and max across the last dimension (1024 samples)
        # and maintain the original dimensions of [49920, 2, 1] for compatibility
        min_vals = signals.min(dim=2, keepdim=True)[0]
        max_vals = signals.max(dim=2, keepdim=True)[0]

        # Combine the min and max into a single tensor for each signal
        # Resulting shape will be [49920, 2, 2] where the last dimension holds min and max values
        min_max_vals = torch.cat((min_vals, max_vals), dim=2)
    
        return min_max_vals



def load_and_prepare_dataset(batch_size, dataset_path, normalize=True, num_workers=1, pin_memory=False, subset_fraction=1.0):
    """
    Load the dataset and prepare it for model training, with options for pinned memory and multiple worker processes.
    Also supports using a subset of the dataset.

    Parameters:
    batch_size (int): Batch size for DataLoader.
    dataset_path (str): Path to the folder containing saved datasets.
    normalize (bool): Whether to normalize the signals.
    num_workers (int): Number of subprocesses to use for data loading.
    pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
    subset_fraction (float): Fraction of the dataset to use (between 0 and 1).

    Returns:
    dict: A dictionary containing DataLoader objects for training, validation, and test sets.
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
            normalize=normalize
        )

        # Determine the subset size for the current split
        subset_size = int(subset_fraction * len(radar_dataset))
        
        # Create random indices and subsets
        indices = np.random.choice(len(radar_dataset), subset_size, replace=False)
        subset = Subset(radar_dataset, indices)

        # Adjust DataLoader initialization to use multiple workers and pin memory
        dataloaders[split] = DataLoader(
            subset, 
            batch_size=batch_size, 
            shuffle=(split == 'train'),
            num_workers=num_workers, 
            pin_memory=pin_memory
        )
    return dataloaders

def evaluate_model_on_test_data_awgn(G, dataloaders, device):
    """
    Evaluate the trained model on test data and calculate mean SNR for each specific SNR value.

    Parameters:
    - model_path (str): Path to the trained model file.
    - dataloaders (dict): Dataloaders for the dataset.
    - device (torch.device): Device to perform the evaluation on.
    """
    G.eval()
    # Assuming ResidualGenerator is your model class

    snr_results = {}  # Dictionary to store results
    all_snr = []

    # Create keys based on the range from -14 to 10 with step size of 2
    for snr in range(-14, 12, 2):  # Using range function with step size of 2
        snr_results[snr] = []  # Assigning None as placeholder value

    with torch.no_grad():
        for (clean_signal, clean_min, clean_max), \
            (distorted_signal, distorted_min, distorted_max), \
            label, snr_distorted_batch, distortions in dataloaders['test']:
            clean_signal, distorted_signal = clean_signal.to(device), distorted_signal.to(device)
            clean_min, clean_max = clean_min.to(device), clean_max.to(device)
            restored_signal = G(distorted_signal)

            clean_signal = denormalize_signal(clean_signal, clean_min, clean_max)
            restored_signal = denormalize_signal(restored_signal, clean_min, clean_max)

            snr_restored_batch = calculate_snr(restored_signal, clean_signal).cpu().numpy()
            for value, snr_value in zip(snr_restored_batch, snr_distorted_batch.cpu().numpy()):
                snr_results[snr_value[0]].append(value)
                all_snr.append(value)

    # Calculate and print the mean SNR for each SNR level
    for snr_level in sorted(snr_results.keys()):
        mean_snr = np.mean(snr_results[snr_level])
        print(f'True SNR {snr_level} dB: Mean Restored SNR: {mean_snr:.2f} dB')


    print(f'Mean SNR dB: {np.mean(all_snr):.2f} dB')

    return snr_results


def evaluate_model_on_test_data_blind(G, dataloaders, device):
    G.eval()
    snr_noisy_list = []
    snr_restored_list = []
    mse_noisy_list = []
    mse_restored_list = []
    psnr_noisy_list = []
    psnr_restored_list = []
    
    total_inference_time = 0  # To accumulate total inference time
    num_samples = 0  # To count the number of samples processed
    
    with torch.no_grad():
        for (clean_signal, clean_min, clean_max), \
               (distorted_signal, distorted_min, distorted_max), \
               label, snr_distorted, distortions in dataloaders['test']:
            clean_signal = clean_signal.to(device)
            distorted_signal = distorted_signal.to(device)
            clean_min, clean_max = clean_min.to(device), clean_max.to(device)
            distorted_min, distorted_max = distorted_min.to(device), distorted_max.to(device)
            
            # Start timing the inference
            start_time = time.time()
            restored_signal = G(distorted_signal)
            # Stop timing the inference
            inference_time = time.time() - start_time
            
            total_inference_time += inference_time
            num_samples += clean_signal.size(0)
            
            # Denormalize signals
            clean_signal = denormalize_signal(clean_signal, clean_min, clean_max)
            distorted_signal = denormalize_signal(distorted_signal, distorted_min, distorted_max)
            restored_signal = denormalize_signal(restored_signal, clean_min, clean_max)

            # Calculate metrics for noisy input
            snr_noisy = snr_distorted.cpu().numpy()
            mse_noisy = F.mse_loss(distorted_signal, clean_signal, reduction='none').mean(dim=[1, 2]).cpu().numpy()
            psnr_noisy = calculate_psnr(distorted_signal, clean_signal).cpu().numpy()

            # Calculate metrics for restored signal
            snr_restored = calculate_snr(restored_signal, clean_signal).cpu().numpy()
            mse_restored = F.mse_loss(restored_signal, clean_signal, reduction='none').mean(dim=[1, 2]).cpu().numpy()
            psnr_restored = calculate_psnr(restored_signal, clean_signal).cpu().numpy()
            
            snr_noisy_list.extend(snr_noisy)
            snr_restored_list.extend(snr_restored)
            mse_noisy_list.extend(mse_noisy)
            mse_restored_list.extend(mse_restored)
            psnr_noisy_list.extend(psnr_noisy)
            psnr_restored_list.extend(psnr_restored)
    
    avg_snr_noisy = np.mean(snr_noisy_list)
    avg_snr_restored = np.mean(snr_restored_list)
    avg_mse_noisy = np.mean(mse_noisy_list)
    avg_mse_restored = np.mean(mse_restored_list)
    avg_psnr_noisy = np.mean(psnr_noisy_list)
    avg_psnr_restored = np.mean(psnr_restored_list)
    
    # Calculate average inference time per sample
    avg_inference_time_per_sample = total_inference_time / num_samples
    
    print(f'Average Noisy SNR: {avg_snr_noisy:.2f} dB')
    print(f'Average Restored SNR: {avg_snr_restored:.2f} dB')
    print(f'Average Noisy MSE: {avg_mse_noisy:.6f}')
    print(f'Average Restored MSE: {avg_mse_restored:.6f}')
    print(f'Average Noisy PSNR: {avg_psnr_noisy:.2f} dB')
    print(f'Average Restored PSNR: {avg_psnr_restored:.2f} dB')
    print(f'Average Inference Time per Sample: {avg_inference_time_per_sample:.6f} seconds')
    
    return {
        'avg_snr_noisy': avg_snr_noisy,
        'avg_snr_restored': avg_snr_restored,
        'avg_mse_noisy': avg_mse_noisy,
        'avg_mse_restored': avg_mse_restored,
        'avg_psnr_noisy': avg_psnr_noisy,
        'avg_psnr_restored': avg_psnr_restored,
        'avg_inference_time_per_sample': avg_inference_time_per_sample
    }


def load_textfile_and_find_best_model(file_path, model_dir):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Extracting epoch and output SNR from each non-empty line
    snr_data = {}
    for line in lines:
        if line.strip():  # Check if the line is not empty
            parts = line.strip().split()
            epoch = int(parts[1][:-1])
            output_snr = float(parts[8])
            snr_data[epoch] = output_snr

    # Finding the epoch with the best output SNR
    best_epoch = max(snr_data, key=snr_data.get)
    best_snr = snr_data[best_epoch]

    print(f"Best Model: Epoch {best_epoch} with Output SNR: {best_snr:.2f}")

    # Construct the path to the best model
    best_model_path = os.path.join(model_dir, f"model_{best_epoch}.pth")

    return best_model_path

def calculate_snr(noisy_signal, clean_signal):
    """
    Calculate the Signal-to-Noise Ratio (SNR) for batched inputs by using
    the absolute squares of the I and Q components.

    Parameters:
    noisy_signal (Tensor): The noisy signals, assumed to be of shape (batch_size, 2, signal_length),
                           where the second dimension contains I and Q components.
    clean_signal (Tensor): The clean signals, assumed to be of shape (batch_size, 2, signal_length),
                           where the second dimension contains I and Q components.

    Returns:
    Tensor: The SNR values in dB for each sample in the batch.
    """
    # Calculate the noise
    noise = noisy_signal - clean_signal

    # Calculate signal and noise power using their I and Q components
    signal_power_i = torch.mean(clean_signal[:, 0, :]**2, dim=-1)
    signal_power_q = torch.mean(clean_signal[:, 1, :]**2, dim=-1)
    noise_power_i = torch.mean(noise[:, 0, :]**2, dim=-1)
    noise_power_q = torch.mean(noise[:, 1, :]**2, dim=-1)

    # Total signal and noise power
    signal_power = signal_power_i + signal_power_q
    noise_power = noise_power_i + noise_power_q

    # Calculate SNR, adding a small epsilon to noise_power to avoid division by zero
    snr = signal_power / (noise_power + 1e-6)
    snr_db = 10 * torch.log10(snr)

    return snr_db

def calculate_psnr(signal, reference, max_value=1.0):
    mse_i = torch.mean((signal[:, 0, :] - reference[:, 0, :])**2, dim=-1)
    mse_q = torch.mean((signal[:, 1, :] - reference[:, 1, :])**2, dim=-1)
    mse = mse_i + mse_q
    psnr = 10 * torch.log10(max_value ** 2 / mse)
    return psnr


def evaluate_and_save_model(epoch, G, D, dataloaders, device, save_dir):
    G.eval()
    with torch.no_grad():
        # Initialize lists to store SNR values for train and validation sets
        snr_values = {'train': [], 'validation': []}
        
        for split in ['train', 'validation']:
            for (clean_signal, clean_min, clean_max), \
               (distorted_signal, distorted_min, distorted_max), \
               label, snr_distorted, distortions in dataloaders[split]:
                clean_signal, distorted_signal = clean_signal.to(device), distorted_signal.to(device)
                clean_min, clean_max = clean_min.to(device), clean_max.to(device)
                restored_signal = G(distorted_signal)

                clean_signal = denormalize_signal(clean_signal, clean_min, clean_max)
                restored_signal = denormalize_signal(restored_signal, clean_min, clean_max)


                # SNR calculation now entirely in PyTorch
                snr_batch = calculate_snr(restored_signal, clean_signal).cpu().numpy()  # Only convert to numpy for storing or printing
                snr_values[split].extend(snr_batch.tolist())  # Convert to list if needed for appending

        # Calculate the average SNR for validation and test sets
        average_snr = {split: torch.tensor(snr_values[split]).mean().item() for split in snr_values}  # Use PyTorch for mean calculation
        print('*'*50)
        print(f'Epoch {epoch}: Validation SNR: {average_snr["validation"]:.2f}, Train SNR: {average_snr["train"]:.2f}')
        print('*'*50)
        
        # Save the models in a single file
        model_path = os.path.join(save_dir, 'model_max_valSNR.pth')
        torch.save({
            'A_state_dict': G.state_dict(),
            'M_state_dict': D.state_dict(),
        }, model_path)

        # Save average SNR values to a text file
        save_logging(save_dir, epoch, average_snr)



def save_logging(save_dir, epoch, average_snr):
    snr_file_path = os.path.join(save_dir, 'average_snr_values.txt')
    with open(snr_file_path, 'a') as f:
        # Combine epoch and SNR values in a single line
        f.write(f'Epoch {epoch}: ' + ', '.join([f'{key} SNR: {value:.2f} dB' for key, value in average_snr.items()]) + '\n')


def calculate_spectrogram_torch(signals, n_fft=256, hop_length=128, win_length=256, power=2.0):
    """
    Calculate the spectrogram of signals using PyTorch, keeping the computation on the GPU.
    
    Parameters:
    - signals: Tensor of shape (batch_size, 2, signal_length)
    - n_fft: Number of FFT points
    - hop_length: Number of samples between successive frames
    - win_length: Each frame of audio is windowed by `window()` and will have length `win_length`
    - power: Exponent for the magnitude spectrogram
    
    Returns:
    - Tensor of spectrograms of shape (batch_size, freq_bins, time_steps)
    """
    # signals is expected to be real-valued, with dimension (batch_size, 2, signal_length)
    signals = signals.permute(0, 2, 1)  # Now shape is (batch_size, signal_length, 2)
    # Make sure signals tensor is contiguous in memory, especially after reshaping
    signals = signals.contiguous()
    signals_complex = torch.view_as_complex(signals)  # Convert to complex tensor

    # Create Hanning window
    window = torch.hann_window(win_length).to(signals.device)
    
    # Compute STFT
    stft = torch.stft(signals_complex, n_fft=n_fft, hop_length=hop_length, win_length=win_length, return_complex=True, window=window)
    
    # Compute spectrogram as the squared magnitude of the STFT
    spectrogram = torch.abs(stft)**power
    
    # Normalize spectrogram
    spectrogram_min = spectrogram.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
    spectrogram_max = spectrogram.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    normalized_spectrogram = 2 * (spectrogram - spectrogram_min) / (spectrogram_max - spectrogram_min) - 1
    
    return normalized_spectrogram




def denormalize_signal(normalized_signal, original_min, original_max):
    """
    Denormalize the signal from the range [-1, 1] back to its original range.

    Parameters:
    normalized_signal (numpy array or torch tensor): The normalized signal.
    original_min (float): The minimum value of the original signal before normalization.
    original_max (float): The maximum value of the original signal before normalization.

    Returns:
    numpy array or torch tensor: Denormalized signal.
    """
    return ((normalized_signal + 1) / 2) * (original_max - original_min) + original_min


class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.early_stop = False
        self.counter = 0

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

def initialize_models(model_type, q, device):
    """
    Initialize the specified type of generator and discriminator models.
    
    Parameters:
        model_type (str): Type of model to initialize ('simple' or 'residual').
        q (int): Specifies the 'q' value for SelfONN Layers, where 1 indicates a standard CNN.
        device (str): Device to which the models will be loaded ('cuda' or 'cpu').
        
    Returns:
        tuple: Initialized generator and discriminator models.
    """
    if model_type == 'simple':
        G = Generator(q=q).to(device)
        D = Discriminator(q=q).to(device)
    elif model_type == 'residual':
        G = ResidualGenerator(q=q).to(device)
        D = ResidualDiscriminator(q=q).to(device)
    return G, D

def save_config(args, config_path):
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=4)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return argparse.Namespace(**config)

def save_best_model(A, M, args):
    torch.save({
        'A_state_dict': A.state_dict(),
        'M_state_dict': M.state_dict(),
    }, args.model_path)
    print(f"Saved best model with improved validation SNR to {args.model_path}")
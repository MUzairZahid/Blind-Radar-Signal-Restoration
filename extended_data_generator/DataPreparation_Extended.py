import h5py
import numpy as np
import os
from sklearn.model_selection import train_test_split
import pickle

def split_data(X_clean, X_noisy, Y_label, Y_SNR, distortions):
    """
    Split data into training, validation, and test sets along with distortions.
    This function directly uses the clean and noisy signal arrays instead of pairing them first, 
    which is more memory efficient and avoids unnecessary zipping and unzipping.
    """
    # Split the dataset into training and validation with a 80-20 split
    (X_clean_train, X_clean_validation, 
    X_noisy_train, X_noisy_validation, 
    Y_label_train, Y_label_validation, 
    Y_SNR_train, Y_SNR_validation,
    distortions_train, distortions_validation) = train_test_split(X_clean, X_noisy, Y_label, Y_SNR, distortions, test_size=0.2, random_state=42, stratify=Y_label)


    
    # Return the split datasets
    return (X_clean_train, X_noisy_train, Y_label_train, Y_SNR_train, distortions_train,
            X_clean_validation, X_noisy_validation, Y_label_validation, Y_SNR_validation, distortions_validation)

def main():
    data_folder_train = "Dataset_1_blind_train/" # Will be used for training and validation data after split.
    data_folder_test = "Dataset_1_blind_test/" # Will be used for additional test data. 

    data_folder_prepared = "Prepared_Dataset/"
    data_file_name = 'extended_dataset.pickle'

    variablesToLoad = [
        'X_clean_signals_all', 
        'X_distorted_signals_all', 
        'X_distortion_awgn_all',
        'X_distortion_echo_all',
        'X_distortion_cci_all',
        'Y_label_all', 
        'Y_SNR_all'

    ]   

    print("Training Data.")

    # Load data from .mat files and adjust the dimensionality if required
    loaded_data_train = {var: np.transpose(np.array(h5py.File(os.path.join(data_folder_train, var + '.mat'), 'r')[var]))
                   for var in variablesToLoad}

    # Display shapes of loaded data for verification
    for var, data in loaded_data_train.items():
        print(f"{var} shape: {data.shape}")

    # Prepare the dataset by extracting the necessary arrays directly
    X_clean = loaded_data_train['X_clean_signals_all']
    X_noisy = loaded_data_train['X_distorted_signals_all']
    Y_label = loaded_data_train['Y_label_all']
    Y_SNR = loaded_data_train['Y_SNR_all']
    distortions = np.stack((loaded_data_train['X_distortion_awgn_all'], loaded_data_train['X_distortion_echo_all'], loaded_data_train['X_distortion_cci_all']), axis=1)


    # Splitting data into train, validation, and test sets along with distortions
    results = split_data(X_clean, X_noisy, Y_label, Y_SNR, distortions)


    print("Test Data.")
    # Load data from .mat files and adjust the dimensionality if required
    loaded_data_test = {var: np.transpose(np.array(h5py.File(os.path.join(data_folder_test, var + '.mat'), 'r')[var]))
                   for var in variablesToLoad}

    # Display shapes of loaded data for verification
    for var, data in loaded_data_test.items():
        print(f"{var} shape: {data.shape}")

    # Prepare the dataset by extracting the necessary arrays directly
    X_clean_test = loaded_data_test['X_clean_signals_all']
    X_noisy_test = loaded_data_test['X_distorted_signals_all']
    Y_label_test = loaded_data_test['Y_label_all']
    Y_SNR_test = loaded_data_test['Y_SNR_all']
    distortions_test = np.stack((loaded_data_test['X_distortion_awgn_all'], loaded_data_test['X_distortion_echo_all'], loaded_data_test['X_distortion_cci_all']), axis=1)




    


    # Organize data into a dictionary for easy access and clarity
    data = {
        'train': {
            'clean': results[0],
            'noisy': results[1],
            'label': results[2],
            'SNR': results[3],
            'distortions' : results[4]
        },
        'validation': {
            'clean': results[5],
            'noisy': results[6],
            'label': results[7],
            'SNR': results[8],
            'distortions' : results[9]
        },
        'test': {
            'clean': X_clean_test,
            'noisy': X_noisy_test,
            'label': Y_label_test,
            'SNR': Y_SNR_test,
            'distortions' : distortions_test
        }
    }


    # Save the organized dataset
    os.makedirs(data_folder_prepared, exist_ok = True)

    file_name = os.path.join(data_folder_prepared, data_file_name)
    with open(file_name, 'wb') as file:
        pickle.dump(data, file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {file_name}")

    # Print the number of examples in each dataset for verification
    print(f"Number of training examples: {len(data['train']['label'])}")
    print(f"Number of validation examples: {len(data['validation']['label'])}")
    print(f"Number of testing examples: {len(data['test']['label'])}")

if __name__ == "__main__":
    main()

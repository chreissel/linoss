"""
This script processes the UEA datasets and saves the processed data in the data_dir/processed directory.
It has been adapted to Jax from https://github.com/jambo6/neuralRDEs
"""

import os
import pickle
import warnings

import jax.numpy as jnp
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sktime.datasets import load_from_arff_to_dataframe
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import h5py


def save_pickle(obj, filename):
    """Saves a pickle object."""
    with open(filename, "wb") as handle:
        pickle.dump(obj, handle, protocol=pickle.HIGHEST_PROTOCOL)


def create_jax_data(bkg_file, sig_file, test_frac=0.5):
    """Creates jax tensors for test and training from the numpy format.

    Args:
        train_file (str): The location of the training data numpy file.
        test_file (str): The location of the testing data numpy file.

    Returns:
        data_train, data_test, labels_train, labels_test: All as jax tensors.
    """

    def h5_loader(path,label, test_frac=0.5):
        with h5py.File(path, 'r') as f:
            group_key = list(f.keys())[0]
            data = f[group_key][()]
        f.close()
        # downsample data by factor 4 to 1024Hz to save space on the GPU
        #data = data[:,::4]
        data = data[:, :, np.newaxis]
        target = np.full(data.shape[0], label)
        return data, target

    def convert_data(data_numpy):
        data_jnumpy = jnp.array(data_numpy)
        return data_jnumpy

    background = h5_loader(bkg_file, label=0)
    signal = h5_loader(sig_file, label=1)
    data_numpy = np.concatenate([background[0], signal[0]], axis=0)
    targets_numpy = np.concatenate([background[1], signal[1]], axis=0)

    # shuffle and split into train/test data
    train_data, test_data, train_labels, test_labels = train_test_split(data_numpy, targets_numpy, test_size=test_frac, random_state=42)
    train_data, test_data = convert_data(train_data), convert_data(test_data)

    encoder = LabelEncoder().fit(train_labels)
    train_labels, test_labels = encoder.transform(train_labels), encoder.transform(
        test_labels
    )
    train_labels, test_labels = jnp.array(train_labels), jnp.array(test_labels)

    return train_data, test_data, train_labels, test_labels


def convert_all_files(data_dir):
    """Convert BNS data into jax data to be stored in /processed."""
    arff_folder = data_dir + "/raw/bns/"

    signal = arff_folder + "/whitened_noiseplusbns_8sec_4096_5k.h5"
    background = arff_folder + "/whitened_noise_8sec_4096_5k.h5"

    save_dir = data_dir + "/processed/gw/bns/"

    if os.path.isdir(save_dir):
        print("Files already exist for: bns")
    else:
        os.makedirs(save_dir)
        train_data, test_data, train_labels, test_labels = create_jax_data(
            background, signal
        )
        data = jnp.concatenate([train_data, test_data])
        labels = jnp.concatenate([train_labels, test_labels])

        unique_rows, indices, inverse_indices = np.unique(
                data, axis=0, return_index=True, return_inverse=True
            )
        data = data[indices]
        labels = labels[indices]
        print(
              f"Deleting {len(inverse_indices) - len(indices)} repeated samples in bns"
        )

        original_idxs = (
           jnp.arange(0, train_data.shape[0]),
           jnp.arange(train_data.shape[0], data.shape[0]),
        )

        save_pickle(data, save_dir + "/data.pkl")
        save_pickle(labels, save_dir + "/labels.pkl")
        save_pickle(original_idxs, save_dir + "/original_idxs.pkl")


if __name__ == "__main__":
    data_dir = "data_dir"
    convert_all_files(data_dir)

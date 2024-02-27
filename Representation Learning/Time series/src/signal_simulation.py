import os
import pickle

import numpy as np
import timesynth as ts
import torch
from tqdm import tqdm


def _create_signal(sig_len, window_size=50):
    n_states = 4
    transition_matrix = np.eye(n_states) * 0.85
    transition_matrix[0, 1] = transition_matrix[1, 0] = 0.05
    transition_matrix[0, 2] = transition_matrix[2, 0] = 0.05
    transition_matrix[0, 3] = transition_matrix[3, 0] = 0.05
    transition_matrix[2, 3] = transition_matrix[3, 2] = 0.05
    transition_matrix[2, 1] = transition_matrix[1, 2] = 0.05
    transition_matrix[3, 1] = transition_matrix[1, 3] = 0.05

    states = []
    sig_1 = []
    sig_2 = []
    sig_3 = []
    pi = np.ones((1, n_states)) / n_states

    for _ in range(sig_len // window_size):
        current_state = np.random.choice(n_states, 1, p=pi.reshape(-1))
        states.extend(list(current_state) * window_size)

        current_signal = _ts_generator(current_state[0], window_size)
        sig_1.extend(current_signal)

        correlated_signal = (
                current_signal * 0.9
                + .03
                + 0.4 * np.random.randn(len(current_signal))
        )
        sig_2.extend(correlated_signal)

        uncorrelated_signal = _ts_generator(
            (current_state[0] + 2) % 4, window_size,
            )
        sig_3.extend(uncorrelated_signal)

        pi = transition_matrix[current_state]

    signals = np.stack([sig_1, sig_2, sig_3])
    return signals, states


def _ts_generator(state, window_size):
    time_sampler = ts.TimeSampler(stop_time=window_size)
    sampler = time_sampler.sample_regular_time(num_points=window_size)
    white_noise = ts.noise.GaussianNoise(std=0.3)

    if state == 0:
        sig_type = ts.signals.GaussianProcess(
            kernel="Periodic",
            lengthscale=1.,
            mean=0.,
            variance=.1,
            p=5,
        )
    elif state == 1:
        sig_type = ts.signals.NARMA(
            order=5,
            initial_condition=[0.671, 0.682, 0.675, 0.687, 0.69],
        )
    elif state == 2:
        sig_type = ts.signals.GaussianProcess(
            kernel="SE",
            lengthscale=1.,
            mean=0.,
            variance=.1,
        )
    elif state == 3:
        sig_type = ts.signals.NARMA(
            order=3,
            coefficients=[0.1, 0.25, 2.5, -0.005],
            initial_condition=[1, 0.97, 0.96],
        )

    timeseries = ts.TimeSeries(sig_type, noise_generator=white_noise)
    samples, _, _ = timeseries.sample(sampler)
    return samples


def _normalize(train_data, test_data):
    """Mean normalization of train and test sets based on train statistics"""
    feature_size = train_data.shape[1]
    sig_len = train_data.shape[2]
    d = [x.T for x in train_data]
    d = np.stack(d, axis=0)

    feature_means = np.mean(train_data, axis=(0, 2))
    feature_std = np.std(train_data, axis=(0, 2))
    np.seterr(divide="ignore", invalid="ignore")
    train_data_n = (
        train_data - feature_means[np.newaxis, :, np.newaxis] /
        np.where(feature_std == 0, 1, feature_std)[np.newaxis, :, np.newaxis]
    )
    test_data_n = (
        test_data - feature_means[np.newaxis, :, np.newaxis] /
        np.where(feature_std == 0, 1, feature_std)[np.newaxis, :, np.newaxis]
    )

    return train_data_n, test_data_n


def simulate_signal(
    n_samples: int = 100,
    sig_len: int = 1000,
    seed: int = 42,
):
    train_size = 0.8

    np.random.seed(seed)

    all_signals = []
    all_states = []

    for _ in tqdm(range(n_samples), desc="Samples"):
        sample_signal, sample_state = _create_signal(sig_len)
        all_signals.append(sample_signal)
        all_states.append(sample_state)

    dataset = np.array(all_signals)
    states = np.array(all_states)

    # Split to train_val and test
    n_train_val = int(len(dataset) * train_size)

    train_val_data = dataset[:n_train_val]
    train_val_state = states[:n_train_val]

    test_data = dataset[n_train_val:]
    test_state = torch.tensor(states[n_train_val:])

    train_val_data, test_data = _normalize(train_val_data, test_data)

    test_data = torch.tensor(test_data).float()

    # Split train_val into train/val
    n_train = int(len(train_val_data) * train_size)

    train_data = torch.tensor(train_val_data[:n_train]).float()
    train_state = torch.tensor(train_val_state[:n_train])

    val_data = torch.tensor(train_val_data[n_train:]).float()
    val_state = torch.tensor(train_val_state[n_train:])

    print(
        f"Dataset Shape:\n"
        f"Train: {train_data.shape}\n"
        f"Val: {val_data.shape}\n"
        f"Test: {test_data.shape}"
    )

    # Save signals to file
    output_data = {
        "x_train": train_data,
        "y_train": train_state,

        "x_val": val_data,
        "y_val": val_state,

        "x_test": test_data,
        "y_test": test_state,

        "x_all": torch.tensor(dataset).float(),
        "y_all": torch.tensor(states),
    }

    return output_data

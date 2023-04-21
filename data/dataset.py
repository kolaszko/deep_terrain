import pickle

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset


class HapticDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        signal = torch.from_numpy(sample['signal']).float()
        friction = torch.tensor(sample['friction']).float()
        label = torch.tensor(sample['label']).long()
        return signal, friction, label


def get_putany_cls_dataset(path, split_size, normalize=True, random_state=42):
    with open(path, 'rb') as f:
        dataset_dict = pickle.load(f, encoding='latin1')

    dataset = []
    for _, samples in dataset_dict.items():
        for sample in samples:
            dataset.append({
                'signal': sample['signal'],
                'friction': sample['friction'],
                'label': sample['label']
            })

    labels = [sample['label'] for sample in dataset]
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=split_size, stratify=labels, random_state=random_state)

    if normalize:
        signals = [s['signal'] for s in train_dataset]
        signals = np.asarray(signals)
        mean = np.mean(signals, (0, 1))
        std = np.std(signals, (0, 1))

        for s in train_dataset:
            s['signal'] = (s['signal'] - mean) / std

        for s in test_dataset:
            s['signal'] = (s['signal'] - mean) / std

    train_dataset, val_dataset = train_test_split(train_dataset, test_size=split_size, stratify=[
                                                  sample['label'] for sample in train_dataset], random_state=random_state)

    weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    return train_dataset, val_dataset, test_dataset, weights


def get_putany_regression_dataset(path, split_size, exclude_classes, normalize=True, random_state=42):
    with open(path, 'rb') as f:
        dataset_dict = pickle.load(f, encoding='latin1')

    dataset = []
    for _, samples in dataset_dict.items():
        for sample in samples:
            dataset.append({
                'signal': sample['signal'],
                'friction': sample['friction'],
                'label': sample['label']
            })

    test_dataset = [sample for sample in dataset if sample['label'] in exclude_classes]
    trainval_dataset = [sample for sample in dataset if sample['label'] not in exclude_classes]

    if normalize:
        signals = [s['signal'] for s in trainval_dataset]
        coeffs = [s['friction'] for s in dataset]

        signals = np.asarray(signals)
        coeffs = np.asarray(coeffs)

        mean_signal = np.mean(signals, (0, 1))
        std_signal = np.std(signals, (0, 1))

        max_coeffs = np.max(coeffs)
        min_coeffs = np.min(coeffs)

        for s in trainval_dataset:
            s['signal'] = (s['signal'] - mean_signal) / std_signal
            s['friction'] = (s['friction'] - min_coeffs) / (max_coeffs - min_coeffs)

        for s in test_dataset:
            s['signal'] = (s['signal'] - mean_signal) / std_signal
            s['friction'] = (s['friction'] - min_coeffs) / (max_coeffs - min_coeffs)

    train_dataset, val_dataset = train_test_split(trainval_dataset, test_size=split_size, stratify=[
        sample['label'] for sample in trainval_dataset], random_state=random_state)

    return train_dataset, val_dataset, test_dataset, None


def get_moist_cls_dataset(path, split_size, random_state=42):
    ds_raw = np.loadtxt(path)

    dataset = []
    for i in range(ds_raw.shape[0]):
        sig = ds_raw[i, 1:]
        sig = np.reshape(sig, (6, sig.shape[0] // 6))
        dataset.append(
            {
                'signal': np.transpose(sig, (1, 0)),
                'friction': 0.0,
                'label': ds_raw[i, 0]
            }
        )

    labels = [sample['label'] for sample in dataset]
    train_dataset, test_dataset = train_test_split(
        dataset, test_size=split_size, stratify=labels, random_state=random_state)

    train_dataset, val_dataset = train_test_split(train_dataset, test_size=split_size, stratify=[
                                                  sample['label'] for sample in train_dataset], random_state=random_state)

    weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    return train_dataset, val_dataset, test_dataset, weights


def get_moist_regression_dataset(path, split_size, exclude_classes, random_state=42):
    ds_raw = np.loadtxt(path)

    moist_table = np.asarray([0.12, 0.2, 0.32, 0.56, 0.64, 0.76])
    moist_table = (moist_table - np.min(moist_table)) / (np.max(moist_table) - np.min(moist_table))

    dataset = []
    for i in range(ds_raw.shape[0]):
        sig = ds_raw[i, 1:]
        sig = np.reshape(sig, (6, sig.shape[0] // 6))
        dataset.append(
            {
                'signal': np.transpose(sig, (1, 0)),
                'friction': moist_table[int(ds_raw[i, 0])],
                'label': ds_raw[i, 0]
            }
        )

    test_dataset = [sample for sample in dataset if sample['label'] in exclude_classes]
    trainval_dataset = [sample for sample in dataset if sample['label'] not in exclude_classes]

    train_dataset, val_dataset = train_test_split(trainval_dataset, test_size=split_size, stratify=[
        sample['label'] for sample in trainval_dataset], random_state=random_state)

    return train_dataset, val_dataset, test_dataset, None

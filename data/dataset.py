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


def get_cls_dataset(path, split_size, normalize=True, random_state=42):
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
        signals = [s['signal'] for s in dataset]
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


def get_regression_dataset(path, split_size, exclude_classes, random_state=42):
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

    train_dataset, val_dataset = train_test_split(trainval_dataset, test_size=split_size, stratify=[
        sample['label'] for sample in trainval_dataset], random_state=random_state)

    return train_dataset, val_dataset, test_dataset, None

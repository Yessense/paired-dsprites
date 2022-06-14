import random
from typing import Tuple, List, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import IterableDataset, DataLoader, Dataset
import itertools
import operator


class Dsprites(Dataset):
    """Store dsprites images"""

    def __init__(self, path='./dataset/data/dsprite_train.npz', max_exchanges=1):
        # ----------------------------------------
        # Load dataset
        # ----------------------------------------

        self.max_exchanges = max_exchanges
        # Load npz numpy archive
        dataset_zip = np.load(path)

        # Images: numpy array -> (737280, 64, 64)
        self.imgs = dataset_zip['imgs']

        # Labels: numpy array -> (737280, 5)
        # Each column contains int value in range of `features_count`
        self.labels = dataset_zip['latents_classes'][:, 1:]

        # ----------------------------------------
        # Dataset info
        # ----------------------------------------

        # Size of dataset (737280)
        self.size: int = self.imgs.shape[0]

        # ----------------------------------------
        # features info
        # ----------------------------------------

        # List of feature names
        self.feature_names: Tuple[str, ...] = ('shape', 'scale', 'orientation', 'posX', 'posY')

        # Feature numbers
        self.features_list: List[int] = list(range(len(self.feature_names)))

        # Count each feature counts
        self.features_count = [3, 6, 40, 32, 32]

        # Getting multipler for each feature position
        self.features_range = [list(range(i)) for i in self.features_count]
        self.multiplier = list(itertools.accumulate(self.features_count[-1:0:-1], operator.mul))[::-1] + [1]

    def _get_element_pos(self, labels: List[int]) -> int:
        """
        Get position of image with `labels` in dataset

        Parameters
        ----------
        labels:

        Returns
        -------
        pos: int
            Position in dataset
        """
        pos = 0
        for mult, label in zip(self.multiplier, labels):
            pos += mult * label
        return pos

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Choose index image
        img = self.imgs[idx]
        labels = self.labels[idx]

        # choice number of exchanges
        n_exchanges = random.randrange(1, self.max_exchanges + 1)

        # select features that will be exchanged
        exchanges = random.sample(population=self.features_list, k=n_exchanges)

        exchange_labels = np.zeros_like(labels, dtype=bool)
        pair_img_labels = labels[:]

        for feature_type in exchanges:
            # Find other feature and add his number to pair_img_labels
            exchange_labels[feature_type] = True
            other_feature = random.choice(self.features_range[feature_type])

            while other_feature == labels[feature_type]:
                other_feature = random.choice(self.features_range[feature_type])

            pair_img_labels[feature_type] = other_feature

        pair_idx: int = self._get_element_pos(pair_img_labels)
        pair_img = self.imgs[pair_idx]

        img = torch.from_numpy(img).float().unsqueeze(0)
        pair_img = torch.from_numpy(pair_img).float().unsqueeze(0)
        exchange_labels = torch.from_numpy(exchange_labels).unsqueeze(-1)

        return img, pair_img, exchange_labels


class PairedDspritesDataset(Dataset):
    def __init__(self,
                 dsprites_path='./dataset/data/dsprites_train.npz',
                 paired_dsprites_path='./dataset/data/100_30_dataset/paired_train.npz'):
        # Load npz numpy archive
        dsprites = np.load(dsprites_path, allow_pickle=True)
        paired_dsprites = np.load(paired_dsprites_path, allow_pickle=True)

        self.data = paired_dsprites['data']
        self.exchanges = paired_dsprites['exchanges']

        # Images: numpy array -> (737280, 64, 64)
        self.imgs = dsprites['imgs']

        # List of feature names
        self.feature_names: Tuple[str, ...] = ('shape', 'scale', 'orientation', 'posX', 'posY')

        # Labels: numpy array -> (737280, 5)
        # Each column contains int value in range of `features_count`
        self.labels = dsprites['latents_classes'][:, 1:]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.imgs[self.data[idx][0]]
        img = torch.from_numpy(img).float().unsqueeze(0)

        pair_img = self.imgs[self.data[idx][1]]
        pair_img = torch.from_numpy(pair_img).float().unsqueeze(0)

        exchange = torch.from_numpy(self.exchanges[idx]).bool().unsqueeze(-1)
        return img, pair_img, exchange


def show_img(labels: Optional[List[int]] = None) -> None:
    """ Plot one image with given feature numbers """
    if labels is None:
        labels = [2, 5, 39, 31, 31]

    img = md.imgs[md._get_element_pos(labels)]
    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    md = Dsprites()

    batch_size = 5
    loader = DataLoader(md, batch_size=batch_size, shuffle=True)

    batch = next(iter(loader))

    fig, ax = plt.subplots(2, batch_size, figsize=(10, 5))
    for i in range(batch_size):
        img = batch[0][i]
        pair_img = batch[1][i]
        exchange_labels = batch[2][i].squeeze()

        ax[0, i].imshow(img.detach().cpu().numpy().squeeze(0), cmap='gray')
        ax[0, i].set_axis_off()
        ax[1, i].imshow(pair_img.detach().cpu().numpy().squeeze(0), cmap='gray')
        ax[1, i].set_axis_off()
        print(
            f'{i} pair has [{" ,".join([md.feature_names[idx] for idx, label in enumerate(exchange_labels) if label])}] feature(s) exchanged')

    plt.show()

    print("Done")

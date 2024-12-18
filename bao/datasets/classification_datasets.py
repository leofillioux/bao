import os
import h5py
import torch
import numpy as np
from .prepare import prepare_dataset
from torch.utils.data import Dataset
from ..configs import default_data_dir
from torchvision.transforms.v2 import RGB
from sklearn.preprocessing import OneHotEncoder
from torchvision.transforms import Resize, ToTensor, Compose

class ClassificationDataset(Dataset):
    def __init__(self,
                 dataset_name: str,
                 names: list[str],
                 mode: str,
                 img_size: tuple[int],
                 data_dir: str = default_data_dir):
        """Initialize the classification dataset.
        
            Args:
                dataset_name [str]: name of the dataset to prepare
                names [list]: list of image names
                mode [str]: mode of the dataset (train, test)
                img_size [tuple]: size of the images
                data_dir [str]: directory to save the prepared dataset to
        """
        self.dataset_path = os.path.join(data_dir, f'{dataset_name}.h5')
        self.names = names
        self.mode = mode
        self.preprocess_img = Compose([ToTensor(), Resize(img_size), RGB()])
        labels = [os.path.dirname(name) for name in self.names]
        if len(set(labels)) == 2:
            self.labels = np.array([0 if label == labels[0] else 1 for label in labels]).reshape(-1, 1)
        else:
            self.labels = OneHotEncoder(sparse=False).fit_transform(np.array(labels).reshape(-1, 1))
    
    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        with h5py.File(self.dataset_path, 'r') as hdf:
            img = np.array(hdf.get(os.path.join(self.mode, self.names[idx])))
        img = self.preprocess_img(img)
        label = torch.tensor(self.labels[idx]).float()
        return img, label

class DCryptOBudding(ClassificationDataset):
    def __init__(self,
                 names: list[str],
                 mode: str,
                 img_size: tuple[int],
                 download: bool,
                 download_dir: str = '/tmp',
                 data_dir: str = default_data_dir):
        """Prepare the D-CryptO budding dataset.

            Args:
                names [list]: list of image names
                mode [str]: mode of the dataset (train, test)
                img_size [tuple]: size of the images
                download [bool]: whether to download the dataset
                download_dir [str]: directory to download the dataset to
                data_dir [str]: directory to save the prepared
        """
        if download:
            prepare_dataset('dcrypto_budding', data_dir=data_dir, download_dir=download_dir)
        super().__init__('dcrypto_budding', names, mode, img_size, data_dir)

class DCryptOOpaque(ClassificationDataset):
    def __init__(self,
                 names: list[str],
                 mode: str,
                 img_size: tuple[int],
                 download: bool,
                 download_dir: str = '/tmp',
                 data_dir: str = default_data_dir):
        """Prepare the D-CryptO opaque dataset.

            Args:
                names [list]: list of image names
                mode [str]: mode of the dataset (train, test)
                img_size [tuple]: size of the images
                download [bool]: whether to download the dataset
                download_dir [str]: directory to download the dataset to
                data_dir [str]: directory to save the prepared
        """
        if download:
            prepare_dataset('dcrypto_opaque', data_dir=data_dir, download_dir=download_dir)
        super().__init__('dcrypto_opaque', names, mode, img_size, data_dir)
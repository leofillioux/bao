import os
import h5py
import torch
import numpy as np
from .prepare import prepare_dataset
from torch.utils.data import Dataset
from ..configs import default_data_dir
from torchvision.transforms.v2 import RGB
from torchvision.transforms import Resize, ToTensor, Compose

class DetectionDataset(Dataset):
    def __init__(self,
                 dataset_name: str,
                 names: list[str],
                 mode: str,
                 img_size: tuple[int],
                 predict_classes: bool,
                 data_dir: str = default_data_dir):
        """Initialize the detection dataset.
            Args:
                dataset_name [str]: name of the dataset to prepare
                names [list]: list of image names
                mode [str]: mode of the dataset (train, test)
                img_size [tuple]: size of the images
                predict_classes [bool]: whether to predict classes (False for localization, True for detection)
                data_dir [str]: directory to save the prepared dataset to
        """
        self.dataset_path = os.path.join(data_dir, f'{dataset_name}.h5')
        self.names = names
        self.mode = mode
        self.predict_classes = predict_classes
        self.preprocess_img = Compose([ToTensor(), Resize(img_size), RGB()])

    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        with h5py.File(self.dataset_path, 'r') as hdf:
            img = np.array(hdf.get(os.path.join(self.mode, 'images', self.names[idx])))
            lbl = np.array(hdf.get(os.path.join(self.mode, 'labels', self.names[idx])))
        pre_w, pre_h = img.shape[-2], img.shape[-1]
        img = self.preprocess_img(img)
        post_w, post_h = img.shape[-2], img.shape[-1]
        w_ratio = post_w / pre_w
        h_ratio = post_h / pre_h

        if self.predict_classes:
            label = {'boxes': torch.tensor(lbl[:, 1:]),
                     'labels': torch.tensor(lbl[:, 0]).long()}
        else:
            if len(lbl.shape) == 2:
                label = {'boxes': torch.tensor(lbl),
                         'labels': torch.ones(lbl[:, 0].shape).long()}
            else:
                label = {'boxes': torch.as_tensor(np.array(np.zeros((0, 4)), dtype=float)),
                         'labels': torch.as_tensor(np.array([], dtype=int), dtype=torch.int64)}

        size_augmentation = np.array([w_ratio, h_ratio, w_ratio, h_ratio]).reshape(1, -1)
        label['boxes'] *= size_augmentation

        return img, label

class Tellu(DetectionDataset):
    def __init__(self,
                 names: list[str],
                 mode: str,
                 img_size: tuple[int],
                 download: bool,
                 download_dir: str = '/tmp',
                 data_dir: str = default_data_dir):
        """Initialize the Tellu dataset (detection).

            Args:
                names [list]: list of image names
                mode [str]: mode of the dataset (train, test)
                img_size [tuple]: size of the images
                download [bool]: whether to download the dataset
                download_dir [str]: directory to download the dataset to
                data_dir [str]: directory to save the prepared
        """
        if download:
            prepare_dataset('tellu', data_dir, data_dir=data_dir, download_dir=download_dir)
        super().__init__('tellu', names, mode, img_size, predict_classes=True, data_dir=data_dir)
    
class OrgaQuant(DetectionDataset):
    def __init__(self,
                 names: list[str],
                 mode: str,
                 img_size: tuple[int],
                 download: bool,
                 download_dir: str = '/tmp',
                 data_dir: str = default_data_dir):
        """Initialize the OrgaQuant dataset (localization).

            Args:
                names [list]: list of image names
                mode [str]: mode of the dataset (train, test)
                img_size [tuple]: size of the images
                download [bool]: whether to download the dataset
                download_dir [str]: directory to download the dataset to
                data_dir [str]: directory to save the prepared
        """
        if download:
            prepare_dataset('orgaquant', data_dir=data_dir, download_dir=download_dir)
        super().__init__('orgaquant', names, mode, img_size, predict_classes=False, data_dir=data_dir)
    
class MultiOrg(DetectionDataset):
    def __init__(self,
                 names: list[str],
                 mode: str,
                 img_size: tuple[int],
                 download: bool,
                 download_dir: str = '/tmp',
                 data_dir: str = default_data_dir):
        """Initialize the MultiOrg dataset (localization).

            Args:
                names [list]: list of image names
                mode [str]: mode of the dataset (train, test)
                img_size [tuple]: size of the images
                download [bool]: whether to download the dataset
                download_dir [str]: directory to download the dataset to
                data_dir [str]: directory to save the prepared
        """
        if download:
            prepare_dataset('multiorg', data_dir=data_dir, download_dir=download_dir)
        super().__init__('multiorg', names, mode, img_size, predict_classes=False, data_dir=data_dir)
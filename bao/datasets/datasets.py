import os
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
from ..configs import default_data_dir
from torchvision.transforms.v2 import RGB
from sklearn.preprocessing import OneHotEncoder
from torchvision.transforms import Resize, ToTensor, Compose

def get_dataset(task: str,
                data_dir: str = default_data_dir,
                **kwargs) -> Dataset:
    """Get the dataset for the given task.

        Args:
            task [str]: task of the dataset (e.g., classification, segmentation)
            data_dir [str]: directory to the saved dataset
            kwargs: additional arguments to pass to the dataset
        
        Returns:
            dataset [torch.utils.data.Dataset]: dataset for the given task
    """
    if 'data_dir' in kwargs:
        data_dir = kwargs['data_dir']
        kwargs.pop('data_dir')

    if task == 'classification':
        dataset = ClassificationDataset
    elif task == 'segmentation':
        dataset = SegmentationDataset
    elif task == 'detection':
        dataset = DetectionDataset
    else:
        raise ValueError(f'Task {task} not supported.')
    return dataset(data_dir=data_dir, **kwargs)

class SegmentationDataset(Dataset):
    def __init__(self,
                 dataset_name: str,
                 data_dir: str,
                 names: list[str],
                 mode: str,
                 img_size: tuple[int]):
        """Initialize the segmentation dataset.
        
            Args:
                dataset_name [str]: name of the dataset to prepare
                data_dir [str]: directory to save the prepared dataset to
                names [list]: list of image names
                mode [str]: mode of the dataset (train, test)
                img_size [tuple]: size of the images
        """
        self.dataset_path = os.path.join(data_dir, f'{dataset_name}.h5')
        self.names = names
        self.mode = mode
        self.preprocess_img = Compose([ToTensor(), Resize(img_size), RGB()])
        self.preprocess_mask = Compose([Resize(img_size)])

    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        with h5py.File(self.dataset_path, 'r') as hdf:
            img = np.array(hdf.get(os.path.join(self.mode, 'images', self.names[idx])))
            mask = np.array(hdf.get(os.path.join(self.mode, 'masks', self.names[idx])))
        img = self.preprocess_img(img)
        mask = self.preprocess_mask(torch.tensor(mask).unsqueeze(0))
        return img, mask

class ClassificationDataset(Dataset):
    def __init__(self,
                 dataset_name: str,
                 data_dir: str,
                 names: list[str],
                 mode: str,
                 img_size: tuple[int]):
        """Initialize the classification dataset.
        
            Args:
                dataset_name [str]: name of the dataset to prepare
                data_dir [str]: directory to save the prepared dataset to
                names [list]: list of image names
                mode [str]: mode of the dataset (train, test)
                img_size [tuple]: size of the images
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

class DetectionDataset(Dataset):
    def __init__(self,
                 dataset_name: str,
                 data_dir: str,
                 names: list[str],
                 mode: str,
                 img_size: tuple[int]):
        """Initialize the detection dataset.
            Args:
                dataset_name [str]: name of the dataset to prepare
                data_dir [str]: directory to save the prepared dataset to
                names [list]: list of image names
                mode [str]: mode of the dataset (train, test)
                img_size [tuple]: size of the images
        """
        self.dataset_path = os.path.join(data_dir, f'{dataset_name}.h5')
        self.names = names
        self.mode = mode
        self.preprocess_img = Compose([ToTensor(), Resize(img_size), RGB()])

    def __len__(self):
        return len(self.names)
    
    def __getitem__(self, idx):
        with h5py.File(self.dataset_path, 'r') as hdf:
            img = np.array(hdf.get(os.path.join(self.mode, 'images', self.names[idx])))
            lbl = np.array(hdf.get(os.path.join(self.mode, 'labels', self.names[idx])))
        img = self.preprocess_img(img)
        label = {'boxes': torch.tensor(lbl[:, 1:]),
                'labels': torch.tensor(lbl[:, 0]).long()}
        return img, label
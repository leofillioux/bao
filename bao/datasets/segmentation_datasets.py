import os
import h5py
import torch
import numpy as np
from .prepare import prepare_dataset
from torch.utils.data import Dataset
from ..configs import default_data_dir
from torchvision.transforms.v2 import RGB
from torchvision.transforms import Resize, ToTensor, Compose

class SegmentationDataset(Dataset):
    def __init__(self,
                 dataset_name: str,
                 names: list[str],
                 mode: str,
                 img_size: tuple[int],
                 data_dir: str = default_data_dir):
        """Initialize the segmentation dataset.
        
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
    
class OrgaSegment(SegmentationDataset):
    def __init__(self,
                 names: list[str],
                 mode: str,
                 img_size: tuple[int],
                 download: bool,
                 download_dir: str = '/tmp',
                 data_dir: str = default_data_dir):
        """Initialize the OrgaSegment dataset.
            
            Args:
                names [list]: list of image names
                mode [str]: mode of the dataset (train, test)
                img_size [tuple]: size of the images
                download [bool]: whether to download the dataset
                download_dir [str]: directory to download the dataset to
                data_dir [str]: directory to save the prepared dataset to
        """
        if download:
            prepare_dataset('orgasegment', data_dir=data_dir, download_dir=download_dir)
        super().__init__('orgasegment', names, mode, img_size, data_dir)

class OrgaExtractor(SegmentationDataset):
    def __init__(self,
                 names: list[str],
                 mode: str,
                 img_size: tuple[int],
                 download: bool,
                 download_dir: str = '/tmp',
                 data_dir: str = default_data_dir):
        """Initialize the OrgaExtractor dataset.
                
                Args:
                    names [list]: list of image names
                    mode [str]: mode of the dataset (train, test)
                    img_size [tuple]: size of the images
                    download [bool]: whether to download the dataset
                    download_dir [str]: directory to download the dataset to
                    data_dir [str]: directory to save the prepared dataset to
        """
        if download:
            prepare_dataset('orgaextractor', data_dir=data_dir, download_dir=download_dir)
        super().__init__('orgaextractor', names, mode, img_size, data_dir)
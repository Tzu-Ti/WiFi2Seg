import torch
from torchvision import transforms
from torch.utils.data import Dataset
from torchvision.transforms.functional import InterpolationMode

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

import json
import numpy as np
from PIL import Image
import os
import einops

def get_data_list(json_path: str, mode: str = None):
    """
    Get the data list from the json file
    :param json_path: the file contain data path
    :param mode: dataset mode, ["train", "val", "test"]

    Return:
    data_list: the data list of the CSI
        ["/root/bindingvolume/CSI_UNCC/parsed/csi0/test_set/M2/posi/241103_170934/1730671877851838316.npz",
        "/root/bindingvolume/CSI_UNCC/parsed/csi0/test_set/M2/posi/241103_170934/1730671867110087557.npz", ...]
    """
    with open(json_path, 'r') as f:
        data_list = json.load(f)

    if mode is None:
        # whole dataset
        return data_list['data']
    else:
        # split dataset
        assert mode in ['train', 'val', 'test'], "mode should be one of ['train', 'val', 'test']"
        return data_list[mode]

class BaseDataset(Dataset):
    def __init__(self, 
                 json_path: str, data_root: str, mode: str = None):
        """
        Base dataset
        :param json_path: the file contain data path
        :param mode: dataset mode, ["train", "val", "test"]
        """
        self.mode = mode

        # Get the data list
        self.data_list = get_data_list(json_path, mode)
        self.data_list = [os.path.join(data_root, data) for data in self.data_list]

    def _normalize(self, x: torch.Tensor, mean: float = 0, std: float = 0.5):
        """
        Normalize the CSI data
        :param x: data, amplitude or phase
        :param mean: mean of target distribution
        :param std: standard deviation of target distribution

        Return:
        normalized data
        """
        return ((x - x.mean()) / x.std()) * std + mean
    
    def __len__(self):
        return len(self.data_list)

class MaskDataset(BaseDataset):
    def __init__(self,
                 json_path: str, data_root: str, mode: str = None,
                 size: tuple = (192, 256)):
        """
        CSI2Mask dataset
        :param json_path: the file contain data path
        :param mode: dataset mode, ["train", "val", "test"]
        :param size: the size of mask
        """
        super().__init__(json_path=json_path, data_root=data_root, mode=mode)
        self.mode = mode

        # Get the data list
        self.data_list = get_data_list(json_path, mode)
        self.data_list = [os.path.join(data_root, data) for data in self.data_list]

        # For transform mask data
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size, interpolation=InterpolationMode.NEAREST),
        ])

    def _get_mask(self, mask_path):
        """
        Get the mask data from the png file
        :param mask_path: the path of the png file

        Return:
        mask: the mask data
        """
        mask = Image.open(mask_path).convert('L')
        mask = self.transform(mask).float()

        return mask

    def __getitem__(self, index):
        csi_path = self.data_list[index]
        
        mask_path = csi_path.replace('csi0', 'rgb').replace('.npz', '_mask.png')
        mask = self._get_mask(mask_path)

        return mask

class CSIDataset(BaseDataset):
    def __init__(self,
                 json_path: str, data_root: str, mode: str = None,
                 major: str = 'csi0', receivers: list = ['csi0', 'csi1', 'csi2']):
        """
        CSI dataset
        :param json_path: the file contain data path
        :param mode: dataset mode, ["train", "val", "test"]
        :param major: the major receiver
        :param receivers: the list of all receivers
        """
        super().__init__(json_path=json_path, data_root=data_root, mode=mode)
        
        self.mode = mode
        self.major = major
        self.receivers = receivers

    def _read_csi(self, csi_path: str):
        """
        Read the CSI data from the npz file
        :param csi_path: the path of the npz file

        Return:
        csi: the CSI data
        """
        csi = np.load(csi_path)

        # parse the data to amplitude and phase
        amp = csi['mag']
        amp = torch.from_numpy(amp)
        amp = self._normalize(amp)
        amp = einops.rearrange(amp, 'l rx sc -> (l sc) rx')
        amp = amp.unsqueeze(1)

        pha = csi['phase']
        pha = torch.from_numpy(pha)
        pha = self._normalize(pha)
        pha = einops.rearrange(pha, 'l rx sc -> (l sc) rx')
        pha = pha.unsqueeze(1)

        return amp, pha

    def _get_all_csi(self, major_csi_path: str):
        """
        Get the CSI data from all receivers npz file
        :param major_csi_path: the path of the major npz file

        Return:
        csi: the CSI data
        """
        # read the major csi data
        amp, pha = self._read_csi(major_csi_path)

        # read the other csi data and concatenate
        for r in self.receivers:
            if r == self.major:
                continue
            csi_path = major_csi_path.replace(self.major, r)
            if not os.path.exists(csi_path):
                raise FileNotFoundError(f"{csi_path} not found")
            r_amp, r_pha = self._read_csi(csi_path)
            amp = torch.cat((amp, r_amp), dim=1)
            pha = torch.cat((pha, r_pha), dim=1)

        return amp, pha

    def __getitem__(self, index):
        major_csi_path = self.data_list[index]

        amp, pha = self._get_all_csi(major_csi_path)

        return amp, pha

class CSI2Mask_Dataset(CSIDataset, MaskDataset):
    def __init__(self,
                 json_path: str, data_root: str, mode: str = None,
                 size: tuple = (192, 256),
                 major: str = 'csi0', receivers: list = ['csi0', 'csi1', 'csi2']):
        """
        CSI2Mask Dataset
        :param json_path: the file contain data path
        :param data_root: 
        :param mode: dataset mode, ["train", None]
        :param size: the size of mask
        :param major:
        :param receivers: 
        """
        MaskDataset.__init__(self, json_path=json_path, data_root=data_root, mode=mode, size=size)
        CSIDataset.__init__(self, json_path=json_path, data_root=data_root, mode=mode, major=major, receivers=receivers)

        self.mode = mode

    def __getitem__(self, index):
        major_csi_path = self.data_list[index]
        amp, pha = self._get_all_csi(major_csi_path)

        # mask
        mask_path = major_csi_path.replace(self.major, 'mask').replace('.npz', '_mask.png')
        mask = self._get_mask(mask_path)

        return amp, pha, mask

class CSI2MaskDataModule(LightningDataModule):
    def __init__(self, dataset_class, configs):
        super().__init__()

        self.dataset_class = dataset_class

        dataset_config = configs['Dataset']
        self.data_root = dataset_config['data_root']
        self.train_json_path = dataset_config['train&val_json_path']
        self.val_json_path = dataset_config['val_json_path']

        test_dataset_config = configs['TestDataset']
        self.test_data_root = test_dataset_config['test_data_root']
        self.test_json_path = test_dataset_config['test_json_path']
        self.test_mode = test_dataset_config['mode']

        self.batch_size = dataset_config['batch_size']
        self.num_workers = dataset_config['num_workers']

    def setup(self, stage):
        if stage == 'fit':
            self.train_dataset = self.dataset_class(
                json_path=self.train_json_path,
                data_root=self.data_root, 
                mode='train'
            )
            self.val_dataset_seen = self.dataset_class(
                json_path=self.train_json_path,
                data_root=self.data_root,
                mode='val'
            )
            self.val_dataset_unseen = self.dataset_class(
                json_path=self.val_json_path,
                data_root=self.data_root,
            )
        elif stage == 'test':
            self.test_dataset = self.dataset_class(
                json_path=self.test_json_path,
                data_root=self.test_data_root,
                mode=self.test_mode
            )

    def dataloader(self, dataset, shuffle):
        return DataLoader(dataset,
                          batch_size=self.batch_size, 
                          shuffle=shuffle, 
                          num_workers=self.num_workers)

    def train_dataloader(self):
        dataloader = self.dataloader(self.train_dataset, shuffle=True)
        return dataloader

    def val_dataloader(self):
        dataloader_seen = self.dataloader(self.val_dataset_seen, shuffle=False)
        dataloader_unseen = self.dataloader(self.val_dataset_unseen, shuffle=False)
        return [dataloader_seen, dataloader_unseen]

    def test_dataloader(self):
        dataloader = self.dataloader(self.test_dataset, shuffle=False)
        return dataloader

if __name__ == '__main__':
    json_path = '/root/workspace/WiFi2Seg/datas/UNCC/train&val.json'
    dataset = CSI2Mask_Dataset(json_path=json_path, data_root='/root/bindingvolume/CSI_UNCC/parsed', mode='train')
    for i in range(len(dataset)):
        amp, pha, mask = dataset[i]
        # print(csi.max(), csi.min(), csi.mean(), csi.std())
        print(amp.shape)
        print(pha.shape)
        print(mask.shape)
        # [amp1, pha1, mask], _, _ = dataset[i]
        break
        
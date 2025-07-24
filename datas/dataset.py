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

def get_data_list(json_path: str, mode: str):
    """
    Get the data list from the json file
    :param json_path: the file contain data path
    :param mode: dataset mode, ["train", "val", "test"]

    Return:
    data_list: the data list of the CSI
        ["/root/SSD/PiWiFi/NYCU/Env4/npy/F3/5_posi/240508_152018/1715152876595249214.npz",
        "/root/SSD/PiWiFi/NYCU/Env3/npy/M3/5_posi/240508_105725/1715137086843263338.npz", ...]
        ["/root/bindingvolume/CSI_dataset_UNCC/Env2/npy/F1/1_posi/241102_134944/1730569892598354411.npz",
        "/root/bindingvolume/CSI_dataset_UNCC/Env1/npy/F2/1_posi/241102_144833/1730573381748239710.npz", ...]
    """
    with open(json_path, 'r') as f:
        data_list = json.load(f)

    return data_list[mode]

class MaskDataset(Dataset):
    def __init__(self,
                 json_path: str, data_root: str, mode: str = 'train',
                 size: tuple = (192, 256)):
        """
        CSI2Mask dataset
        :param json_path: the file contain data path
        :param mode: dataset mode, ["train", "val", "test"]
        :param size: the size of mask
        """
        self.mode = mode

        # Get the data list
        self.data_list = get_data_list(json_path, mode)
        self.data_list = [os.path.join(data_root, data) for data in self.data_list]

        # For transform mask data
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(size, interpolation=InterpolationMode.NEAREST),
        ])

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
        
        mask_path = csi_path.replace('npy', 'img').replace('.npz', '_mask.png')
        mask = self._get_mask(mask_path)

        return mask

    def __len__(self):
        return len(self.data_list)

class CSI2MaskDataset(MaskDataset):
    def __init__(self,
                 json_path: str, data_root: str, mode: str = 'train',
                 size: tuple = (192, 256),
                 amp_offset: float = 60000, pha_offset: float = 28000):
        """
        CSI2Mask dataset
        :param json_path: the file contain data path
        :param mode: dataset mode, ["train", "val", "test"]
        :param size: the size of mask
        :param amp_offset: the offset of amplitude data
        :param pha_offset: the offset of phase data
        """
        super().__init__(json_path=json_path, data_root=data_root, mode=mode, size=size)
        
        self.mode = mode
        self.amp_offset = amp_offset
        self.pha_offset = pha_offset

        # Accroding to the Env, classify the data into different Env
        self.env_dict = {}
        for data in self.data_list:
            env = self._get_env(data)
            if env not in self.env_dict:
                self.env_dict[env] = []
            self.env_dict[env].append(data)

    def _get_env(self, csi_path: str):
        """
        Get the number of environment of the CSI data
        :param csi_path: the path of the npz file

        Return:
        env: the number of environment
        """
        parts = csi_path.split('/')
        for part in parts:
            if 'Env' in part or '_set' in part:
                env = part
        return env

    def _get_csi(self, csi_path: str):
        """
        Get the CSI data from the npz file
        :param csi_path: the path of the npz file

        Return:
        csi: the CSI data
        """
        csi = np.load(csi_path)

        # parse the data to amplitude and phase
        amp = csi['mag'].astype(np.float32) / self.amp_offset
        amp = torch.from_numpy(amp)
        amp = self._normalize(amp)

        pha = csi['pha'].astype(np.float32) / self.pha_offset
        pha = torch.from_numpy(pha)
        pha = self._normalize(pha)
        
        return amp, pha

    def __getitem__(self, index):
        csi_path = self.data_list[index]
        env = self._get_env(csi_path)
        amp, pha = self._get_csi(csi_path)
        
        # choose another data in the same or different env
        if self.mode == 'train':
            if np.random.rand() > 0.5:
                another_env = env
                label = 1
            else:
                keys = list(self.env_dict.keys())
                keys.remove(env)
                another_env = np.random.choice(keys)
                label = -1
        
            another_csi_path = np.random.choice(self.env_dict[another_env])
            another_amp, another_pha = self._get_csi(another_csi_path)

        # mask
        mask_path = csi_path.replace('npy', 'img').replace('.npz', '_mask.png')
        rgb_path = csi_path.replace('npy', 'img').replace('.npz', '.jpg')
        mask = self._get_mask(mask_path)

        if self.mode == 'train':
            return [amp, pha, mask], [another_amp, another_pha], torch.tensor(label)
        else:
            return amp, pha, mask

    def __len__(self):
        return len(self.data_list)

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
                mode='val'
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
    json_path = '/root/workspace/CoWIP/datas/NYCU/train&val.json'
    dataset = CSI2MaskDataset(json_path=json_path, data_root='/root/bindingvolume/CSI_dataset_NYCU', mode='train')
    for i in range(len(dataset)):
        [amp1, pha1, mask], [amp2, pha2], label = dataset[i]
        print(amp1.shape)
        print(pha1.shape)
        print(mask.shape)
        print(amp2.shape)
        print(pha2.shape)
        # [amp1, pha1, mask], _, _ = dataset[i]
        break
        
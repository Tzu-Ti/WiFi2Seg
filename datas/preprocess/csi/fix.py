from tqdm import tqdm
import numpy as np
import os
import glob

data_root = "/root/bindingvolume/CSI_UNCC/parsed"
for path in tqdm(glob.glob(os.path.join(data_root, "csi[0-2]", "env5", "*", "*", "*", "*.npz"))):
    csi = np.load(path)
    mag = csi['mag']
    phase = csi['phase']
    if mag.shape[2] == 2025:
        first = mag[:, :, :1001]
        second = mag[:, :, 1024:1997]
        mag = np.concatenate([first, second], axis=2)
        
    np.savez_compressed(
        path,
        mag=mag,
        phase=phase
    )

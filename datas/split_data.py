import argparse
import glob
import os
import random
import json
from tqdm import tqdm

def parse_args():
    """
    example
    :param data_root: /root/SSD/PiWiFi/NYCU
    :param person_number: 1
    :param mode: train (or val, test. train -> train&val.json, val -> val.json, test -> test.json)
    """
    parser = argparse.ArgumentParser(description="Split data script")
    parser.add_argument('--raw_data_root', required=True, help='Root directory of the raw data')
    parser.add_argument('--glob_path', required=True, help='Glob path to find the data')
    parser.add_argument('--mode', required=True, choices=['split', 'whole'], help='split or whole dataset')
    parser.add_argument('--name', required=True, help='Name of the dataset')
    parser.add_argument('--ratio', default="8:2", help='Train/Val split ratio')
    return parser.parse_args()

def main():
    args = parse_args()
    print(f"Data root directory: {args.raw_data_root}")
    print(f"Glob path: {args.glob_path}")
    print(f"Mode: {args.mode}")
    if args.mode == 'split':
        print(f"Split ratio: {args.ratio}")

    dataset_name = f"{args.name}.json"
    print(f"Dataset name: {dataset_name}")

    # get all file path
    all_csi_paths = []
    rgb_paths = glob.glob(os.path.join(args.raw_data_root, 'rgb', args.glob_path, '*.jpg'))
    print(f"Total number of RGB files found: {len(rgb_paths)}")

    # check if CSI corresponding npy file exists
    for rgb_path in tqdm(rgb_paths):
        csi0_path = rgb_path.replace('raw', 'parsed').replace('rgb', 'csi0').replace('.jpg', '.npz')
        csi1_path = csi0_path.replace('csi0', 'csi1')
        csi2_path = csi0_path.replace('csi0', 'csi2')
        if os.path.exists(csi0_path) and os.path.exists(csi1_path) and os.path.exists(csi2_path):
            all_csi_paths.append(csi0_path)  # just need one of the csi files
    length = len(all_csi_paths)
    print(f"Total number of complete CSI sets found: {length}")

    if args.mode == 'split':
        ratio0, ratio1 = map(int, args.ratio.split(':'))
        assert ratio0 + ratio1 == 10, "The sum of the split ratio must be 10"
        split_idx = int(length * ratio0 / (ratio0 + ratio1))

        # shuffle and save to json
        random.shuffle(all_csi_paths)
        datas = {
            'train': all_csi_paths[:split_idx],
            'val': all_csi_paths[split_idx:]
        }
        print(f"Number of training data: {len(datas['train'])}")
        print(f"Number of validation data: {len(datas['val'])}")
    elif args.mode == 'whole':
        datas = {
            'data': all_csi_paths
        }
        print(f"Number of data: {len(datas['data'])}")

    with open(dataset_name, 'w') as f:
        json.dump(datas, f)

if __name__ == "__main__":
    main()
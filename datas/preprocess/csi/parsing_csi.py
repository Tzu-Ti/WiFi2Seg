from picoscenes import Picoscenes
import argparse
import glob, os
import numpy as np
from multiprocessing.pool import ThreadPool
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

def parser():
    parser = argparse.ArgumentParser(description="Parser for CSI dataset preprocessing")
    # data path
    parser.add_argument("--dataset_root", required=True, help="Path to the root directory of the CSI dataset")
    parser.add_argument("--glob_path", required=True, help="Glob path to find the CSI files")
    # data settings
    parser.add_argument("--csi_subcarrier", type=int, default=2025, help="Number of subcarriers in CSI data")
    parser.add_argument("--csi_length", type=int, default=25, help="Length of CSI data")
    # multi processing
    parser.add_argument("--num_workers", type=int, default=1, help="Number of workers for multiprocessing")
    return parser.parse_args()

def bin_search(arr, target):
    # return target index or the index of closest value to target
    left, right = 0, len(arr) - 1
    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    left = min(len(arr) - 1, max(0, left))
    right = max(0, min(len(arr) - 1, right))
    try:
        return right if abs(arr[right] - target) < abs(arr[left] - target) else left
    except:
        print(f"len(arr):{len(arr)}, target:{target}, left:{left}, right:{right}")
        exit()

def parse_csi(rx_file, csi_subcarrier, csi_length):
    # read csi data
    print(f"Reading  {rx_file}...")
    rx_csi = Picoscenes(rx_file)
    print("CSI data have been read")

    # parsing csi
    # print("Parsing CSI data...")
    rx_frames_dict = {}
    for rx_frames in tqdm(rx_csi.raw, desc="parsing CSI"):
        if len(rx_frames.get('CSI').get('SubcarrierIndex')) != csi_subcarrier:
            raise ValueError(f"Expected rx have {csi_subcarrier} subcarriers, but got {len(rx_frames.get('CSI').get('SubcarrierIndex'))}")
        # store the magnitude and phase of each csi frame, key is the timestamp
        rx_frames_dict[rx_frames.get('RxSBasic').get('systemns')] = {
            'Mag': np.array(rx_frames.get('CSI').get('Mag')),
            'Phase': np.array(rx_frames.get('CSI').get('Phase')),
        }
    # print("CSI data have been parsed")
    return rx_frames_dict

def linear_fitting(data):
    for i in range(len(data)):
        for j in range(len(data[i])):
            noise_0 = np.poly1d(np.polyfit(np.arange(1001), data[i, j, :1001], 1))(np.arange(1001))
            noise_tail = np.poly1d(np.polyfit(np.arange(1001, 1974), data[i, j, 1001:], 1))(np.arange(1001, 1974))
            noise = np.concatenate([noise_0, noise_tail], axis=0)
            data[i, j, :] = data[i, j, :] - noise
    return data

def save_npz(save_path, frame_mag, frame_phase):
    # 確保記憶體連續，避免背景 thread 複製開銷
    frame_mag   = np.ascontiguousarray(frame_mag, dtype=np.float32)
    frame_phase = np.ascontiguousarray(frame_phase, dtype=np.float32)

    if None in frame_phase:
        raise ValueError(f"Error: None in phase data, skip saving {save_path}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(
        save_path,
        mag=frame_mag,
        phase=frame_phase
    )

def register_csi_image(rx_files, rx_name, image_folders, csi_subcarrier, csi_length, num_workers):
    for rx_file, img_folder in tqdm(zip(rx_files, image_folders), total=len(rx_files), desc=f"Processing {rx_name}"):
        rx_frames_dict = parse_csi(rx_file, csi_subcarrier, csi_length)

        img_paths = sorted(glob.glob(os.path.join(img_folder, "*.jpg")))
        # print(len(img_paths), "images found in", img_folder)

        # the timestamp of each csi frames in one .csi
        rx_timestamps = list(rx_frames_dict.keys())

        finded = 0
        # parsing image
        for img_path in tqdm(img_paths, total=len(img_paths), desc="register image and CSI"):
            img_name = os.path.basename(img_path) # get img name
            img_name = os.path.splitext(img_name)[0] # without extension
            timestamp = int(img_name)

            # find the closest csi frame to the image timestamp
            center_ts_index = bin_search(rx_timestamps, timestamp)

            # check there are enough csi frames around the image timestamp
            if center_ts_index > csi_length and center_ts_index < len(rx_timestamps) - csi_length:
                # get the csi frames around the image timestamp
                frames_ts = rx_timestamps[center_ts_index - csi_length:center_ts_index + csi_length + 1]
            else: frames_ts = None
            
            executor = ThreadPoolExecutor(max_workers=num_workers)
            futures = []
            if frames_ts:
                finded += 1
                frame_mag = None
                frame_phase = None
                for ts in frames_ts:
                    # get the csi data of the current timestamp
                    data = rx_frames_dict[ts]
                    # split 2 antennas data
                    mag = data['Mag']
                    mags = np.concatenate([np.expand_dims(mag[:csi_subcarrier], axis=0),
                                           np.expand_dims(mag[csi_subcarrier:], axis=0)], axis=0)
                    phase = data['Phase']
                    phases = np.concatenate([np.expand_dims(phase[:csi_subcarrier], axis=0),
                                             np.expand_dims(phase[csi_subcarrier:], axis=0)], axis=0)
                    if frame_mag is None:
                        frame_mag = np.expand_dims(mags, axis=0)
                        frame_phase = np.expand_dims(phases, axis=0)
                    else:
                        frame_mag = np.concatenate([frame_mag, np.expand_dims(mags, axis=0)], axis=0)
                        frame_phase = np.concatenate([frame_phase, np.expand_dims(phases, axis=0)], axis=0)
                
                # remove pilot and guard
                first_frame_phase = frame_phase[:, :, :1001]
                second_frame_phase = frame_phase[:, :, 1024:1997]
                frame_phase = np.concatenate((first_frame_phase, second_frame_phase), axis=2)
                # phase, linear fitting to remove noise
                frame_phase = linear_fitting(frame_phase)
                # convert to float32
                frame_mag = frame_mag.astype(np.float32)
                frame_phase = frame_phase.astype(np.float32)
                # save the csi data and image data
                save_path = img_path.replace('raw', 'parsed')
                save_path = save_path.replace('rgb', rx_name)
                save_path = save_path.replace('.jpg', '.npz')
                # use thread to save data
                futures.append(executor.submit(save_npz, save_path, frame_mag, frame_phase))

            # 等所有檔案都寫完
            for f in as_completed(futures):
                f.result()  # 如需錯誤處理可包 try/except
            executor.shutdown(wait=True)

        # print(f"Found {finded} images with enough csi frames.")
        if finded == 0:
            with open('problem data.txt', 'a') as f:
                f.write(f"{img_folder}\n")
        # print("-------------------------------------")

def main():
    args = parser()

    # glob csi paths
    rx0_files = sorted(glob.glob(os.path.join(args.dataset_root, "csi0", args.glob_path, "*", "*.csi")))
    rx1_files = sorted(glob.glob(os.path.join(args.dataset_root, "csi1", args.glob_path, "*", "*.csi")))
    rx2_files = sorted(glob.glob(os.path.join(args.dataset_root, "csi2", args.glob_path, "*", "*.csi")))
    print(f"Found {len(rx0_files)} CSI files in csi0")
    print(f"Found {len(rx1_files)} CSI files in csi1")
    print(f"Found {len(rx2_files)} CSI files in csi2")

    # glob image paths
    image_folders = sorted(glob.glob(os.path.join(args.dataset_root, "rgb", args.glob_path, "*")))

    print(f"Found {len(image_folders)} image files in rgb")

    # check data length
    if not (len(rx0_files) == len(rx1_files) == len(rx2_files) == len(image_folders)):
        raise ValueError("The number of CSI files and image files do not match.")

    register_csi_image(rx0_files, 'csi0', image_folders, args.csi_subcarrier, args.csi_length, args.num_workers)
    register_csi_image(rx1_files, 'csi1', image_folders, args.csi_subcarrier, args.csi_length, args.num_workers)
    register_csi_image(rx2_files, 'csi2', image_folders, args.csi_subcarrier, args.csi_length, args.num_workers)
    

    

        
if __name__ == "__main__":
    main()

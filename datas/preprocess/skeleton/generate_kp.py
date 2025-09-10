# Description: Generate skeleton from image

import argparse
import pyopenpose as op
import glob, os
import cv2
import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
# OpenPose folder
parser.add_argument("--model_folder", help="Path to OpenPose models", default="openpose/models")
# data setting
parser.add_argument("--dataset_root", required=True, help="Path to the root directory of the dataset")
parser.add_argument("--glob_path", required=True, help="Glob path to find the rgb files")
args = parser.parse_args()

# OpenPose parameters
params = dict()
params["model_folder"] = args.model_folder
params["heatmaps_add_parts"] = True
params["heatmaps_add_bkg"] = True
params["heatmaps_add_PAFs"] = True
params["heatmaps_scale"] = 2
params["net_resolution"] = "256x192"

# Starting OpenPose
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

# parse image
all_images = glob.glob(os.path.join(args.dataset_root, 'rgb', args.glob_path, '*', '*.jpg'))

for image_path in tqdm(all_images):
    datum = op.Datum()
    imageToProcess = cv2.imread(image_path)
    imageToProcess = cv2.resize(imageToProcess, (256, 192))

    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    keypoints = datum.poseKeypoints
    heatmaps = datum.poseHeatMaps.copy()
    heatmaps = (heatmaps).astype(dtype='uint8')

    save_path = image_path.replace('.jpg', '.npy')
    save_path = save_path.replace('rgb', '2d_pose')
    save_path = save_path.replace('raw', 'parsed')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.save(save_path, heatmaps)

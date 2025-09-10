from mmseg.apis import init_model, inference_model
import glob, os
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

def parser():
    parser = argparse.ArgumentParser(description="Parser for Mask2Former preprocessing")
    # data path
    parser.add_argument("--dataset_root", required=True, help="Path to the root directory of the CSI dataset")
    parser.add_argument("--glob_path", required=True, help="Glob path to find the RGB files")
    # data settings
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for mask generation")
    # inference settings
    parser.add_argument("--batch_size", type=int, default=12, help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=8, help="Number of workers for data loading")
    # model path
    parser.add_argument("--config_path", help="Path to the model configuration file",
                        default="/root/workspace/mmsegmentation/configs/mask2former/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640.py")
    parser.add_argument("--ckpt_path", help="Path to the model checkpoint file",
                        default="/root/workspace/mmsegmentation/ckpts/mask2former_swin-l-in22k-384x384-pre_8xb2-160k_ade20k-640x640_20221203_235933-7120c214.pth")

    return parser.parse_args()

def normalize(x):
    return (x - x.min()) / (x.max() - x.min())

def f1(x, r):
    # gamma correction
    return 255 * (x / 255) ** r

def f2(x, s, t):
    # quadratic transformation
    return ((t - s) / (s ** 2 - 255 * s)) * x ** 2 + (1 - (255 * (t - s)) / (s ** 2 - 255 * s)) * x

def preprocess(img):
    # preprocess the image
    r, s, t = 0.385, 0.567, 0.265
    ratio = 0.65
    img = (1 - ratio) * f1(img, r) + ratio * f2(img, s, t)
    img = img / img.max()
    img = (img - 0.5) * 1.15 + 0.5
    img = np.clip(img, 0, 1) * 255
    img = img.astype(np.uint8)
    return img

def usm_sharpness(img):
    # apply unsharp masking
    blur_img = cv2.GaussianBlur(img, (0, 0), 3)
    img = cv2.addWeighted(img, 1.1, blur_img, -0.1, 0)
    return img

def read_and_preprocess(img_path):
    # read and preprocess the image
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to read image: {img_path}")
    img = preprocess(img)
    img = usm_sharpness(img)
    return img, img_path

def inference(model, imgs, paths, threshold):
    # inference the model
    results = inference_model(model, imgs)

    for result, img_path in zip(results, paths):
        mask = result.seg_logits.data
        
        person_mask = normalize(mask[12])
        person_mask[person_mask < threshold] = 0
        person_mask[person_mask >= threshold] = 1

        gt = person_mask.cpu().numpy() * 255
        gt = gt.astype(np.uint8)

        # Save the mask
        mask_path = img_path.replace("raw", "parsed")
        mask_path = mask_path.replace(".jpg", ".png")
        mask_folder = os.path.dirname(mask_path)
        os.makedirs(mask_folder, exist_ok=True)
        cv2.imwrite(mask_path, gt)

def main():
    args = parser()
    # Configure model
    print("> Initialize Model...")
    model = init_model(args.config_path, args.ckpt_path)

    # get data paths
    img_data = glob.glob(os.path.join(args.dataset_root, args.glob_path, "*.jpg"))
    if not img_data:
        print("No images found. Please check the glob path.")
        return

    # Start Preprocess
    print("> Preprocessing...")
    batch_image = []
    batch_path = []
    with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
        futures = [executor.submit(read_and_preprocess, img_path) for img_path in tqdm(img_data, desc="Reading and Preprocessing Images")]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing Images"):
            try:
                img, img_path = f.result()
            except Exception as e:
                print(f"Error processing image: {e}")
            batch_image.append(img)
            batch_path.append(img_path)
            if len(batch_image) >= args.batch_size:
                # Inference
                inference(model, batch_image, batch_path, args.threshold)
                batch_image = []
                batch_path = []


if __name__ == '__main__':
    main()
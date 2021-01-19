import numpy as np
import argparse
import trimesh
import os
import glob
import cv2
from tqdm import tqdm
from shutil import copyfile
from tools import handle_custom_dataset

parser = argparse.ArgumentParser(description='Grasp mining')
parser.add_argument('-m', '--model', type=str, required=True, help='Path to the .ply model file')
parser.add_argument('-d', '--dataset', type=str, required=True, help="Path to the dataset file (e.g. dataset/train.txt)")
parser.add_argument('-o', '--output', type=str, required=True, help="Output directory for formatted dataset")
parser.add_argument('-n', '--name', type=str, default="custom", help="Dataset name")

def distance(point_one, point_two):
    return ((point_one[0] - point_two[0]) ** 2 +
            (point_one[1] - point_two[1]) ** 2 + (point_one[2] - point_two[2]) ** 2) ** 0.5

def max_distance(mesh):
    vertices = mesh.vertices.tolist()
    return max(distance(p1, p2) for p1, p2 in zip(vertices, vertices[1:]))

def main(args):
    if not os.path.isfile(args.dataset):
        print(f"Invalid dataset file: {args.dataset}")
        exit(1)
    if not (os.path.isfile(args.model) and os.path.splitext(args.model)[1] == ".ply"):
        print(f"Invalid model path: {args.model}")
        exit(1)
    if not os.path.isdir(args.output):
        print(f"Invalid output path: {args.output}")
        exit(1)

    # Create output directory structure
    output_root = os.path.join(args.output, args.name)
    print(f"Storing formatted dataset in {output_root}")
    os.mkdir(output_root)
    os.mkdir(os.path.join(output_root, "rgb"))
    os.mkdir(os.path.join(output_root, "mask"))
    os.mkdir(os.path.join(output_root, "pose"))

    # Get source dataset root directory
    dataset_root = os.path.dirname(args.dataset)

    # Extract object diameter
    model = trimesh.load(args.model)
    diameter = max_distance(model)
    with open(os.path.join(output_root, "diameter.txt"), "w") as f:
        f.write(str(diameter))

    # Copy model file
    copyfile(args.model, os.path.join(output_root, "model.ply"))

    idx = 0
    calib = None
    sample_names = []
    with open(args.dataset, "r") as f:
        sample_names = f.read().splitlines()

    for sample_name in tqdm(sample_names):
        # Check meta file
        metaFile = os.path.join(dataset_root, f"meta/{sample_name}.pkl")
        if not os.path.isfile(metaFile):
            print(f"Could not find meta file for: {sample_name}\nLooked for {metaFile}")
            continue
        meta = np.load(metaFile, allow_pickle=True)

        # Check RGB image
        rgbFile = os.path.join(dataset_root, f"rgb/{sample_name}.jpg")
        if not os.path.isfile(rgbFile):
            print(f"Could not find rgb file for: {sample_name}\nLooked for {rgbFile}")
            continue

        # Read object mask
        maskFile = os.path.join(dataset_root, f"segm/{sample_name}.png")
        if not os.path.isfile(maskFile):
            print(f"Could not find mask file for: {sample_name}\nLooked for {maskFile}")
        else:
            src = cv2.imread(maskFile, cv2.IMREAD_UNCHANGED)
            mask = src[:, :, 2]
            if np.all(mask == 0):
                print(f"Found empty mask file: {maskFile}")
                continue

        # Write (or check) camera parameters
        if calib is None:
            calib = meta['cam_calib']
            np.savetxt(os.path.join(output_root, "camera.txt"), calib)
        elif np.any(calib != meta['cam_calib']):
            print("Camera calibration matrix changed within dataset, which is currently not supported! Skipping file.")
            print(f"Calib: {calib}")
            print(f"Calib2: {meta['cam_calib']}")
            continue

        # Copy RGB images
        copyfile(rgbFile, os.path.join(output_root, f"rgb/{idx}.jpg"))
        # Save object mask
        cv2.imwrite(os.path.join(output_root, f"mask/{idx}.png"), mask)
        # Save pose information
        pose = meta['cam_extr'] @ meta['affine_transform']
        np.save(os.path.join(output_root, f"pose/pose{idx}.npy"), pose)

        idx = idx + 1

    print(f"Copied {idx} samples to new dataset.")

    handle_custom_dataset.sample_fps_points(output_root)
    handle_custom_dataset.custom_to_coco(output_root)

    print("Done!")


if __name__ == '__main__':
    main(parser.parse_args())

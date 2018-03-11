import argparse
import glob
import json
import numpy as np
import os

from pose import Pose, Part, PoseSequence
from pprint import pprint


def main():

    parser = argparse.ArgumentParser(description='Pose Trainer Parser')
    parser.add_argument('--input_folder', type=str, default='poses', help='input folder for json files')
    parser.add_argument('--output_folder', type=str, default='poses_compressed', help='output folder for npy files')
    
    args = parser.parse_args()

    video_paths = glob.glob(os.path.join(args.input_folder, '*'))
    video_paths = sorted(video_paths)

    # Get all the json sequences for each video
    all_ps = []
    for video_path in video_paths:
        all_ps.append(parse_sequence(video_path, args.output_folder))
    return video_paths, all_ps


def parse_sequence(json_folder, output_folder):
    """Parse a sequence of OpenPose JSON frames and saves a corresponding numpy file.

    Args:
        json_folder: path to the folder containing OpenPose JSON for one video.
        output_folder: path to save the numpy array files of keypoints.

    """
    json_files = glob.glob(os.path.join(json_folder, '*.json'))
    json_files = sorted(json_files)

    num_frames = len(json_files)
    all_keypoints = np.zeros((num_frames, 18, 3))
    for i in range(num_frames):
        with open(json_files[i]) as f:
            json_obj = json.load(f)
            keypoints = np.array(json_obj['people'][0]['pose_keypoints'])
            all_keypoints[i] = keypoints.reshape((18, 3))
    
    output_dir = os.path.join(output_folder, os.path.basename(json_folder))
    np.save(output_dir, all_keypoints)


def load_ps(filename):
    """Load a PoseSequence object from a given numpy file.

    Args:
        filename: file name of the numpy file containing keypoints.
    
    Returns:
        PoseSequence object with normalized joint keypoints.
    """
    all_keypoints = np.load(filename)
    return PoseSequence(all_keypoints)


if __name__ == '__main__':
    main()

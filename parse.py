import argparse
import glob
import json
import numpy as np
import os

from pose import Pose, Part, PoseSequence
from pprint import pprint


def main():
    '''
    Parser temporarily disabled for notebook
    '''
    #parser = argparse.ArgumentParser(description='Pose Trainer Parser')
    #parser.add_argument('--input_folder', type=str, default='poses', help='input folder for json files')
    
    #args = parser.parse_args()

    #video_paths = glob.glob(os.path.join(args.input_folder, '*'))
    video_paths = glob.glob(os.path.join('poses', '*'))
    video_paths = sorted(video_paths)
    #print(video_paths)

    # Get all the json sequences for each video
    all_ps = []
    for video_path in video_paths:
        all_ps.append(parse_sequence(video_path))
    return all_ps


def parse_sequence(json_folder):
    """Parse a sequence of OpenPose JSON frames into a PoseSequence object.

    Args:
        json_folder: path to the folder containing OpenPose JSON for one video.
    
    Returns:
        PoseSequence object containing normalized poses in sequence.
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
    
    pose_seq = PoseSequence(all_keypoints)
    return pose_seq


if __name__ == '__main__':
    main()

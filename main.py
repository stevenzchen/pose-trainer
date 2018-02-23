import argparse
import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Pose Trainer')
    parser.add_argument('--display', type=int, default=0, help='display openpose video')
    parser.add_argument('--input_folder', type=str, default='videos', help='input folder for videos')
    parser.add_argument('--output_folder', type=str, default='poses', help='output folder for poses')

    args = parser.parse_args()

    # read filenames from the videos directory
    videos = os.listdir('videos')

    # openpose requires running from its directory
    os.chdir('openpose')

    for video in videos:
        print("processing video file:" + video)
        video_path = os.path.join('..', args.input_folder, video)
        output_path = os.path.join('..', args.output_folder, os.path.splitext(video)[0])
        openpose_path = os.path.join('bin', 'OpenPoseDemo.exe')
        subprocess.call([openpose_path, 
                        '--video', video_path, 
                        '--write_keypoint_json', output_path])



if __name__ == "__main__":
    main()
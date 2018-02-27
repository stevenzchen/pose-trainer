import argparse
import os
import sys
import subprocess
import numpy as np
import matplotlib.pyplot as plt

from parse import parse_sequence
from evaluate import evaluate_pose


def main():
    parser = argparse.ArgumentParser(description='Pose Trainer')
    parser.add_argument('--mode', type=str, default='evaluate', help='Pose Trainer application mode')
    parser.add_argument('--display', type=int, default=1, help='display openpose video')
    parser.add_argument('--input_folder', type=str, default='videos', help='input folder for videos')
    parser.add_argument('--output_folder', type=str, default='poses', help='output folder for pose JSON')
    parser.add_argument('--video', type=str, help='input video filepath for evaluation')
    parser.add_argument('--exercise', type=str, default='bicep_curl', help='exercise type to evaluate')

    args = parser.parse_args()

    if args.mode == 'batch_json':
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
    
    elif args.mode == 'evaluate':
        if args.video:
            print("processing video file...")
            video = os.path.basename(args.video)
            
            output_path = os.path.join('..', os.path.splitext(video)[0])
            openpose_path = os.path.join('bin', 'OpenPoseDemo.exe')
            os.chdir('openpose')
            subprocess.call([openpose_path, 
                            '--video', os.path.join('..', args.video), 
                            '--write_keypoint_json', output_path])
            pose_seq = parse_sequence(output_path)
            (correct, feedback) = evaluate_pose(pose_seq, args.exercise)
            if correct:
                print('Exercise performed correctly!')
            else:
                print('Exercise could be improved:')
            print(feedback)
        else:
            print('No video file specified.')
            return
    
    else:
        print('Unrecognized mode flag.')
        return




if __name__ == "__main__":
    main()
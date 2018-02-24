import glob
import json
import numpy as np

from pose import Pose, Part, PoseSequence
from pprint import pprint

def main():
    videos = glob.glob("poses/*")

    # Get all the json sequences for each video
    # pose_seq = [[json for video 1],[json for video 2],...]
    pose_seq = []
    for v in videos:
        json_frames = sorted(glob.glob(v+"/*.json"))
        pprint(json_frames)
        pose_seq.append(json_frames)
        break
    #pprint(pose_seq)

    # For every video
    for json_frames in pose_seq:
        frame_count = len(json_frames)
        data = np.zeros((frame_count, 18, 3))
        for i, frame in enumerate(json_frames):
            temp_data = json.load(open(frame))
            pose_keypoints = np.array(temp_data["people"][0]["pose_keypoints"])
            data[i] = np.array_split(pose_keypoints, 18)
            #pprint(len(_))
        #pose_obj = PoseSequence(len(json_frames), _)
        #pprint(data.shape)
        ps = PoseSequence(data)
        for pose in ps.poses:
            #pass
            print(pose.rwrist.x, pose.rwrist.y)

main()

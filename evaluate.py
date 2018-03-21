import os
import numpy as np


def evaluate_pose(pose_seq, exercise):
    """Evaluate a pose sequence for a particular exercise.

    Args:
        pose_seq: PoseSequence object.
        exercise: String name of the exercise to evaluate.

    Returns:
        correct: Bool whether exercise was performed correctly.
        feedback: Feedback string.

    """
    if exercise == 'bicep_curl':
        return _bicep_curl(pose_seq)
    elif exercise == 'shoulder_press':
        return _shoulder_press(pose_seq)
    elif exercise == 'front_raise':
        return _front_raise(pose_seq)
    elif exercise == 'shoulder_shrug':
        return _shoulder_shrug(pose_seq)
    else:
        return (False, "Exercise string not recognized.")


def _bicep_curl(pose_seq):
    # find the arm that is seen most consistently
    poses = pose_seq.poses
    right_present = [1 for pose in poses 
            if pose.rshoulder.exists and pose.relbow.exists and pose.rwrist.exists]
    left_present = [1 for pose in poses
            if pose.lshoulder.exists and pose.lelbow.exists and pose.lwrist.exists]
    right_count = sum(right_present)
    left_count = sum(left_present)
    side = 'right' if right_count > left_count else 'left'

    print('Exercise arm detected as: {}.'.format(side))

    if side == 'right':
        joints = [(pose.rshoulder, pose.relbow, pose.rwrist, pose.rhip, pose.neck) for pose in poses]
    else:
        joints = [(pose.lshoulder, pose.lelbow, pose.lwrist, pose.lhip, pose.neck) for pose in poses]

    # filter out data points where a part does not exist
    joints = [joint for joint in joints if all(part.exists for part in joint)]

    upper_arm_vecs = np.array([(joint[0].x - joint[1].x, joint[0].y - joint[1].y) for joint in joints])
    torso_vecs = np.array([(joint[4].x - joint[3].x, joint[4].y - joint[3].y) for joint in joints])
    forearm_vecs = np.array([(joint[2].x - joint[1].x, joint[2].y - joint[1].y) for joint in joints])

    # normalize vectors
    upper_arm_vecs = upper_arm_vecs / np.expand_dims(np.linalg.norm(upper_arm_vecs, axis=1), axis=1)
    torso_vecs = torso_vecs / np.expand_dims(np.linalg.norm(torso_vecs, axis=1), axis=1)
    forearm_vecs = forearm_vecs / np.expand_dims(np.linalg.norm(forearm_vecs, axis=1), axis=1)

    upper_arm_torso_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(upper_arm_vecs, torso_vecs), axis=1), -1.0, 1.0)))
    upper_arm_forearm_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(upper_arm_vecs, forearm_vecs), axis=1), -1.0, 1.0)))

    # use thresholds learned from analysis
    upper_arm_torso_range = np.max(upper_arm_torso_angles) - np.min(upper_arm_torso_angles)
    upper_arm_forearm_min = np.min(upper_arm_forearm_angles)

    print('Upper arm and torso angle range: {}'.format(upper_arm_torso_range))
    print('Upper arm and forearm minimum angle: {}'.format(upper_arm_forearm_min))

    correct = True
    feedback = ''

    if upper_arm_torso_range > 35.0:
        correct = False
        feedback += 'Your upper arm shows significant rotation around the shoulder when curling. Try holding your upper arm still, parallel to your chest, ' + \
                    'and concentrate on rotating around your elbow only.\n'
    
    if upper_arm_forearm_min > 70.0:
        correct = False
        feedback += 'You are not curling the weight all the way to the top, up to your shoulders. Try to curl your arm completely so that your forearm is parallel with your torso. It may help to use lighter weight.\n'

    if correct:
        return (correct, 'Exercise performed correctly! Weight was lifted fully up, and upper arm did not move significantly.')
    else:
        return (correct, feedback)
    
def _front_raise(pose_seq):
    poses = pose_seq.poses

    right_present = [1 for pose in poses 
            if pose.rshoulder.exists and pose.relbow.exists and pose.rwrist.exists]
    left_present = [1 for pose in poses
            if pose.lshoulder.exists and pose.lelbow.exists and pose.lwrist.exists]
    right_count = sum(right_present)
    left_count = sum(left_present)
    side = 'right' if right_count > left_count else 'left'

    print('Exercise arm detected as: {}.'.format(side))
    
    if side == 'right':
        joints = [(pose.rshoulder, pose.relbow, pose.rwrist, pose.rhip, pose.neck) for pose in poses]
    else:
        joints = [(pose.lshoulder, pose.lelbow, pose.lwrist, pose.lhip, pose.neck) for pose in poses]

    # filter out data points where a part does not exist
    joints = [joint for joint in joints if all(part.exists for part in joint)]
    joints = np.array(joints)
    
    # Neck to hip
    back_vec = np.array([(joint[4].x - joint[3].x, joint[4].y - joint[3].y) for joint in joints])
    # Check range of motion of the back
    back_vec_range = np.max(back_vec, axis=0) - np.min(back_vec, axis=0)
    print("Horizontal range of motion for back: %s" % back_vec_range[0])
    
    # Shoulder to hip    
    torso_vecs = np.array([(joint[0].x - joint[3].x, joint[0].y - joint[3].y) for joint in joints])
    # Arm
    arm_vecs = np.array([(joint[0].x - joint[2].x, joint[0].y - joint[2].y) for joint in joints])
    
    # normalize vectors
    torso_vecs = torso_vecs / np.expand_dims(np.linalg.norm(torso_vecs, axis=1), axis=1)
    arm_vecs = arm_vecs / np.expand_dims(np.linalg.norm(arm_vecs, axis=1), axis=1)
    
    # Check if raised all the way up
    angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(torso_vecs, arm_vecs), axis=1), -1.0, 1.0)))
    print("Max angle between torso and arm when lifting: ", np.max(angles))

    correct = True
    feedback = ''

    if back_vec_range[0] > 0.3:
        correct = False
        feedback += 'Your back shows significant movement. Try keeping your back straight and still when you lift the weight. Consider using lighter weight.\n'

    if np.max(angles) < 90.0:
        correct = False
        feedback += 'You are not lifting the weight all the way up. Finish with wrists at or slightly above shoulder level.\n'

    if correct:
        return (correct, 'Exercise performed correctly! Weight was lifted fully up, and no significant back movement was detected.')
    else:
        return (correct, feedback)

def _shoulder_shrug(pose_seq):
    poses = pose_seq.poses

    joints = [(pose.lshoulder, pose.rshoulder, pose.lelbow, pose.relbow, pose.lwrist, pose.rwrist) for pose in poses]

    # filter out data points where a part does not exist
    joints = [joint for joint in joints if all(part.exists for part in joint)]
    joints = np.array(joints)
    
    # Shoulder position
    shoulders = np.array([(joint[0].y, joint[1].y) for joint in joints])

    # Straining back
    shoulder_range = np.max(shoulders, axis=0) - np.min(shoulders, axis=0)
    print("Range of motion for shoulders: %s" % np.average(shoulder_range))
    
    # Shoulder to elbow    
    upper_arm_vecs = np.array([(joint[0].x - joint[2].x, joint[0].y - joint[2].y) for joint in joints])
    # Elbow to wrist
    forearm_vecs = np.array([(joint[2].x - joint[4].x, joint[2].y - joint[4].y) for joint in joints])
    
    # normalize vectors
    upper_arm_vecs = upper_arm_vecs / np.expand_dims(np.linalg.norm(upper_arm_vecs, axis=1), axis=1)
    forearm_vecs = forearm_vecs / np.expand_dims(np.linalg.norm(forearm_vecs, axis=1), axis=1)
    
    # Check if raised all the way up
    upper_arm_forearm_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(upper_arm_vecs, forearm_vecs), axis=1), -1.0, 1.0)))
    upper_forearm_angle = np.max(upper_arm_forearm_angles)
    print("Max upper arm and forearm angle: ", upper_forearm_angle)

    correct = True
    feedback = ''

    if np.average(shoulder_range) < 0.1:
        correct = False
        feedback += 'Your shoulders do not go through enough motion. Squeeze and raise your shoulders more through the exercise.\n'

    if upper_forearm_angle > 30.0:
        correct = False
        feedback += 'Your arms are bending when lifting. Keep your arms straight and still, and focus on moving only the shoulders.\n'

    if correct:
        return (correct, 'Exercise performed correctly! Shoulders went through full range of motion, and arms remained straight.')
    else:
        return (correct, feedback) 

def _shoulder_press(pose_seq):
    poses = pose_seq.poses
    
    right_present = [1 for pose in poses 
            if pose.rshoulder.exists and pose.relbow.exists and pose.rwrist.exists]
    left_present = [1 for pose in poses
            if pose.lshoulder.exists and pose.lelbow.exists and pose.lwrist.exists]
    right_count = sum(right_present)
    left_count = sum(left_present)
    side = 'right' if right_count > left_count else 'left'

    print('Exercise arm detected as: {}.'.format(side))
    
    if side == 'right':
        joints = [(pose.rshoulder, pose.relbow, pose.rwrist, pose.rhip, pose.neck) for pose in poses]
    else:
        joints = [(pose.lshoulder, pose.lelbow, pose.lwrist, pose.lhip, pose.neck) for pose in poses]

    # filter out data points where a part does not exist
    joints = [joint for joint in joints if all(part.exists for part in joint)]
    joints_ = np.array(joints)
    
    # Neck to hip
    back_vec = np.array([(joint[4].x - joint[3].x, joint[4].y - joint[3].y) for joint in joints])
    # Check range of motion of the back
    # Straining back
    back_vec_range = np.max(back_vec, axis=0) - np.min(back_vec, axis=0)
    print("Range of motion for back: %s" % back_vec_range[0])
    
    # Rolling shoulder too much
    elbow = joints_[:, 1]
    elbow_x = np.array([joint.x for joint in elbow])

    neck = joints_[:, 4]
    neck_x = np.array([joint.x for joint in neck])
    elbow_neck_dist = 0 
    if side =='right':
        elbow_neck_dist = np.min(elbow_x - neck_x)
        print("Minimum distance between elbow and neck: ", np.min(elbow_x - neck_x))
    else:
        elbow_neck_dist = np.min(neck_x - elbow_x)
        print("Minimum distance between elbow and neck: ", np.min(neck_x - elbow_x))
    
    # Shoulder to elbow    
    upper_arm_vecs = np.array([(joint[0].x - joint[1].x, joint[0].y - joint[1].y) for joint in joints])
    # Elbow to wrist
    forearm_vecs = np.array([(joint[2].x - joint[1].x, joint[2].y - joint[1].y) for joint in joints])
    
    # normalize vectors
    upper_arm_vecs = upper_arm_vecs / np.expand_dims(np.linalg.norm(upper_arm_vecs, axis=1), axis=1)
    forearm_vecs = forearm_vecs / np.expand_dims(np.linalg.norm(forearm_vecs, axis=1), axis=1)
    
    # Check if raised all the way up
    upper_arm_forearm_angles = np.degrees(np.arccos(np.clip(np.sum(np.multiply(upper_arm_vecs, forearm_vecs), axis=1), -1.0, 1.0)))
    upper_forearm_angle = np.max(upper_arm_forearm_angles)
    print("Max upper arm and forearm angle: ", np.max(upper_arm_forearm_angles))

    correct = True
    feedback = ''

    if back_vec_range[0] > 0.16:
        correct = False
        feedback += 'Your back shows significant movement while pressing. Try keeping your back straight and still when you lift the weight.\n'
    
    if elbow_neck_dist < -0.12:
        correct = False
        feedback += 'You are rolling your shoulders when you lift the weights. Try to steady your shoulders and keep them parallel.\n'
    
    if upper_forearm_angle < 178:
        correct = False
        feedback += 'You are not lifting the weight all the way up. Extend your arms through the full range of motion. Lower the weight if necessary.\n'

    if correct:
        return (correct, 'Exercise performed correctly! Weight was lifted fully up, shoulders remained parallel, and no significant back movement was detected.')
    else:
        return (correct, feedback)

    
    
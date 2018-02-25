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
    if exercise = 'bicep_curl':
        return _bicep_curl(pose_seq)
    else:
        return (False, "Exercise string not recognized.")


def _bicep_curl(pose_seq):
    # TODO: find the arm that is seen most consistently

    # TODO: for that arm, look at the initial elbow position

    # TODO: track the normalized deviance from initial position
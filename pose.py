

class PoseSequence:
    def __init__(self, sequence):
        self.poses = []
        for parts in sequence:
            self.poses.append(Pose(parts))
        
        # normalize poses based on the average torso pixel length
        torso_lengths = np.array([Part.dist(pose.neck, pose.lhip) for pose in self.poses if pose.neck.exists and pose.lhip.exists] +
                                 [Part.dist(pose.neck, pose.rhip) for pose in self.poses if pose.neck.exists and pose.rhip.exists])
        mean_torso = np.mean(torso_lengths)

        for pose in poses:
            for attr, part in pose:
                setattr(attr, part / mean_torso)



class Pose:
    PART_NAMES = ['nose', 'neck',  'rshoulder', 'relbow', 'rwrist', 'lshoulder', 'lelbow', 'lwrist', 'rhip', 'rknee', 'rankle', 'lhip', 'lknee', 'lankle', 'reye', 'leye', 'rear', 'lear']

    def __init__(self, parts):
        """Construct a pose for one frame, given an array of parts

        Arguments:
            parts - 18 * 3 ndarray of x, y, confidence values
        """
        for name, vals in zip(PART_NAMES, parts):
            setattr(self, name, Part(vals))
    
    def __iter__(self):
        for attr, value in self.__dict__.iteritems():
            yield attr, value


class Part:
    def __init__(self, vals):
        self.x = vals[0]
        self.y = vals[1]
        self.c = vals[2]
        self.exists = c != 0.0

    def __floordiv__(self, scalar):
        __truediv__(self, scalar)

    def __truediv__(self, scalar):
        return Part([self.x / scalar, self.y / scalar, self.c])

    @staticmethod
    def dist(cls, part1, part2):
        return np.sqrt(np.square(part1.x - part2.x) + np.square(part1.y - part2.y))
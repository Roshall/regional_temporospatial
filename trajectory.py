from collections import namedtuple


TrajTrack = namedtuple('TrajTrace', 'tId start_frame track')
RawTraj = namedtuple('RawTraj', 'fps life_long bbox traj_track')
Traj_Meta = namedtuple('meta_key', 'duration loc')

class Trajectory:
    def __init__(self, traj_point):
        self.points = traj_point
        self.segment_map = None

    def segment(self, region_id):
        assert self.segment_map is not None
        return self.segment_map.where_contain(region_id, None)

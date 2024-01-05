from collections import namedtuple
from dataclasses import dataclass
from typing import Sequence

TrajTrack = namedtuple('TrajTrace', 'tId clsId start_frame track')
RawTraj = namedtuple('RawTraj', 'fps life_long bbox traj_track')
Traj_Meta = namedtuple('meta_key', 'duration loc')


class Trajectory:
    def __init__(self, traj_point):
        self.points = traj_point
        self.segment_map = None

    def segment(self, region_id):
        assert self.segment_map is not None
        return self.segment_map.where_contain(region_id, None)


@dataclass(slots=True)
class BasicTrajectorySeg:
    id: int
    begin: int
    label: int

    def __lt__(self, other):
        return self.begin < other.begin


@dataclass(slots=True)
class TrajectoryIntervalSeg(BasicTrajectorySeg):
    len: int


@dataclass(slots=True)
class TrajectorySequenceSeg(BasicTrajectorySeg):
    points: Sequence

    @property
    def len(self):
        return len(self.points)

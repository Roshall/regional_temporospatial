from collections import namedtuple
from dataclasses import dataclass
from typing import Sequence

TrajTrack = namedtuple('TrajTrace', 'tId clsId start_frame track')
RawTraj = namedtuple('RawTraj', 'fps life_long bbox traj_track')
Traj_Meta = namedtuple('meta_key', 'duration loc')


@dataclass(slots=True)
class Trajectory:
    id: int
    label: int
    seg: list[int]


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

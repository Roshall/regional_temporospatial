from config import config
from trajectory import Traj_Meta

border = [list(range(8, 31, 2)) + [31], list(range(16, 71, 4))+ [71]]
config.gird_border = border
objs = [list(range(5)), {i: str(i) for i in range(4)}, 'hello', (1, 2), 2, 3.4]
keys = [Traj_Meta(102, (19, 47)), Traj_Meta(5, (17, 55)), Traj_Meta(50, (24, 49)),
        Traj_Meta(10, (23, 47)), Traj_Meta(66, (31, 20)), Traj_Meta(76, (30, 41))]
durations = [2, 43,234,456,4345,7,534,6,435,56]
begin = [45,342,67,4,7,234,89,453,890,2]
content = 'sdf', 'gdf', 'asdf', 123, 4,6, 'f', 'fgh,', ';kl', 'df'
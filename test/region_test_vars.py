from utilities.config import config
from utilities.trajectory import Traj_Meta

border = [list(range(8, 31, 2)) + [31], list(range(16, 71, 4))+ [71]]
config.gird_border = border
objs = [list(range(5)), {i: str(i) for i in range(4)}, 'hello', (1, 2), 2, 3.4]
keys = [Traj_Meta((19, 47), 102), Traj_Meta((17, 55), 5), Traj_Meta((24, 49), 50),
        Traj_Meta((23, 47), 10), Traj_Meta((31, 20), 66), Traj_Meta((30, 41), 76)]
durations = [2, 43,234,43,40,7,534,6,435,56]
begin = [45,342,67,4,7,234,89,453,890,2]
content = 'sdf', 'gdf', 'asdf', 123, 4,6, 'f', 'fgh,', ';kl', 'df'
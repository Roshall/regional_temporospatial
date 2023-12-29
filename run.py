import os.path
import dill
from time import perf_counter as now

from rest import build_tempo_spatial_index, base_query
from utilities import dataset
from utilities.box2D import Box2D
from utilities.config import config
from utilities.data_preprocessing import traj_data, gen_border


def get_index(file_name):
    base = os.path.basename(file_name).split('.')
    traj_fname = f'resource/{base[0]}_traj.pkl'
    if os.path.exists(traj_fname):
        with open(traj_fname, 'rb') as f:
            trajs = dill.load(f)
    else:
        data, cols, cls_map = dataset.load_yolo_for(file_name)
        trajs = traj_data(data, cols, 5, cls_map, scale=1)
        with open(traj_fname, 'wb') as f:
            dill.dump(trajs, f)
    broders = gen_border(trajs.bbox, 16, 11)
    config.gird_border = broders
    temp_spt = build_tempo_spatial_index(trajs)
    return temp_spt


def query():
    region = Box2D((3280, 3551, 1429, 1665))
    label = {2: 1}
    interval = 0, 60 * 60 * 10 * 6
    duration = 1, 100
    count = 0
    start = now()
    for _ in base_query(tempo_spat, region, label, duration, interval):
        count += 1
    end = now()
    print('result count:', count, 'using', end - start, 's')


filename = '../resource/archeday_0410.pkl'
tempo_spat = get_index(filename)
query()





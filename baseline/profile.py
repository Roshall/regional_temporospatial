from time import perf_counter as now

from baseline.search_methods import naive_sliding
import helper as f
from utilities.box2D import Box2D
from utilities.data_preprocessing import group_by_frame
from utilities.dataset import load_yolo_for

if __name__ == '__main__':
    file_name = '/home/lg/VDBM/spatiotemporal/resource/dataset/traj_taipei_0412.pkl'
    data, _, _, = load_yolo_for(file_name)

    label = {0: 1}
    interval = 0, 10*60*30
    duration = 3, 100
    region = Box2D((546, 727, 427, 569))
    frames = group_by_frame(data, interval)

    count = 0
    start = now()
    windows = f.sliding_window(frames, duration[0])
    for i in naive_sliding(windows, region, label, duration[0]):
        count += 1
    end = now()
    print('result count:', count, 'using', end - start, 's.')



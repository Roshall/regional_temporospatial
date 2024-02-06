import os.path

import dill
from time import perf_counter as now

from configs import cfg
from indices.builder import build_tempo_spatial_index
from search.base import base_search
from search.baseline.search_methods import sliding_framework
from search.one_pass import one_pass_search
from utilities import dataset
from utilities.box2D import Box2D
from utilities.config import config
from utilities.data_preprocessing import traj_data, gen_border, view_field
from utilities.dataset import load_yolo_for


def get_index(file_name, cfg):
    data, cols, cls_map = dataset.load_yolo_for(file_name)
    trajs = traj_data(data, cols, 5, cls_map, scale=1)
    # broders = gen_border(trajs.bbox, 16, 11)
    broders = gen_border(view_field(data, *cols[-2:]), 8, 6)
    config.gird_border = broders
    temp_spt = build_tempo_spatial_index(trajs, cfg)
    return temp_spt


def query():
    query_content = [
        Box2D((3280, 3551, 1429, 1665)),  # region
        {0: 1},  # label
        (5, 100),  # duration
        (0, 60 * 60 * 60),  # interval
    ]
    query_content[0] = Box2D((700, 1000, 427, 569))
    match cfg.QUERY:
        case 'index_one_pass':
            search_mtd = one_pass_search
        case 'index_base':
            cfg.merge_from_file('configs/region_base.yml')
            search_mtd = base_search
        case 'sliding_base':
            search_mtd = sliding_framework
        case _:
            raise ValueError(f'wrong query type: {cfg.QUERY}')
    if cfg.QUERY.startswith('index'):
        data = get_index(filename, cfg)
    else:
        data, _, _ = load_yolo_for(filename)
    count = 0
    start = now()
    for _ in search_mtd(data, *query_content):
        count += 1
    end = now()
    print(cfg.QUERY, 'result count:', count, 'using', end - start, 's')


def find_bug():
    # FIXME: found base search's BUG -> expend method is too aggressive.
    query_content = [
        Box2D((3280, 3551, 1429, 1665)),  # region
        {0: 1},  # label
        (5, 100),  # duration
        (0, 60 * 60 * 60),  # interval
    ]
    query_content[0] = Box2D((700, 1000, 427, 569))
    data = get_index(filename, cfg)
    one = set((frozenset(ids), (s, e)) for ids, s, e in one_pass_search(data, *query_content))
    cfg.merge_from_file('configs/region_base.yml')
    data = get_index(filename, cfg)
    two = set((frozenset(comv.labels), tuple(comv.interval)) for comv in base_search(data, *query_content))
    print('In one_pass but not  base: ', one - two)
    print('In base but not one_pass: ', two - one)


filename = '../resource/dataset/traj_taipei_0412.pkl'
query()
# find_bug()

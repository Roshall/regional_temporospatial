import numpy as np

from utilities.point_relation import enclose_tester


def test_contain_tester():
    region_points = ((-55.01776, 14.5065), (15.6927, 45.35849), (-24.165772, -56.203964))
    positive = [(-46.39392, 18.24323), (-28.54745, 9.5945), (-25.99528, -11.77992), (-25.99528, -11.77992),
                (9.41608, 8.63744), (16.43455, -14.65111), (16.43455, -14.65111), (46.544, -25.352)]
    negative = [(-47.84805, 63.76854), (52.85191, 38.75597), (6.06171, 42.46171), (33.35329, 7.54815),
                (75.59061, 20.56501), (-75.78417, -36.60658), (-36.95196, -28.41927),
                (-17.05008, -54.65698), (42.78191, -47.97593)]
    enclose = enclose_tester(region_points)
    res = enclose(positive+negative)
    assert (res == np.array([True] * len(positive) + [False] * len(negative))).all()


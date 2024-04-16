# -- coding: utf-8 --

class WiseRLFactory(object):
    @staticmethod
    def wiserl(use_ray):
        if use_ray == True:
            from .ray.ray_wise_rl import RayWiseRL
            return RayWiseRL()
        else:
            from .multi.multi_wise_rl import MultiWiseRL
            return MultiWiseRL()
        
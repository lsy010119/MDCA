import numpy as np


class WP:

    def __init__(self,id,loc,is_cp=False,collide_with=False):
        
        self.id = id                        # UAV ID

        self.loc = loc                      # location of waypoint
        
        self.is_cp = is_cp                  # is collision point

        self.collide_with = collide_with    # with which id of UAV collides


def multiple_insert(wps,id):

    wplist = []

    for i in range(len(wps)):

        wp = WP(id,wps[i])

        wplist.append(wp)

    return wplist
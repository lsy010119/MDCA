import numpy as np
from lib.waypoint import WP

### Waypoint Manipulate functions ###

def check_intersection(wp1_1, wp1_2, wp2_1, wp2_2):
    '''
    Check the intersection between two segment wp1_1-----wp1_2 and wp2_1-----wp2_2

    with the formula below

        L1 = ( 1 - t )*wp1_1 + t*wp1_2 
        L2 = ( 1 - s )*wp2_1 + s*wp2_2

    input : WP wp1_1 , WP wp1_2 , WP wp2_1 , WP wp2_2

    output : [0] or wp_c
    '''


    ### waypoint coordinates ###
    wp1_1 = wp1_1.loc
    wp1_2 = wp1_2.loc
    wp2_1 = wp2_1.loc
    wp2_2 = wp2_2.loc

    x1 = wp1_1[0]
    y1 = wp1_1[1]
    x2 = wp1_2[0]
    y2 = wp1_2[1]
    
    x3 = wp2_1[0]
    y3 = wp2_1[1]
    x4 = wp2_2[0]
    y4 = wp2_2[1]



    if (y4 - y3)*(x2 - x1) - (x4 -x3)*(y2 - y1) == 0: 
        
        ''' Case#1 : Segments are Parallel '''

        return np.zeros(1) # no intersections
    
    else:

        t = ( (x4 - x3)*(y1- y3) - (y4 - y3)*(x1 - x3) ) / ( (y4 - y3)*(x2 - x1) - (x4 -x3)*(y2 - y1) )
        s = ( (x2 - x1)*(y1- y3) - (y2 - y1)*(x1 - x3) ) / ( (y4 - y3)*(x2 - x1) - (x4 -x3)*(y2 - y1) )
        
        if ( t < 0 or 1 < t ) or ( s < 0 or 1 < s ):

            ''' Case#2 : Intersection locates outer of the segment '''

            return np.zeros(1) # no intersections

        else:

            ''' Case#3 : Intersection lies on the segments '''

            wp_c = np.zeros((2))

            wp_c[0] = x1 + t*(x2 -x1)
            wp_c[1] = y1 + t*(y2 -y1)

            return wp_c # intersection coordinate


def insert_collision_point(UAVs):
    '''
    Check the collision points of UAVs

    input : [ UAV uav1, UAV uav2, .... UAV uavk ]

    output : [ UAV uav1', UAV uav2', .... UAV uavk' ], int N_c
    '''

    K = len(UAVs)                  # total number of UAVs

    UAVs_inserted = UAVs           # copy for results

    N_c = 0                        # total number of collision points

    for i in range(K):

        for j in range(K-i-1):

            uavi = UAVs[i]         # ith uav
            uavj = UAVs[i+j+1]     # jth uav

            for ik in range(len(uavi.wp)-1,0,-1):  # for N_i waypoints of uav_i

                ### n'th segment of i'th UAV ###
                wpi_1 = uavi.wp[ik-1] 
                wpi_2 = uavi.wp[ik]

                for jk in range(len(uavj.wp)-1,0,-1): # for N_j waypoints of uav_j

                    ### m'th segment of j'th UAV ###
                    wpj_1 = uavj.wp[jk-1]
                    wpj_2 = uavj.wp[jk]

                    ### check intersection ###
                    wp_c = check_intersection(wpi_1,wpi_2,wpj_1,wpj_2)

                    if len(wp_c) != 1: # if intersection point exists

                        wpi_j = WP(i,wp_c,is_cp=True,collide_with=i+j+1)
                        wpj_i = WP(i+j+1,wp_c,is_cp=True,collide_with=i)

                        UAVs_inserted[i].wp.insert(ik,wpi_j)
                        UAVs_inserted[i+j+1].wp.insert(jk,wpj_i)

                        N_c += 1

    return UAVs_inserted, N_c


def split_segment(UAVs,interval):
    '''
    Split the segments of UAVs with given interval

    input : [ UAV uav1, UAV uav2, .... UAV uavk ]

    output : [ UAV uav1', UAV uav2', .... UAV uavk' ]
    '''

    K = len(UAVs)                  # total number of UAVs

    UAVs_inserted = UAVs           # copy for results


    for i in range(K):             # for i'th UAVs

        UAVs[i].calculate_dist()   # calculate [d1,d2,...,dN]

        for n in range(len(UAVs[i].wp)-1,0,-1):     # iterate from backward waypoints

            n_split = int(UAVs[i].d[n-1]/interval)  # number of waypoints to insert in the segment

            start_point = UAVs[i].wp[n-1].loc
            end_point = UAVs[i].wp[n].loc

            for m in range(1, n_split):             # inserting n_split-1 waypoints

                insert_loc = (m/n_split)*start_point + (1-m/n_split)*end_point

                insert_point = WP(i,insert_loc)

                UAVs_inserted[i].wp.insert(n,insert_point)


        UAVs_inserted[i].calculate_dist()  # re-calculate [d1',d2',...,dN']

        UAVs_inserted[i].N = len(UAVs_inserted[i].wp) # recounting the number of waypoints

    return UAVs_inserted

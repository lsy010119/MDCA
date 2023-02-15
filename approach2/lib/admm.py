import numpy as np
from numpy import array,zeros,zeros_like,ones

class ADMM:

    def __init__(self, UAVs, v_min, v_max, d_safe, N, N_c, t_st):

        ### UAVs ###
        self.UAVs   = UAVs

        ### Constraints ###
        self.v_min  = v_min
        self.v_max  = v_max
        self.d_safe = d_safe

        ### Numbers ###
        self.K      = len(UAVs)
        self.N      = N
        self.N_c    = N_c

        ### Variables ###
        self.t      = zeros((self.N,1))
        self.d      = zeros((self.N-self.K,1))
        self.t_st   = t_st

        ### Coefficient Matrices ###

        # Cost function
        self.q = zeros((self.N,1))
        self.Q = zeros((int(self.N*(self.N-1)/2),self.N))

        # Constraints 
        self.S_1 = zeros((self.N-self.K,self.N))
        self.S_2 = zeros((self.N_c,self.N))
        self.S_3 = zeros((self.K,self.N))

        # Augmented Lagrangian
        

        ### Initializing ###

        N_temp =  0

        for id,uav in enumerate(UAVs):

            # d
            self.d[N_temp:N_temp+uav.N-1,0] = uav.d

            # S_1
            self.S_1[N_temp:N_temp+uav.N-1, N_temp+id:N_temp+id+uav.N-1] += np.diag(-np.ones(uav.N-1))
            self.S_1[N_temp:N_temp+uav.N-1, N_temp+id+1:N_temp+id+uav.N] += np.diag(np.ones(uav.N-1))

            # S_2

            # S_3
            self.S_3[id,N_temp+id] = 1


            N_temp += uav.N-1

        print(self.S_3)
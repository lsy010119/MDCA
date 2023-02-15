import numpy as np
from numpy import array,zeros,zeros_like,ones
from numpy.linalg import inv, norm

import matplotlib.pyplot as plt

class ADMM:

    def __init__(self, UAVs, v_min, v_max, d_safe, N, N_c, t_st, c_set):

        ### UAVs ###
        self.UAVs   = UAVs

        ### Constraints ###
        self.v_min  = v_min
        self.v_max  = v_max
        self.d_safe = d_safe
        self.t_safe = d_safe/v_min

        ### Numbers ###
        self.K      = len(UAVs)
        self.N      = N
        self.N_c    = N_c

        ### Variables ###
        self.t      = zeros((self.N,1))
        self.s      = zeros((self.N-self.K,1))
        self.x      = zeros((self.N_c,1))
        self.lam    = 10*ones((self.N+self.N_c,1))

        ### Constants ###
        self.d      = zeros((self.N-self.K,1))
        self.c_set  = c_set
        self.t_st   = t_st
        self.p      = 10

        ### Coefficient Matrices ###

        # Cost function
        self.q = zeros((self.N,1))
        self.Q = zeros((int(self.K*(self.K-1)/2),self.N))

        # Constraints 
        self.S_1 = zeros((self.N-self.K,self.N))
        self.S_2 = zeros((self.N_c,self.N))
        self.S_3 = zeros((self.K,self.N))

        # Augmented Lagrangian
        

        ### Initializing ###

        N_temp  = 0
        N_temp2 = 0
        N_temp3 = 0

        # t_init, q, d, S_1, S_3, Q
        for id,uav in enumerate(UAVs):

            # t_init
            dt_init = 2*uav.d/(self.v_max + self.v_min)
            ti_init = self.t_st[id]*np.ones(uav.N)+np.tril(np.ones(uav.N),k=-1)[:,:-1]@dt_init

            self.t[N_temp2:N_temp2+uav.N,0] = ti_init

            # q
            self.q[N_temp2+uav.N-1,0] = 1

            # d
            self.d[N_temp:N_temp+uav.N-1,0] = uav.d

            # S_1
            self.S_1[N_temp:N_temp+uav.N-1, N_temp2:N_temp2+uav.N-1] += np.diag(-np.ones(uav.N-1))
            self.S_1[N_temp:N_temp+uav.N-1, N_temp2+1:N_temp2+uav.N] += np.diag(np.ones(uav.N-1))

            # S_3
            self.S_3[id,N_temp2] = 1

            N_temp  += uav.N-1
            N_temp2 += uav.N
            N_temp4 =  0

            # Q
            for id2 in range(self.K-id-1):

                N_temp4 += self.UAVs[id+id2+1].N

                self.Q[N_temp3 + id2 ,N_temp2 - 1] = 1

                self.Q[N_temp3 + id2 ,N_temp2 + N_temp4 - 1] = -1

            N_temp3 += self.K - id - 1

        # S_2
        for idx, collision_pair in enumerate(self.c_set):
            
            N_temp5i = 0
            N_temp5j = 0

            id_i,id_j,idx_n,idx_m = collision_pair

            for i in range(id_i): N_temp5i += self.UAVs[i].N
            for j in range(id_j): N_temp5j += self.UAVs[j].N

            self.S_2[idx,N_temp5i+idx_n] = 1
            self.S_2[idx,N_temp5j+idx_m] = -1

        # s_init
        self.s = self.d/self.v_min - self.S_1@self.t

        # x_init
        self.x = self.S_2@self.t

        # P
        self.P = self.Q.T @ self.Q

        # A
        self.A = np.block([[self.S_1],[self.S_2],[self.S_3]])

        # B
        self.B = np.block([[np.eye(self.N-self.K)],[np.zeros((self.N_c+self.K,self.N-self.K))]])
        
        # C
        self.C = np.block([[np.zeros((self.N-self.K,self.N_c))],[-np.eye(self.N_c)],[np.zeros((self.K,self.N_c))]])

        # a
        self.a = np.block([[self.d/self.v_min],[np.zeros((self.N_c,1))],[t_st]])


        ### Constant Matrices & Inverses ###

        # t update step : R & R^{-1}
        self.Rt = self.P + (self.p/2)*self.A.T@self.A
        self.Rt_inv = inv(self.Rt)

        # s update step : R & R^{-1}
        self.Rs = (self.p/2)*np.eye((self.N-self.K))
        self.Rs_inv = (2/self.p)*np.eye((self.N-self.K))

        self.S_1_inv = self.S_1.T@inv(self.S_1@self.S_1.T)

        # x update step : R & R^{-1}
        self.Rx = (self.p/2)*np.eye((self.N_c))
        self.Rx_inv = (2/self.p)*np.eye((self.N_c))



    def update_t(self):

        s,x,lam,A,B,C,a,p,q = self.s,self.x,self.lam,self.A,self.B,self.C,self.a,self.p,self.q

        # r = (1/2)*( q + A.T@lam + p*A.T@(B@s+C@x-a))

        r = (1/2)*( q.T + lam.T@A + p*( s.T@B.T@A + x.T@C.T@A - a.T@A ) ).T

        self.t = -self.Rt_inv@r

    
    def update_s(self):

        t,x,lam,A,B,C,a,p,q = self.t,self.x,self.lam,self.A,self.B,self.C,self.a,self.p,self.q

        # r = (1/2)*B.T@(lam + p*(A@t+C@x-a))

        r = (1/2)*( lam.T@B + p*( t.T@A.T@B + x.T@C.T@B - a.T@B ) ).T

        self.s = -self.Rs_inv@r

        self.proj_S()

    
    def update_x(self):

        t,s,lam,A,B,C,a,p,q = self.t,self.s,self.lam,self.A,self.B,self.C,self.a,self.p,self.q

        # r = -(1/2)*C.T@(lam + p*(A@t+B@s-a))

        r = (1/2)*( lam.T@C + p*( t.T@A.T@C + s.T@B.T@C - a.T@C ) ).T

        self.x = -self.Rx_inv@r

        self.proj_D()


    def update_lam(self):

        t,s,x,lam,A,B,C,a,p,q = self.t,self.s,self.x,self.lam,self.A,self.B,self.C,self.a,self.p,self.q

        self.lam = lam + p*( A@t + B@s + C@x - a )


    def proj_S(self):

        for i in range(self.N-self.K):

            s_i = self.s[i]

            if s_i < 0:

                self.s[i] = 0

            elif s_i > self.d[i]/self.v_min - self.d[i]/self.v_max:

                self.s[i] = self.d[i]/self.v_min - self.d[i]/self.v_max


        # self.t = self.S_1_inv@(self.d/self.v_min - self.s)


    def proj_D(self):

        for i in range(self.N_c):

            x_i = self.x[i]

            idx_i, idx_j = np.where(self.S_2[i] == 1)[0][0], np.where(self.S_2[i] == -1)[0][0]

            if abs(x_i) < self.t_safe and x_i >= 0:

                # ti_j, tj_i = self.t[idx_i], self.t[idx_j]

                # ti_j_ = (ti_j+tj_i)/2 + self.t_safe/2
                # tj_i_ = (ti_j+tj_i)/2 - self.t_safe/2

                # self.t[idx_i], self.t[idx_j] = ti_j_, tj_i_

                self.x[i] = self.t_safe

            elif abs(x_i) < self.t_safe and x_i < 0:

                # ti_j, tj_i = self.t[idx_i], self.t[idx_j]

                # ti_j_ = (ti_j+tj_i)/2 - self.t_safe/2
                # tj_i_ = (ti_j+tj_i)/2 + self.t_safe/2

                # self.t[idx_i], self.t[idx_j] = ti_j_, tj_i_

                self.x[i] = -self.t_safe


    def cost_function(self):

        J = (self.t.T@self.P@self.t + self.q.T@self.t)[0,0]

        return J


    def constraints(self):

        C = self.A@self.t - self.a

        return C


    def run(self,N_iter):

        J_list = np.zeros(N_iter)
        C_list = np.zeros(N_iter)

        for i in range(N_iter):

            self.update_t()
            self.update_s()
            self.update_x()
            self.update_lam()

            J = self.cost_function()
            C = self.constraints()

            J_list[i] = J

        return self.t
        
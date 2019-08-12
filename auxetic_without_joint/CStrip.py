import symbfb.Nonlinear_System as sns
import symbfb.compstrip as d2c
import numpy as np
import pdb;
from math import *
import scipy as sc

class CStrip(sns.Nonlinear_System):

    def __init__(self,l=2.,e=0.1, path_sym = 3, refinedefault = 0,n=10):
        super(CStrip,self).__init__(N=None)
        #pdb.set_trace()
        self.refine = refinedefault
        self.l = l
        self.e = e
        a = 0.5
        Lenght = 3

        self.x_coords = None
        self.y_coords = None
        self.x_or_y = None
        self.vertx = None
        self.verty = None
        self.vertu = None
        self.vertv = None

        self.cstrip = d2c.compstripProblem(l_in = l,e_in = e,a_in = a,Lenght_in = Lenght,nu_in=0.3,mu0_in=0.1,kappa_in=0, n_in=n,refine=refinedefault)
        print("compstrip problem initialised")
        self.get_dof_coords()

        self.N = 2*self.vertx.shape[0]
        n = self.N
        self.Fvalid = False
        self.Jvalid = False
        self.x = np.zeros((n+4,1))
        self.F = np.zeros((n,1))
        self.J = np.zeros((n,n))
        self.Jlambda = np.zeros((n,1))
        self.bigJ = np.zeros((n,n+4))

        print("before calling compsritProblem")
        #pdb.set_trace()

        print("after calling compsritProblem")

        self.problem_x= []
        self.problem_y= []        



        self.compute_reps(path_sym)
        #self.T=np.identity(self.N)
        print("fin reps")
        w,v = np.linalg.eig(self.T)
        #print("w,v")
        #print(w,v)
        #print(v[0].shape)
        #print("fin w,v")
        indices = []
        for i in range(w.shape[0]):
            if np.abs(w[i]-1)<1e-10:
                indices.append(i)
        tmp = v[:,indices]
        #print("tmp")
        #print(tmp)
        q,r = np.linalg.qr(tmp)
        #print("q,r")
        #print(q,r)
        self.fpbasis = np.real(q)
        a='''        if(path_sym==1):
             print("before killing rigid body movement \n",self.fpbasis)
             Corrected = False
             i = 0
             while(Corrected==False and i< self.fpbasis.shape[1]):
                  if(self.fpbasis[0,i] !=0):
                        for j in range(self.fpbasis.shape[0]):
                              if(self.fpbasis[j,i]!=0):
                                    print("applying change",j,i,self.fpbasis[j,i])
                              self.fpbasis[j,i]=0
                        Corrected=True
                  else:
                        i +=1
             print(Corrected, "after killing rigid body movement \n",self.fpbasis)'''

        print(self.fpbasis.shape)
        

        if(path_sym==3):
              self.fpbasis = np.identity(2*self.vertx.shape[0])
        
        print('Dimension of fixed point basis ',self.fpbasis.shape[1])
        
        self.path = None
        print("fin ini")
        #self.constrained_dofs = self.cstrip.get_constrained_dofs()
        
    def compute_reps(self,path_sym):
        l = self.l
        #e = self.e

        vertx=self.vertx
        verty=self.verty
        vertu=self.vertu
        vertv=self.vertv
        #pdb.set_trace()
        #print('vertu.shape[0]',vertu.shape[0])
        ndofs = 2*vertx.shape[0]
        #print('vertx.shape[0]',vertx.shape[0])
        #print('vertu,vertv',vertu,vertv)
        
        TSigmax = np.zeros((ndofs,ndofs))
        TSigmay = np.zeros((ndofs,ndofs))
        TC2 = np.zeros((ndofs,ndofs))

   

        for idx1 in range(vertx.shape[0]):
            counter = 0
            for idx2 in range(vertx.shape[0]):
                if(abs(verty[idx1] - verty[idx2]) < 1e-5 and abs(vertx[idx1] + vertx[idx2]) < 1e-5):
                    #print(idx1,vertx[idx1],verty[idx1],idx2,vertx[idx2],verty[idx2],vertu[idx1],vertu[idx2],vertv[idx1],vertv[idx2])
                    TSigmay[vertu[idx1],vertu[idx2]] = -1
                    TSigmay[vertv[idx1],vertv[idx2]] = 1
                    counter +=1
                if(abs(verty[idx1] - l + verty[idx2]) < 1e-5 and abs(vertx[idx1] - vertx[idx2]) < 1e-5):
                    #print(idx1,vertx[idx1],verty[idx1],idx2,vertx[idx2],verty[idx2],vertu[idx1],vertu[idx2],vertv[idx1],vertv[idx2])
                    TSigmax[vertu[idx1],vertu[idx2]] = 1
                    TSigmax[vertv[idx1],vertv[idx2]] = -1
                    counter +=1
                if(abs(verty[idx1] - l + verty[idx2]) < 1e-5 and abs(vertx[idx1] + vertx[idx2]) < 1e-5):
                    #print(idx1,vertx[idx1],verty[idx1],idx2,vertx[idx2],verty[idx2],vertu[idx1],vertu[idx2],vertv[idx1],vertv[idx2])
                    TC2[vertu[idx1],vertu[idx2]] = -1
                    TC2[vertv[idx1],vertv[idx2]] = -1
                    counter +=1
            if(counter != 3):
                self.problem_x.append(vertx[idx1])
                self.problem_y.append(verty[idx1])
                print("problem !!",counter)
        
        #group order e,C2, \sigma x, \sigma y 
        T_list = [np.identity(ndofs),TC2,TSigmax,TSigmay]

        for i in range(4):
            fpbasis = np.array(T_list[i]).astype(np.float64)
            fpbasistemp = np.vstack((fpbasis,np.zeros((3,fpbasis.shape[1]))))
            last_vector = np.zeros((fpbasistemp.shape[0],3))
            last_vector[-1,2] = 1
            last_vector[-2,1] = 1
            last_vector[-3,0] = 1
            T_list[i] = np.hstack((fpbasistemp,last_vector))
            print(T_list[i].shape)

        ndofs=ndofs+3





        print(len(T_list))
        XiA1=[1,1,1,1]
        XiA2=[1,1,-1,-1]
        XiB1=[1,-1,1,-1]
        XiB2=[1,-1,-1,1]

        Ximu=[XiA1,XiA2,XiB1,XiB2]
       

        gmu=[1,1,1,1]
        etanu=[0,0,0,0]
        Projectormu=[np.zeros((ndofs,ndofs)),np.zeros((ndofs,ndofs)),np.zeros((ndofs,ndofs)),np.zeros((ndofs,ndofs)),np.zeros((ndofs,ndofs)),np.zeros((ndofs,ndofs))]
        Projectormu1=[np.zeros((ndofs,ndofs)).astype(complex),np.zeros((ndofs,ndofs)).astype(complex),np.zeros((ndofs,ndofs)).astype(complex),np.zeros((ndofs,ndofs)).astype(complex),np.zeros((ndofs,ndofs)).astype(complex),np.zeros((ndofs,ndofs)).astype(complex)]
        Projectormu21=[np.zeros((ndofs,ndofs)),np.zeros((ndofs,ndofs)),np.zeros((ndofs,ndofs)),np.zeros((ndofs,ndofs)),np.zeros((ndofs,ndofs)),np.zeros((ndofs,ndofs))]


        for j in range(4):
            for i in range(len(T_list)):
                Projectormu[j] += gmu[j]*(1./4.)*Ximu[j][i]*T_list[i].copy()
                etanu[j] += Ximu[j][i]*np.trace(T_list[i])
            etanu[j] = 1/4*etanu[j]
            #Projectormu[j] = (etanu[j] / np.trace(Projectormu[j]))* Projectormu[j].copy()
            #Projectormu1[j] = (etanu[j] / np.trace(Projectormu1[j]))* Projectormu1[j].copy()
        self.etanu=etanu.copy()
        self.Projectormu=Projectormu.copy()

        print(etanu)


        #cumputing an ON basis for each Projectormu1
        basis = [0,0,0,0] #mu k i


        for i in range(4):
            basis[i]=np.int(etanu[i])*[0]
            for k in range(np.int(etanu[i])):
                basis[i][k]=np.int(gmu[i])*[0]
               

        self.basis_full=[]

        for i in range(4):
            indices=[]
            w,v=np.linalg.eig(Projectormu[i])
            #print("w sorted",np.sort(w))
            for j in range(w.shape[0]):
                if np.abs(w[j]-1)<1e-10:
                    indices.append(j)
            #print("indices", indices)
            print("for mu = ",i," ",len(indices),"eigen vector having 1 as eigen value")
            eigen_vector = v[:,indices]
            print("mu =",i," eigen_vector = \n",eigen_vector.shape)
            q = self.GS(eigen_vector)
            for k in range(np.int(etanu[i])):
                basis[i][k][0]=np.real(q[:,k])/np.linalg.norm(np.real(q[:,k]))
     
        for i in range(4):
            for k in range(np.int(etanu[i])):
                self.basis_full.append(basis[i][k][0])


        self.basis=basis.copy()


        self.T_list=T_list






        if(path_sym==0):
             Tavg = TSigmax + TSigmay
             num_reps = 2
             print("sigmax")
             print(TSigmax)
             print("sigmay")
             print(TSigmay)
             self.T = 1/(num_reps) * Tavg
             print(Tavg.shape)
             print("Tavg")
             print(Tavg)
        if(path_sym==1):
             Tavg = TSigmax
             num_reps = 1
             print("sigmax")
             print(TSigmax)
             self.T = 1/(num_reps) * Tavg
             print(Tavg.shape)
             print("Tavg")
             print(Tavg)
        if(path_sym==2):
             Tavg = TSigmay
             num_reps = 1
             print("sigmay")
             print(TSigmay)
             self.T = 1/(num_reps) * Tavg
             print(Tavg.shape)
             print("Tavg")
             print(Tavg)
        if(path_sym==3):
             self.T = np.identity(ndofs)
        if(path_sym==4): #TC2
             Tavg = TC2
             num_reps = 1
             print("TC2")
             print(TC2)
             self.T = 1/(num_reps) * Tavg
             print(Tavg.shape)
             print("Tavg")
             print(Tavg)

    def set_x(self,x):
        #self.set_config(x[:-1,:])
        #self.set_param(x[-1,0])
        self.cstrip.set_dofs(x)
        self.FValid = False
        self.Jvalid = False
        #print("Set_x...",np.linalg.norm(x-self.path.project(x.copy())))

    def set_config(self,x):
        print("WARNING THIS IS BAD")
        self.Fvalid = False
        self.Jvalid = False
        self.x[:-1,:] = x.copy()
        self.cstrip.set_dofs(x.reshape(-1))

    def set_param(self,param):
        print("WARNING THIS IS BAD")
        self.Fvalid = False
        self.Jvalid = False
        self.x[-1,0] = param
        self.cstrip.set_load(param)

    def get_x(self):
        return self.x.copy()

    def get_F(self):
        self.F = self.cstrip.F_resid()
        self.F = self.F.reshape(-1,1)
        #print("Get_F...",np.linalg.norm(self.F - self.path.project(self.F.copy())))
        return self.F.copy()

    def get_J(self):
        #print("get_J is dead...",flush=True)
        cols = np.eye(self.N+4)
        J = np.zeros((self.N+4,self.N+4))
        for i in range(self.N+4):
            J[:,i] = self.get_J_times_v(cols[:,i].reshape(-1))
        return J

    def get_J_times_v(self,v):
        return self.cstrip.get_J_times_u(v)

    def get_J_lambda(self):
        self.Jlambda = self.cstrip.J_lambda()
        return self.Jlambda.copy()

    def postprocesssolution(self,x):
        #x[:-1] = self.cstrip.postprocesssolution(x[:-1])
        #return x
        return self.cstrip.postprocesssolution(x)
        #return x
    
    def output_results(self):
        #self.cstrip.output_results()
        pass
        
    def get_dof_coords(self):
        self.x_coords = self.cstrip.dof_x_coords()
        self.y_coords = self.cstrip.dof_y_coords()
        self.x_or_y = self.cstrip.dof_x_or_y()
        self.vertx = self.cstrip.get_vertx()
        self.verty = self.cstrip.get_verty()
        self.vertu = self.cstrip.get_vertu()
        self.vertv = self.cstrip.get_vertv()
        
    def constrain_paca_last_row(self,u):
        u[:-1,0] = self.cstrip.constrain_paca_last_row(u[:-1,0].reshape(-1))
        return u
    
    def set_step_size(self,ss):
        self.cstrip.set_step_size(ss)
        return
    
    def set_last_solution(self,u):
        self.cstrip.set_last_solution(u)
        return

    def proj(self, u, v):
        # notice: this algrithm assume denominator isn't zero
        return u * np.dot(v,u) / np.dot(u,u)  
 
    def GS(self, V):
        V = 1.0 * V     # to float
        U = np.copy(V)
        for i in range(1, V.shape[1]):
            for j in range(i):
                U[:,i] -= self.proj(U[:,j], V[:,i])
        # normalize column
        den=(U**2).sum(axis=0) **0.5
        E = U/den
        # assert np.allclose(E.T, np.linalg.inv(E))
        return E


    

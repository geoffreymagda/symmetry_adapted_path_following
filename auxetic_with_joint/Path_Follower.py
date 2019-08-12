import numpy as np
from enum import Enum
import time
#import pdb; 

class State(Enum):
    PATHFOLLOW = 0
    CRITPTFIND = 1
    
class CritPt(object):
    def __init__(self):
        self.point = None
        self.nullvecs = None
        self.spectrum = None
        self.nb_eigen_value = None

class Path_Follower(object):
    
    def __init__(self):
        self.phase_null_vector = None
        self.phase_null_vector2 = None
        self.num_pos_eigs = None
        self.crit_pts = []
        self.tan_list = []
    
    def array_to_str(self,array,text_file,name):
        #work only for 1D array
        string = name + str(array.shape)
        text_file.write(string)
        string = ""
        for i in range (array.shape[0]):
            string = string + str(array[i])
        text_file.write(string+"\n")

    def run(self,nls,x0,nsteps = 60, step_size=6e-3, initial_direction = 1, do_crit_pt = True, path_id = 0,name="no description",criteria = 1e-12,tan_reduc=1,reduced_step_size=3*1e-3):
        tstart = time.time()
        state = State.PATHFOLLOW # 0 = Path Following, 1 = Equilibrium Point Search
        #pdb.set_trace()
        step = 0
        raw_solns = [x0.reshape(-1)]
        current_pt = x0.reshape(-1)
        last_tan = None
        
        nls.set_x(current_pt)
        initial_resid = np.linalg.norm(nls.get_F())
        print("Initial Residual...",initial_resid,time.time()-tstart)
        
        import datetime

        resid_tracker = []
        if((initial_direction is not None and initial_direction is not 1 and initial_direction is not -1)==False):
             name_initial_direction = initial_direction
        else:
             name_initial_direction="given"
        name = "simulation "+ name + "with nsteps =" + str(nsteps) + "with step_size =" + str(step_size) +"and initial direction =" + str(name_initial_direction)+str(datetime.datetime.now())+".txt"
        text_file = open(name, "w")
        self.array_to_str(x0,text_file,"initial point")

        #text_file.write("Purchase Amount: " 'TotalAmount')
        reduction= False
        found_two_cp=False
        

        self.load_list=[]
        self.trace_list=[]

        fpbasis = np.array(nls.fpbasis).astype(np.float64)
        #print(fpbasis)
        fpbasistemp = np.vstack((fpbasis,np.zeros((4,fpbasis.shape[1]))))
        last_vector = np.zeros((fpbasistemp.shape[0],4))
        last_vector[-1,3] = 1
        last_vector[-2,2] = 1
        last_vector[-3,1] = 1
        last_vector[-4,0] = 1
        fpbasistemp = np.hstack((fpbasistemp,last_vector))
        #print(fpbasistemp)
        #fpbasistemp=np.identity(392)
        projector = np.dot(fpbasistemp,fpbasistemp.T)


        stored_step_size = step_size
        large_step_size = step_size

        J = nls.get_J()[:-1,:-1]
        w,v = np.linalg.eig(J)
        print(np.sort(w)[:5])
        self.num_pos_eigs = 0
        for i in range(w.shape[0]):
            if np.abs(w[i])<1e-14:
                if self.phase_null_vector is not None:
                    print("Multiple null vectors",w[i])
                    tmp_eigen_vector = v[:,i]
                    print(tmp_eigen_vector)
                    self.array_to_str(tmp_eigen_vector,text_file,"initial eigen vector number" + str(i))
                    print_and_save("eigen_vector",nls,tmp_eigen_vector,amplification=1000)
                else:
                    print("Got null vector with eig value:",w[i])
                    self.phase_null_vector = v[:,i]
                    tmp_eigen_vector = v[:,i]
                    print(tmp_eigen_vector)
                    self.array_to_str(tmp_eigen_vector,text_file,"initial eigen vector number" + str(i))
                    print_and_save("eigen_vector",nls,tmp_eigen_vector,amplification=1000)
            elif np.real(w[i])>0:
                self.num_pos_eigs = self.num_pos_eigs + 1
           
        nls.path.eq_pts.append(current_pt)
        while (step < nsteps and len(self.crit_pts)<4):
            
            self.load_list.append(current_pt[-1])
            self.trace_list.append(-current_pt[-2]-current_pt[-4])

            nls.set_x(current_pt)

            print(current_pt[-4],current_pt[-3],current_pt[-2],current_pt[-1])
            
            if (step == 0) and (initial_direction is not None)and (initial_direction is not 1)and (initial_direction is not -1):
                print("Using the initial direction provided")
                tan = initial_direction.reshape(-1)
            else:
                _,s,vh = np.linalg.svd(nls.get_J()[:-1*tan_reduc,:])
                print("SVD: ",s[-5:], s.shape, vh.shape)
                if path_id == 0:
                    tan = (vh[-1,:]).reshape(-1)
                elif path_id == 1:
                    tan = (vh[-1,:] + vh[-2,:] + vh[-3,:]).reshape(-1)
            
            #print("Tangent Shape",tan.shape)
            tan = nls.path.project(tan.copy())
            
            tan = tan / np.linalg.norm(tan)
            

            if(step >0):
                tan=nls.path.eq_pts[-1]-nls.path.eq_pts[-2]
                tan = nls.path.project(tan.copy())
                tan = tan / np.linalg.norm(tan)


            if last_tan is None:
                if tan[-1] < 0 and initial_direction==1:
                    tan = -1.0 * tan
                elif tan[-1] > 0 and initial_direction==-1:
                    tan = -1.0 * tan
                elif step == 0 and (initial_direction is not None)and (initial_direction is not 1)and (initial_direction is not -1):
                    print("Tan is 0 as expected")
            else:
                print("Tan dot Last Tan:",np.dot(last_tan,tan))
                if np.dot(last_tan,tan) < 0:
                #if tan[-1] < 0:
                    tan = -1.0 * tan
                if np.dot(last_tan,tan) < 0.7:
                    print("WARNING !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            

            if(step >0):
                tan=nls.path.eq_pts[-1]-nls.path.eq_pts[-2]
                tan = nls.path.project(tan.copy())
                tan = tan / np.linalg.norm(tan)

            
            if(len(self.crit_pts)==2 and found_two_cp==False ):
                step_size=large_step_size
                stored_step_size=large_step_size
                found_two_cp= True


            if( reduction == True):
                step_size=reduced_step_size
                stored_step_size=reduced_step_size
                reduction=False

            self.tan_list.append(tan)
            new_pt = current_pt + step_size * tan
            
            nls.set_x(new_pt)
            
            if state == State.PATHFOLLOW:
                step = step + 1
            #The below code checks the system matrix with a numerical derivative

            resid = np.linalg.norm(nls.get_F())
            
            nls.set_last_solution(current_pt)
            nls.set_step_size(step_size)
            
            resid_counter = 0

            while resid > criteria:
                #pdb.set_trace()
                if(resid_counter %10==0):
                     print("RESID " + str(resid))
                     #print(min(abs(np.linalg.eig(A)[0])))
                resid_counter = resid_counter + 1
                A = np.dot(np.dot(fpbasistemp.T,nls.get_J()),fpbasistemp)
                #if(np.linalg.matrix_rank(A)!=A.shape[0]):
                     #print("not inversible")

                b = np.dot(fpbasistemp.T,nls.get_F().reshape(-1))
                
                r = m = b.shape[0]
                beta = np.linalg.norm(b)
                V = np.zeros((r,m))
                H = np.zeros((r+1,m))
                Z = np.zeros((r,m))
                V[:,0] = b / beta
                x_test = 0*b

                temp_solns = [x_test]
                doProjectCounter = 0

                for j in range(m):
                    #pdb.set_trace()                    
                    z = V[:,j]
                    Z[:,j] = z# / np.linalg.norm(z)
                    w = np.dot(A,Z[:,j])
                    
                    for i in range(j+1):
                        H[i,j] = np.dot(w,V[:,i])
                        w = w - H[i,j]*V[:,i]
                
                    H[j+1,j] = np.linalg.norm(w)
                    
                    if(j+1<m):
                        #print(H[j+1,j])
                        V[:,j+1] = w/H[j+1,j]
                        
                    rhs = np.zeros(j+2)
                    rhs[0] = beta
                    y = np.linalg.lstsq(H[:j+2,:j+1],rhs,rcond=None)
                    y = y[0]
                    
                    x_test = np.dot(Z[:,:j+1],y)
                    
                    b_test = np.dot(A,x_test)
                    
                    gmresid = np.linalg.norm(b-b_test)
                    
                    #resid_tracker.append([j+1,gmresid])
                    
                    if gmresid < 0.9*criteria:
                        #print("GMRES Converged...",j,gmresid)
                        for idx,item in enumerate(temp_solns):
                            resid_tracker.append([idx+1,np.linalg.norm(item-x_test)])
                        break
                    
                    temp_solns.append(x_test)
                
                new_pt = new_pt - np.dot(fpbasistemp,x_test)
                nls.set_x(new_pt)

                resid = np.linalg.norm(np.dot(fpbasistemp.T,nls.get_F()))
                #print("Corrector Residual...",resid)
                
            if state==State.PATHFOLLOW:
                raw_solns.append(new_pt.copy())
                nls.path.eq_pts.append(new_pt.copy())
                current_pt = new_pt.copy()
                last_tan = tan.copy()
                print("Step: ",step, "Load: ",current_pt[-1],self.num_pos_eigs,time.time()-tstart)
                self.array_to_str(current_pt,text_file,"current_pt at the end of step"+ str(step))
                #print("point",new_pt)
            
            if do_crit_pt:
                if state==State.PATHFOLLOW:
                    num_pos_eigs = 0
                    nls.set_x(new_pt)
                    J1 = nls.get_J()[:-1,:-1]
                    J = nls.get_J()
                    Jred = np.dot(np.dot(fpbasistemp.T,nls.get_J()),fpbasistemp)[:-1,:-1]
                    w1,v1 = np.linalg.eig(J1)
                    w,v = np.linalg.eig(J)
                    wred,vred = np.linalg.eig(Jred)
                    self.phase_null_vector2 = None
                    print('min eigen values list -1', np.sort(w1)[:5])
                    print('min eigen values list', np.sort(w)[:5])
                    print('min eigen values list reduced system -1', np.sort(wred)[:5])
                    print("1",min(w1))
                    self.array_to_str(np.sort(w1),text_file,"sorted eigen value at the end of step"+ str(step))
                    for i in range(w1.shape[0]):
                        if np.abs(w1[i])<1e-14:
                            if self.phase_null_vector2 is not None:
                                print("Multiple null vectors",w1[i])
                                tmp_eigen_vector = v1[:,i]
                                #print(tmp_eigen_vector)
                                self.array_to_str(tmp_eigen_vector,text_file,"eigen vector number" + str(i) + "for step" + str(step))
                            else:
                                print("Got null vector with eig value:",w1[i])
                                tmp_eigen_vector = v1[:,i]
                                #print(tmp_eigen_vector)
                                self.phase_null_vector2 = v1[:,i]
                                self.array_to_str(tmp_eigen_vector,text_file,"eigen vector number" + str(i) + "for step" + str(step))
                        elif np.real(w1[i])>0:
                            num_pos_eigs += 1
                        if(w1[i] < 5*1e-6 and w1[i]>0):
                            reduction=True
                    #self.phase_null_vector = self.phase_null_vector / np.linalg.norm(self.phase_null_vector)
                    #self.phase_null_vector2 = self.phase_null_vector2 / np.linalg.norm(self.phase_null_vector2)
                    #print("Null Vector Checker:",np.dot(self.phase_null_vector,self.phase_null_vector2))
                    if num_pos_eigs != self.num_pos_eigs:
                        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$Crossed critical point")
                        text_file.write("\n Crossed critical point \n")
                        new_num_pos_eigs = num_pos_eigs
                        minstepsize = 0
                        maxstepsize = step_size
                        step_size = 0.5*(minstepsize + maxstepsize)
                        current_pt = raw_solns[-2].copy()
                        state = State.CRITPTFIND
                elif state == State.CRITPTFIND:
                    num_pos_eigs = 0
                    nls.set_x(new_pt)
                    J = nls.get_J()[:-1,:-1]
                    w,v = np.linalg.eig(J)
                    nullvecs = []
                    num_null_vecs = 0
                    wsorted = w.copy()
                    wsorted.sort()
                    self.array_to_str(wsorted,text_file,"critical eigen value sorted")
                    print("wsorted",wsorted[:5])
                    #print(wsorted[-3:])
                    for i in range(w.shape[0]):
                        if np.abs(w[i])<1e-14:
                            #nullvecs.append(v[:,i])
                            num_null_vecs += 1
                        elif np.real(w[i])>0:
                            num_pos_eigs += 1
                    print("nb positiv eigen values before and after",self.num_pos_eigs,num_pos_eigs)
                    if num_pos_eigs >= self.num_pos_eigs:
                        minstepsize = 0
                        current_pt=new_pt.copy()
                    else:
                        maxstepsize = step_size
                    if maxstepsize - minstepsize < criteria/10:
                        for i in range(w.shape[0]):
                            if np.abs(w[i])<2*min(abs(w)):
                                nullvecs.append(v[:,i])
                            self.array_to_str(v[:,i],text_file,"critical eigen vector for eigen value"+str(w[i]))
                        print("Found Critical Point.  Load:",new_pt[-1], len(nullvecs))
                        self.array_to_str(new_pt,text_file,"critical point")
                        cp = CritPt()
                        cp.point = new_pt.copy()
                        cp.nullvecs = nullvecs
                        cp.spectrum = w.copy()
                        self.crit_pts.append(cp)
                        self.num_pos_eigs = self.num_pos_eigs - 1
                        step_size = stored_step_size
                        current_pt = raw_solns[-1]
                        state = State.PATHFOLLOW
                    else:
                        step_size = 0.5*(minstepsize + maxstepsize)
                        print("Searching for CP: step size: ",step_size,self.num_pos_eigs,num_pos_eigs,num_null_vecs)
                        
        text_file.close()           
        return resid_tracker



import matplotlib.pyplot as plt
import numpy as np

def print_and_save(name,nls,point,amplification=1):
    newxcoords = nls.vertx.copy()
    newycoords = nls.verty.copy()
    point = np.real(point)
    cp=nls.postprocesssolution(point.reshape(-1))
    cp.reshape(-1,1)
    Uxx=cp[-4]
    Uxy=cp[-3]
    Uyy=cp[-2]
    for idx in range(newxcoords.shape[0]):
        uidx = nls.vertu[idx]
        vidx = nls.vertv[idx]
        newxcoords[idx] = newxcoords[idx]   + amplification*(cp[uidx] + Uxx*nls.vertx[idx] + Uxy*(nls.verty[idx]-1))
        newycoords[idx] = newycoords[idx] + amplification*(cp[vidx] + Uxy*nls.vertx[idx] + Uyy*(nls.verty[idx]-1))
    plt.figure()
    plt.scatter(newxcoords,newycoords)
    plt.axis('equal')
    plt.savefig(name+'.png',dpi=1000)
    plt.show()



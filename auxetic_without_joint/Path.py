import numpy as np

class Path(object):
    '''
    classdocs
    '''
    
    def __init__(self, 
                 initial_point,
                 nonlinear_system,
                 group = None,
                 rep = None,
                 initial_direction = None
                 ):
        '''
        Constructor
        '''
        
        self.length = 0.0
        self.group = group
        #assert initial_point is symbfb.Equilibrium_Point, \
        #"%r initial_point expected to be Equilibrium_Point" % self
        self.eq_pts_raw= [initial_point]
        self.eq_pts = [nonlinear_system.postprocesssolution(initial_point)]
        self.nonlinear_system = nonlinear_system
        self.nonlinear_system.set_last_solution(initial_point)
        self.basis = None
        if group is not None:
            assert rep is not None, "%r set group but not rep" % self
            self.basis = group.compute_path_basis(rep)
        self.resid_tracker = []
        if initial_direction is not None:
            self.initial_direction = initial_direction.reshape(-1,1)
        else:
            self.initial_direction = None
        self.projector = None
            
    def project(self,x):
        if self.basis is None:
            return x
        else:
            if (self.projector is None) and (self.basis is not None):
                fpbasis = np.array(self.basis).astype(np.float64)
                
                fpbasistemp = np.vstack((fpbasis,np.zeros((4,fpbasis.shape[1]))))
                last_vector = np.zeros((fpbasistemp.shape[0],4))
                #last_vector[-1:,0,None] = 1
                last_vector[-1,3] = 1
                last_vector[-2,2] = 1
                last_vector[-3,1] = 1
                last_vector[-4,0] = 1
                fpbasistemp = np.hstack((fpbasistemp,last_vector))
                print('fpbasisshape',fpbasistemp.shape)
                self.projector = np.dot(fpbasistemp,fpbasistemp.T)
                print('Projector shape',self.projector.shape)

            return np.dot(self.projector,x)
        

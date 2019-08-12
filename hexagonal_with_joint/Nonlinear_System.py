from abc import ABCMeta, abstractmethod

class Nonlinear_System(object,metaclass=ABCMeta):
    def __init__(self,N):
        '''
        Constructor
        '''
        self.N = N
        
    def set_x(self,x):
        self.set_config(x[:-1,:])
        self.set_param(x[-1,0])
        
    @abstractmethod
    def set_config(self,x):
        pass
    
    @abstractmethod
    def set_param(self,param):
        pass
    
    @abstractmethod
    def get_x(self):
        pass
    
    @abstractmethod
    def get_F(self):
        pass
    
    @abstractmethod
    def get_J(self):
        pass
    
    @abstractmethod
    def get_J_lambda(self):
        pass

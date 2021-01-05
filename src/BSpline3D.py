'''
Created on Jan 4, 2021

@author: dlezcan1
'''

from scipy.interpolate import BPoly
import numpy as np


class BSpline3D( object ):
    '''
    3-D Bspline curve representation
    '''

    def __init__( self, order: int, coeffs:np.ndarray = None ):
        '''
        Constructor
        '''
        self.order = order
        
        self.coeffs = coeffs
        
        self.basis = self.bern_basis( order )
        
    # __init__
    
    @property
    def coeffs( self ):
        return self._coeffs
    
    # coeffs
    
    @coeffs.setter
    def coeffs( self, c ):
        
        if c is None:
            self._coeffs = np.eye(self.order)
            
        elif ( c.shape[0] != self.order ) and ( c.shape[1] != self.order ):
            self._coeffs = np.eye( self.order )
            
        else:
            self._coeffs = c
            
    # coeffs:setter
    
    @staticmethod
    def bern_basis( order ):
        C = np.eye( order )
        
        basis = []
        for i in range( order ):
            coeff = C[i,:].reshape( -1, 1 )
            basis.append( BPoly( coeff, [0, 1] ) )
        
        # for
    
        return basis
        
    # bern_basis
    
    def __call__( self, s, der:int = 0 ):
        A = []
        for i in range( self.order ):
            A.append( self.basis[i]( s , nu = der ) )
            
        # for
        
        A = np.array( A )  # numpify A
        
        return self.coeffs @ A
    
    # __call__

# class:BSpline3D        
        

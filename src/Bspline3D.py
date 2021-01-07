from scipy.interpolate import BPoly
import numpy as np


class Bspline3D():
    ''' 3-D BSpline fitting '''
    
    def __init__( self, pts = None, order:int = 3, qmin = None, qmax = None ):
        # pts
        self.pts = pts
        
        # order
        assert( order > 0 and isinstance( order, int ) )
        self.order = order
        
        # qmin
        if qmin is None:
            self.qmin = np.min( pts, axis = 0 )

        else:
            self.qmin = qmin

        # qmax
        if qmax is None:
            self.qmax = np.max( pts, axis = 0 )
        
        else:
            self.qmax = qmax
            
        assert( self.qmax > self.qmin )
    
    # __init__
    
    def _generate_berntensor2( X, qmin, qmax, order: int ):
        """Function to generatea tensor of the 3-D Bernstein functions where 
           F_ijk = b_i(x)*b_j(y)*b_k(z).
        
           @author: Dimitri Lezcano
           
           @param X:     the input to be used for the bernstein tensor
           
           @param qmin:  the minimum value for scaling
           
           @param qmax:  the maximum value for scaling
           
           @param order: the order of which you would like the 
                         the Bernstein polynomials to be.
                         
           @return: a numpy array of the Bernstein tensor
           
        """
        bern_basis = _generate_Bpoly_basis( order )
        
        X_prime = _scale_to_box( X, qmin, qmax )[0]
        if X.ndim > 1:
            X_px = X_prime[:, 0].reshape( ( -1, 1 ) )
            X_py = X_prime[:, 1].reshape( ( -1, 1 ) )
            X_pz = X_prime[:, 2].reshape( ( -1, 1 ) )
            bern_matrix = np.zeros( ( len( X ), ( order + 1 ) ** 3 ) )
        
        # if
        
        else:
            X_px, X_py, X_pz = X_prime
            bern_matrix = np.zeros( ( 1, ( order + 1 ) ** 3 ) )
        
        # else
        
        bern_ijk = lambda i, j, k: ( ( bern_basis[i]( X_px ) ) * ( bern_basis[j]( X_py ) ) * 
                                     ( bern_basis[k]( X_pz ) ) )
        
        for i in range( order + 1 ):
            for j in range( order + 1 ):
                for k in range( order + 1 ):
                    val = bern_ijk( i, j, k )
                    val = val.reshape( -1 )
                    bern_matrix[:, i * ( order + 1 ) ** 2 + j * ( order + 1 ) + k] = val
                    
                # for
            # for
        # for
        
        return bern_matrix
    
    # _generate_berntensor2
    
    def _generate_berntensor( N, order ):
        bern_basis = generate_Bpoly_basis( order )
        
        s = np.linspace(0, 1, N)
        
        
    # _generate_berntensor
    
    def _generate_Bpoly_basis( N: int ):
        """This function is to generate a basis of the bernstein polynomial basis
        
           @author: Dimitri Lezcano
           
           @param N: an integer representing the highest order or the 
                     Bernstein polynomial.
        
           @return:  A list of Bernstein polynomial objects of size N, that will 
                     individually be B_0,n, B_1,n, ..., B_n,n
                    
        """
        zeros = np.zeros( N + 1 )
        
        x_break = [0, 1]
        basis = []
        
        for i in range( N + 1 ):
            c = np.copy( zeros )
            c[i] = 1
            c = c.reshape( ( -1, 1 ) )
            basis.append( BPoly( c, x_break ) )
            
        # for 
        
        return basis
    
    # _generate_Bpoly_basis
    
    def _scale_to_box( X, qmin, qmax ):
        """A Function to scale an input array of vectors and return 
           the scaled version from 0 to 1.
           
           @author Dimitri Lezcano
           
           @parap X:    a numpy array where the rows are the corresponding vectors
                        to be scaled.
                     
           @param qmin: a value that represents the minimum value for scaling
           
           @param qmax: a value that represents the maximum value for scaling
         
           @return: X', the scaled vectors given from the function.
           
        """
        div = np.abs( qmax - qmin )
        
        X_prime = ( X - qmin ) / div  # normalized input
        
        return X_prime, qmin, qmax

    # _scale_to_box
    
    def fit( self, order:int = None ):
        if not order is None:
            self.order = order
        
        # generate the bernstein polynomial tensor    
        bern_tensor = _generate_berntensor( self.pts, self.qmin, self.qmax, self.order )
        
    # fit
    
# class:Bspline3D

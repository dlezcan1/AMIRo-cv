'''
Created on Jan 4, 2021

@author: dlezcan1
'''

from scipy.interpolate import BPoly
from scipy.spatial.transform import Rotation
import numpy as np
import open3d as o3d

from typing import (
    List,
)


class BSplineND( object ):
    '''
    N-D Bspline curve representation
    '''

    def __init__( self, order: int, coeffs:np.ndarray = None ):
        '''
        Constructor
        '''
        self.order = order

        self.coeffs = coeffs

        self.basis: List[BPoly] = self.bern_basis( order )

        # data values
        self._x    = None
        self._y    = None

        self.qmin = None
        self.qmax = None

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

        basis = [
            BPoly(
                C[i, :].reshape(-1, 1),
                [0, 1]
            )
            for i in range(order)
        ]

        return basis

    # bern_basis

    def __call__( self, s, der:int = 0 ):
        return self.eval_unscale(self._scale(s), der=der)

    # __call__

    def eval_unscale(self, s, der: int=0):
        """ Evaluate with already unscaled"""
        A = np.stack(
            [
                basis(s, nu=der)
                for basis in self.basis
            ],
            axis=1
        )

        return A @ self.coeffs

    # eval_unscale

    def _scale(self, data: np.ndarray):
        if self.qmin == self.qmax:
            return np.ones_like(data)

        retval = (data - self.qmin) / (self.qmax - self.qmin)

        if np.any(retval > 1) or np.any(retval < 0):
            pass

        return retval

    # _scale

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, order:int = 2):
        assert x.ndim == 1
        assert x.shape[0] == y.shape[0]

        poly = cls(order=order)

        poly._x = x
        poly._y = y

        poly.qmin = x.min()
        poly.qmax = x.max()

        A = np.stack(
            [
                basis(poly._scale(x), nu=0)
                for basis in poly.basis
            ],
            axis=1
        ) # ( N, order )

        coeffs, *_ = np.linalg.lstsq(A, y, rcond=None) # (order, y.shape[1])
        poly.coeffs = coeffs

        return poly

    # fit


# class:BSpline3D

def icp(
    a: np.ndarray,
    b: np.ndarray,
    max_correspondence_distance: float = 0.001,
    **kwargs
):
    """ Perform iterative closest point using Open3D library (does not assume point correspondence)

    Args:
        a                          : point cloud A
        b                          : point cloud B
        max_correspondence_distance: float of the maximum allowed correspondence points-pair distance

    Returns:
        (4, 4) SE3 Matrix for solutoin to tf @ a = b
    
    """
    # TODO
    A = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(a))
    B = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(b))

    results = o3d.pipelines.registration.registration_icp(
        A,
        B,
        max_correspondence_distance,
        **kwargs
    )

    return results.transformation

# icp


def point_cloud_registration(
        a: np.ndarray,
        b: np.ndarray,
        rotation_about_idx: int = None
    ):
    """ Point cloud registration (Assumes point correspondence handled)

        Transforms a -> b using tf @ a = b

        Args:
            a: point cloud A
            b: point cloud B
            rotation_about_idx: int of which point to rotate about

        Returns:
            (4, 4) SE3 matrix for solution to tf @ a = b

    """
    assert a.ndim == 2
    assert b.ndim == 2
    assert a.shape == b.shape

    # center each to solve for rotation
    if rotation_about_idx is not None:
        a_hat = a - a[rotation_about_idx]
        b_hat = b - b[rotation_about_idx]

    # if
    else:
        a_hat: np.ndarray = a - np.mean(a, axis=0, keepdims=True)
        b_hat: np.ndarray = b - np.mean(b, axis=0, keepdims=True)

    # else

    # determine rotation
    H = a_hat.T @ b_hat
    delta_skew = H.T - H
    delta = np.array([
        -delta_skew[1, 2],
         delta_skew[0, 2],
        -delta_skew[0, 1],
    ])

    G         = np.zeros((4, 4), dtype=H.dtype)
    G[0, 0]   = np.trace(H)
    G[0, 1:]  = G[1:, 0] = delta
    G[1:, 1:] = H + H.T - np.trace(H) * np.eye(3)

    evals, evecs = np.linalg.eig(G)
    idx_min      = np.argmax(evals)
    quat         = evecs[:, idx_min]

    tf = np.eye(4, dtype=H.dtype)
    if np.linalg.norm(quat) != 0:
        quat = quat[[1, 2, 3, 0]]
        quat /= np.linalg.norm(quat)
        tf[:3, :3] = Rotation.from_quat(quat).as_matrix()

    # if

    tf[:3, -1] = np.mean(b, axis=0) - tf[:3, :3] @ np.mean(a, axis=0)

    return tf

# point_cloud_registration

def point_cloud_registation_ransac(
    a: np.ndarray,
    b: np.ndarray,
    num_samples: int = None,
    num_trials: int = 3_000,
    inlier_threshold: float = 0.1,
    break_on_all_inliers: bool = True,
    random_seed: int = None,
):
    """ Perform Point cloud registration using RANSAC (no correspondences)    

        Args:
            a: point cloud to be transformed
            b: point cloud 'a' will be transformed to
            num_samples: The number of samples used to pair with each of the point clouds
            inlier_threshold: float of the L2 distance required to determine if inlier or not
            break_on_all_inliers: True means once all points are inliers, algorithm will stop, otherwise will continue to refine
            random_seed: provide random seeding for consistent results
        Returns:
            tf_best: best transform computed from RANSAC
            most_inliers: the number of inliers detected
            best_error: the best computed L2 error
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # parse inputs
    N           = a.shape[0]
    num_samples = num_samples if num_samples is not None else N

    # begin ransac algorithm
    tf_best      = np.eye(4)
    most_inliers = 0
    best_error   = np.inf
    for _ in range(num_trials):
        # randomly choose pairs
        choices_a = np.random.permutation(N)[:num_samples]
        choices_b = np.sort(np.random.permutation(N)[:num_samples])

        a_test = a[choices_a]
        b_test = b[choices_b]

        tf_test = point_cloud_registration(
            a_test,
            b_test,
            rotation_about_idx=None,
        )
        a_tf = transformSE3(a, tf_test)
        
        errors      = np.linalg.norm(a_tf - b, ord=2, axis=1)
        mean_error  = np.mean(errors)
        num_inliers = np.sum(errors <= inlier_threshold)

        if num_inliers > most_inliers:
            tf_best      = tf_test
            most_inliers = num_inliers
            best_error   = mean_error

            if break_on_all_inliers and (num_inliers == N):
                break

            # if
        # 
        elif (num_inliers == most_inliers) and (mean_error < best_error):
            tf_best      = tf_test
            most_inliers = num_inliers
            best_error   = mean_error

        # elif
    # for

    return tf_best, most_inliers, best_error

# point_cloud_registation_ransac

def transformSE3(points: np.ndarray, *args):
    """ Transform 3D points using SE3 rigid transforms
    
        Args:
            points: (N, 3/4) points array
            pose: (4, 4) SE3 array       (if 1 argument provided)
            R: (3, 3) Rotation matrix    (if 2 arguments provided)
            t: (3,)   translation vector (if 2 arguments provided)

        Returns:
            transformed points (N, 3/4) same size as input points
    
    """
    if len(args) == 1:
        F = args[0]

    # if
    elif len(args) == 2:
        F = np.eye(4)
        F[:3, :3] = args[0]
        F[:3, 3]  = args[1]

    # elif
    else:
        raise ValueError("Either provided a 4x4 matrix, or a 3x3 and a 3x1 set of rotation matrix and translation vector")
    
    # else
    
    points_h = points.copy()
    if points_h.shape[1] == 3:
        points_h = np.hstack(
            (
                points_h,
                np.ones((points.shape[0], 1), dtype=points.dtype)
            )
        )

    # if

    points_tf_h = points_h @ F.T

    if points.shape[1] == 3:
        points_tf_h = points_tf_h[:, :3]

    return points_tf_h

# transformSE3
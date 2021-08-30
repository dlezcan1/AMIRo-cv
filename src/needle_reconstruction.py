"""
Created on Aug 26, 2021

This is a library/script to perform stereo needle reconstruciton and process datasets


@author: Dimitri Lezcano

"""

import stereo_needle_proc as stereo_needle

import numpy as np
import cv2 as cv
from abc import ABC, abstractmethod

from collections import namedtuple

StereoPair = namedtuple( 'StereoPair', [ 'left', 'right' ] )  # stereo pair of images


class StereoNeedleReconstruction( ABC ):  # TODO: stereo needle reconstruction abstract class
    """ Basic class for stereo needle reconstruction"""

    def __init__( self, stereo_params: dict, img_left: np.ndarray, img_right: np.ndarray ):
        self.stereo_params = stereo_params
        self.images = StereoPair( img_left, img_right )
        self.left = img_left
        self.right = img_right
        self.needle_shape = None

    # __init__

    @abstractmethod
    def reconstruct_needle( self ) -> np.ndarray:
        """
            Reconstruct the 3D needle shape from the left and right image pair

        """
        pass

    # reconstruct_needle


# class: StereoNeedleReconstruction

class StereoNeedleRefReconstruction( StereoNeedleReconstruction ):
    """ Class for Needle Image Reference Reconstruction """

    def __init__( self, stereo_params: dict, img_left: np.ndarray, img_right: np.ndarray, ref_left: np.ndarray,
                  ref_right: np.ndarray ):
        super().__init__( stereo_params, img_left, img_right )
        self.ref_pair = StereoPair( ref_left, ref_right )
        self.ref_left = ref_left
        self.ref_right = ref_right

    # __init__

    def reconstruct_needle( self ) -> np.ndarray:
        # TODO: needle reconstruction reference imaging
        pass

    # reconstruct_needle


# class:StereoNeedleRefReconstruction

class StereoInsertionExperiment:  # TODO: Insertion experiment for stereo reconstructions
    def __init__( self, stereo_param_file: str ):
        self.stereo_params = stereo_needle.load_stereoparams_matlab( stereo_param_file )
        pass

    # __init__

    @staticmethod
    def configure_dataset():
        pass

    # configure_dataset


# class: StereoInsertionExperiment


def main():
    pass


# main

if __name__ == "__main__":
    main()

# if __main__

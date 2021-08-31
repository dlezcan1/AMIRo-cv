"""
Created on Aug 26, 2021

This is a library/script to perform stereo needle reconstruciton and process datasets


@author: Dimitri Lezcano

"""
import argparse
import glob
import os
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Union, List

import cv2 as cv
import numpy as np

# import stereo_needle_proc as stereo_needle
from stereo_needle_proc import load_stereoparams_matlab, needle_reconstruction_ref


@dataclass
class ImageROI:
    image: np.ndarray
    roi: List[ int ] = field( default_factory=list )
    blackout: List[ int ] = field( default_factory=list )


# dataclass: ImageROI

@dataclass
class StereoPair:
    left: Union[ np.ndarray, list ] = None
    right: Union[ np.ndarray, list ] = None


# dataclass: StereoPair

@dataclass
class StereoImagePair:
    left: ImageROI
    right: ImageROI


# dataclass: StereoImagePair

class StereoRefInsertionExperiment:
    directory_pattern = r".*[/,\\]Insertion([0-9]+)[/,\\]([0-9]+).*"  # data directory pattern

    def __init__( self, stereo_param_file: str, data_dir: str, insertion_depths: list,
                  roi: tuple = None, blackout: tuple = None ):
        self.stereo_params = load_stereoparams_matlab( stereo_param_file )
        self.data_directory = os.path.normpath( data_dir )
        self.insertion_depths = [ depth for depth in insertion_depths if
                                  depth > 0 ] + [ 0 ]  # make sure non-negative insertion depth

        self._needle_reconstructor = StereoNeedleRefReconstruction( self.stereo_params, None, None, None, None )

        # set the datasets ROI
        if roi is not None:
            self._needle_reconstructor.roi.left = roi[ 0 ]
            self._needle_reconstructor.roi.right = roi[ 1 ]

        # if

        # set the image blackout regions
        if blackout is not None:
            self._needle_reconstructor.blackout.left = blackout[ 0 ]
            self._needle_reconstructor.blackout.right = blackout[ 1 ]

        # if

        # configure the dataset
        self.dataset, self.processed_data = self.configure_dataset( self.data_directory, self.insertion_depths )

    # __init__

    @classmethod
    def configure_dataset( cls, directory: str, insertion_depths: list ) -> (list, list):
        """
            Configure a dataset based on the directory:

            :param directory: string of the main data directory
            :param insertion_depths: a list of insertion depths that are to be processed.

        """
        dataset = [ ]
        processed_dataset = [ ]

        if directory is None:
            return dataset

        # if

        directories = glob.glob( os.path.join( directory, 'Insertion*/*/' ) )

        # iterate over the potential directories
        for d in directories:
            res = re.search( cls.directory_pattern, d )

            if res is not None:
                insertion_hole, insertion_depth = res.groups()
                insertion_hole = int( insertion_hole )
                insertion_depth = float( insertion_depth )

                # only include insertion depths that we want to process
                if insertion_depth in insertion_depths:
                    dataset.append( (d, insertion_hole, insertion_depth) )

                # if

                if os.path.isfile( os.path.join( d, 'left-right_3d-pts.csv' ) ):
                    # load the processed data
                    pts_3d = np.loadtxt( os.path.join( d, 'left-right_3d-pts.csv' ), delimiter=',' )
                    processed_dataset.append( (d, insertion_hole, insertion_depth, pts_3d) )  # processed_dataset

                # if

            # if
        # for

        return dataset, processed_dataset

    # configure_dataset

    def process_dataset( self, dataset: list, save: bool = True, overwrite: bool = False, **kwargs ):
        """ Process the dataset

            :param dataset:   List of the data (data_dir, insertion_hole, insertion_depth
            :param save:      (Default = True) whether to save the processed data or not
            :param overwrite: (Default = False) whether to overwrite already processed datasets
            :param kwargs:    the keyword arguments to pass to StereoNeedleRefReconstruction.reconstruct_needle
        """
        # iterate over the insertion holes
        insertion_holes = set( [ data[ 1 ] for data in dataset ] )

        for insertion_hole in insertion_holes:
            # grab all of the relevant insertion holes
            dataset_hole = filter( lambda data: (data[ 1 ] == insertion_hole) and (data[ 2 ] > 0), dataset )
            dataset_hole_ref = \
                list( filter( lambda data: (data[ 1 ] == insertion_hole) and (data[ 2 ] == 0), dataset ) )[ 0 ]

            # load the reference images
            left_ref_file = os.path.join( dataset_hole_ref[ 0 ], 'left.png' )
            right_ref_file = os.path.join( dataset_hole_ref[ 0 ], 'right.png' )
            self._needle_reconstructor.load_image_pair( left_ref_file, right_ref_file, reference=True )

            # iterate over the datasets
            for d, _, insertion_depth in dataset_hole:
                # see if the data has been processed already
                idx = np.argwhere( list(
                        map(
                                lambda row: all(
                                        (row[ 0 ] == d, row[ 1 ] == insertion_hole, row[ 2 ] == insertion_depth) ),
                                self.processed_data ) ) ).flatten()

                # check if data is already processed
                if len( idx ) > 0 and not overwrite:
                    continue

                # if

                # load the next image pair
                left_file = os.path.join( d, 'left.png' )
                right_file = os.path.join( d, 'right.png' )
                self._needle_reconstructor.load_image_pair( left_file, right_file, reference=False )

                # perform the 3D reconstruction
                pts_3d = self._needle_reconstructor.reconstruct_needle( **kwargs )

                # add to the processed dataset

                if len( idx ) == 0:
                    self.processed_data.append( (d, insertion_hole, insertion_depth, pts_3d) )

                # if
                elif overwrite:  # overwrite the dataset
                    self.processed_data[ idx[ 0 ] ] = (d, insertion_hole, insertion_depth, pts_3d)

                # else

                # save the data (if you would like to save it)
                if save:
                    self._needle_reconstructor.save_3dpoints( directory=d )
                    self._needle_reconstructor.save_processed_images( directory=d )

                # if
            # for
        # for

    # process_dataset


# class: StereoRefInsertionExperiment


class StereoNeedleReconstruction( ABC ):
    """ Basic class for stereo needle reconstruction"""
    save_fbase = 'left-right_{:s}'

    def __init__( self, stereo_params: dict, img_left: np.ndarray, img_right: np.ndarray ):
        self.stereo_params = stereo_params
        self.image = StereoPair( img_left, img_right )

        self.roi = StereoPair( [ ], [ ] )
        self.blackout = StereoPair( [ ], [ ] )

        self.needle_shape = None
        self.img_points = StereoPair( None, None )
        self.img_bspline = StereoPair( None, None )
        self.processed_images = None
        self.processed_figures = None

    # __init__

    def load_image_pair( self, left_file: str = None, right_file: str = None ):
        """ Load the image pair. If the one of the files is none, that image will not be loaded

            :param left_file: (Default = None) string of the left file to laod
            :param right_file: (Default = None) string of the right file to laoad

        """

        if left_file is not None:
            self.image.left = cv.imread( left_file, cv.IMREAD_COLOR )

        # if

        if right_file is not None:
            self.image.right = cv.imread( right_file, cv.IMREAD_COLOR )

        # if

    # load_image_pair

    @abstractmethod
    def reconstruct_needle( self, **kwargs ) -> np.ndarray:
        """
            Reconstruct the 3D needle shape from the left and right image pair

        """
        pass

    # reconstruct_needle

    def save_3dpoints( self, outfile: str = None, directory: str = '' ):
        """ Save the 3D reconstruction to a file """

        if self.needle_shape is not None:
            if outfile is None:
                outfile = self.save_fbase.format( '3d-pts' ) + '.csv'

            # if

            outfile = os.path.join( directory, outfile )

            np.savetxt( outfile, self.needle_shape, delimiter=',' )
            print( "Saved reconstructed shape:", outfile )

        # if

    # save_3dpoints

    def save_processed_images( self, directory: str = '.' ):
        """ Save the images that have now been processed

            :param directory: (Default = '.') string of the directory to save the processed images to.
        """
        # the format string for saving the figures
        save_fbase = os.path.join( directory, self.save_fbase )

        if self.processed_images is not None:
            for key, img in self.processed_images.items():
                cv.imwrite( save_fbase.format( key ) + '.png', img )
                print( "Saved figure:", save_fbase.format( key ) + '.png' )

            # for
        # if

        if self.processed_figures is not None:
            for key, fig in self.processed_figures.items():
                fig.savefig( save_fbase.format( key + '-fig' ) + '.png' )
                print( "Saved figure:", save_fbase.format( key + '-fig' ) + '.png' )

            # for
        # if

    # save_processed_images


# class: StereoNeedleReconstruction


class StereoNeedleRefReconstruction( StereoNeedleReconstruction ):
    """ Class for Needle Image Reference Reconstruction """

    def __init__( self, stereo_params: dict, img_left: np.ndarray, img_right: np.ndarray, ref_left: np.ndarray,
                  ref_right: np.ndarray ):
        super().__init__( stereo_params, img_left, img_right )
        self.reference = StereoPair( ref_left, ref_right )

    # __init__

    def load_image_pair( self, left_file: str = None, right_file: str = None, reference: bool = False ):
        """ Load the image pair. If the one of the files is none, that image will not be loaded

            :param left_file: (Default = None) string of the left file to laod
            :param right_file: (Default = None) string of the right file to laoad
            :param reference: (Default = False) whether we are loading the reference image or not
        """
        if not reference:
            super().load_image_pair( left_file, right_file )

        # if
        else:
            if left_file is not None:
                self.reference.left = cv.imread( left_file, cv.IMREAD_COLOR )

            # if

            if right_file is not None:
                self.reference.right = cv.imread( right_file, cv.IMREAD_COLOR )

            # if

        # else

    # load_image_pair

    def reconstruct_needle( self, **kwargs ) -> np.ndarray:
        """
            Reconstruct the needle shape

            Keyword arguments:
                window size: 2-tuple of for window size of the stereo template matching (must be odd)
                zoom:        the zoom value for for the template maching algorithm
                alpha:       the alpha parameter in stereo rectification
                sub_thresh:  the threshold value for the reference image subtraction

        """
        # keyword argument parsing
        if 'window_size' in kwargs.keys():
            window_size = kwargs[ 'window_size' ]

        # if
        else:
            window_size = (201, 51)

        # else

        if 'zoom' in kwargs.keys():
            zoom = kwargs[ 'zoom' ]

        # if
        else:
            zoom = 1.0

        # else

        if 'alpha' in kwargs.keys():
            alpha = kwargs[ 'alpha' ]

        # if
        else:
            alpha = 0.6

        # else

        if 'sub_thresh' in kwargs.keys():
            sub_thresh = kwargs[ 'sub_thresh' ]

        # if
        else:
            sub_thresh = 60

        # else

        if 'proc_show' in kwargs.keys():
            proc_show = kwargs[ 'proc_show' ]

        # if
        else:
            proc_show = False

        # else

        # perform stereo reconstruction
        pts_3d, pts_l, pts_r, bspline_l, bspline_r, imgs, figs = \
            needle_reconstruction_ref( self.image.left, self.reference.left,
                                       self.image.right, self.reference.right,
                                       stereo_params=self.stereo_params, recalc_stereo=True,
                                       bor_l=self.blackout.left, bor_r=self.blackout.right,
                                       roi_l=self.roi.left, roi_r=self.roi.right,
                                       alpha=alpha, winsize=window_size, zoom=zoom,
                                       sub_thresh=sub_thresh, proc_show=proc_show )
        # set the current fields
        self.needle_shape = pts_3d[ :, 0:3 ]  # remove 4-th axis

        self.img_points.left = pts_l
        self.img_points.right = pts_r

        self.img_bspline.left = pts_l
        self.img_bspline.right = pts_r

        self.processed_images = imgs
        self.processed_figures = figs

        return pts_3d[ :, 0:3 ]

        # reconstruct_needle


# class:StereoNeedleRefReconstruction

def __get_parser() -> argparse.ArgumentParser:
    """ Configure the argument parser"""
    parser = argparse.ArgumentParser(
            description='Perform 3D needle reconstruction of the needle insertion experiments.' )

    # stereo parameters
    parser.add_argument( 'stereoParamFile', type=str, help='Stereo Calibration parameter file' )

    # data directory 
    parser.add_argument( 'dataDirectory', type=str, help='Needle Insertion Experiment directory' )
    parser.add_argument( '--insertion-depths', type=float, nargs='+', default=[ 0, 105, 110, 115, 120 ],
                         help="The insertion depths of the needle to be parsed." )
    parser.add_argument( '--show-processed', action='store_true', help='Show the processed data' )
    parser.add_argument( '--save', action='store_true', help='Save the processed data or not' )
    parser.add_argument( '--force-overwrite', action='store_true', help='Overwrite previously processed data.' )

    # image region of interestes
    parser.add_argument( '--left-roi', nargs=4, type=int, default=[ ], help='The left image ROI to use',
                         metavar=('TOP_Y', 'TOP_X', 'BOTTOM_Y', 'BOTTOM_X') )
    parser.add_argument( '--right-roi', nargs=4, type=int, default=[ ], help='The right image ROI to use',
                         metavar=('TOP_Y', 'TOP_X', 'BOTTOM_Y', 'BOTTOM_X') )

    parser.add_argument( '--left-blackout', nargs='+', type=int, default=[ ],
                         help='The blackout regions for the left image' )
    parser.add_argument( '--right-blackout', nargs='+', type=int, default=[ ],
                         help='The blackout regions for the right image' )

    # reconstruction parameters
    parser.add_argument( '--zoom', type=float, default=1.0, help="The zoom for stereo template matching" )
    parser.add_argument( '--window-size', type=int, nargs=2, default=(201, 51), metavar=('WIDTH', 'HEIGHT'),
                         help='The window size for stereo template matching' )
    parser.add_argument( '--alpha', type=float, default=0.6, help='The alpha parameter for stereo rectification.' )
    parser.add_argument( '--subtract-thresh', type=float, default=60,
                         help='The threshold for reference image subtraction.' )

    return parser


# __get_parser

def main( args=None ):
    # parse the arguments
    parser = __get_parser()
    pargs = parser.parse_args( args )

    # image ROI parsing
    if len( pargs.left_roi ) > 0:
        left_roi = [ pargs.left_roi[ 0:2 ], pargs.left_roi[ 2:4 ] ]

    # if
    else:
        left_roi = [ ]

    # else

    if len( pargs.right_roi ) > 0:
        right_roi = [ pargs.right_roi[ 0:2 ], pargs.right_roi[ 2:4 ] ]

    # if
    else:
        right_roi = [ ]

    # else

    # image blackout region parsing
    if len( pargs.left_blackout ) > 0:
        assert (len( pargs.left_blackout ) % 4 == 0)  # check if there are adequate pairs
        left_blackout = [ ]
        for i in range( 0, len( pargs.left_blackout ), 4 ):
            left_blackout.append( [ [ pargs.left_blackout[ i:i + 2 ],
                                      [ pargs.left_blackout[ i + 2:i + 4 ] ] ] ] )

        # for

    # if
    else:
        left_blackout = [ ]

    # else

    if len( pargs.right_blackout ) > 0:
        assert (len( pargs.right_blackout ) % 4 == 0)  # check if there are adequate pairs
        right_blackout = [ ]
        for i in range( 0, len( pargs.right_blackout ), 4 ):
            right_blackout.append( [ [ pargs.right_blackout[ i:i + 2 ],
                                       [ pargs.right_blackout[ i + 2:i + 4 ] ] ] ] )

        # for

    # if
    else:
        right_blackout = [ ]

    # else

    # instantiate the Insertion Experiment data processor
    processor = StereoRefInsertionExperiment( pargs.stereoParamFile, pargs.dataDirectory,
                                              pargs.insertion_depths, roi=(left_roi, right_roi),
                                              blackout=(left_blackout, right_blackout) )

    # process the dataset
    stereo_kwargs = { 'zoom'       : pargs.zoom,
                      'window_size': pargs.window_size,
                      'alpha'      : pargs.alpha,
                      'sub_thresh' : pargs.subtract_thresh,
                      'proc_show'  : pargs.show_processed }
    processor.process_dataset( processor.dataset, save=pargs.save, overwrite=pargs.force_overwrite, **stereo_kwargs )

    print( "Program completed." )


# main

if __name__ == "__main__":
    main()

# if __main__

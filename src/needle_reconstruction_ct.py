import argparse as ap
from dataclasses import (
    dataclass,
    field,
)
import os
import json

from typing import (
    Any,
    Dict,
    List,
    Tuple,
    Union
)

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt
import pandas as pd
import skimage
from sklearn.cluster import (
    KMeans,
)

import dicom
from util import (
    BSplineND,
    icp,
    point_cloud_registration,
)
import stereo_needle_proc

@dataclass
class ROI3D:
    top_left    : List[int] = field(default_factory=lambda: [ 0,  0,  0])
    bottom_right: List[int] = field(default_factory=lambda: [-1, -1, -1])

    @property
    def tl(self):
        return self.top_left
    
    @tl.setter
    def tl(self, tl: List[int]):
        self.top_left = tl

    @property
    def br(self):
        return self.bottom_right
    
    @br.setter
    def br(self, br: List[int]):
        self.bottom_right = br

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        obj = cls()

        if "top_left" in d.keys():
            obj.top_left = d["top_left"]

        if "bottom_right" in d.keys():
            obj.bottom_right = d["bottom_right"]

        return obj
    
    # from_dict

    def get_mask(self, shape: Tuple[int], invert: bool = False):
        mask = np.zeros(shape, dtype=bool)

        mask[
            self.tl[0]:self.br[0],
            self.tl[1]:self.br[1],
            self.tl[2]:self.br[2],
        ] = True

        if invert:
            mask = np.logical_not(mask)

        return mask
    
    # get_mask
    
    def to_dict(self):
        return {
            "top_left"    : self.top_left,
            "bottom_right": self.bottom_right,
        }
    
    # to_dict

# dataclass: ROI3D

@dataclass
class CTNeedleReconstructionOptions:
    # image properties
    # - region of interests
    roi           : ROI3D = field( default_factory=ROI3D )
    roi_needle    : ROI3D = field( default_factory=ROI3D )
    roi_fiducials : ROI3D = field( default_factory=ROI3D )

    # - blackout regions
    blackout_regions          : List[ROI3D] = field( default_factory=list )
    blackout_regions_needle   : List[ROI3D] = field( default_factory=list )
    blackout_regions_fiducials: List[ROI3D] = field( default_factory=list )

    # segmentation properties
    threshold: int = 10_000

    # interpolation options
    bspline_order   : int   = -1
    interpolation_ds: float = 0.5

    # experiment options
    fiducial_locations: npt.NDArray[np.float64] = None

    @property
    def num_fiducials(self):
        return self.fiducial_locations.shape[0]
    
    # num_fiducials

    @classmethod
    def _add_arguments_to_parser(cls, parser: ap.ArgumentParser):
        # ROIs
        for roi_name in [
            "",
            "-needle",
            "-fiducials",
        ]:
            parser.add_argument(
                f"--roi{roi_name}",
                nargs=4,
                type=float,
                required=False,
                default=None,
                metavar=(
                    "TOP-LEFT X", 
                    "TOP-LEFT Y",
                    "BOTTOM-RIGHT X",
                    "BOTTOM-RIGHT Y",
                ),
            )

        # for

        # blackout regions
        for bor_name in [
            "",
            "-needle",
            "-fiducials",
        ]:
            parser.add_argument(
                f"--blackout-region{bor_name}",
                nargs='+',
                type=float,
                required=False,
                default=None,
                metavar=[
                    "TOP-LEFT X", 
                    "TOP-LEFT Y",
                    "BOTTOM-RIGHT X",
                    "BOTTOM-RIGHT Y",
                ],
            )

        # for

        parser.add_argument(
            "--threshold",
            type=int,
            required=False,
            default=None,
            help="The threshold for segmenting the needle from the background of CT scan"
        )

        parser.add_argument(
            "--bspline-order",
            type=int,
            default=None,
            required=False,
            help="The interpolation polynomial order used for interpolating the 3D needle shape",
        )

        parser.add_argument(
            "--interpolation-ds",
            type=float,
            default=None,
            required=False,
            help="The interpolation arclength increments to use."
        )

        parser.add_argument(
            "--fiducial-locations",
            nargs='+',
            default=None,
            required=False,
            help="The locations of the fiducials in their own coordinate system",
            metavar=["X", "Y", "Z"],
        )
        
    # _add_arguments_to_parser

    @classmethod
    def from_dict(cls, d: Dict[str, Any]):
        obj = cls()

        # set region of interestes
        for roi_key in [
            "roi", 
            "roi_needle", 
            "roi_fiducials",
        ]:
            setattr(obj, roi_key, ROI3D.from_dict(d.get(roi_key, dict())))

        # set blackout regions
        for bor_key in [
            "blackout_regions",
            "blackout_regions_needle",
            "blackout_regions_fiducials",
        ]:
            setattr(
                obj,
                bor_key,
                list( map( ROI3D.from_dict, d.get( bor_key, list() ) ) )
            )

        # for

        # set simple elements
        for key in [
            "threshold", 
            "bspline_order",
            "interpolation_ds",
        ]:
            if key in d.keys():
                setattr(obj, key, d[key])

            # if
        # for

        # set numpy arrays
        for np_key in [
            "fiducial_locations",
        ]:
            if np_key in d.keys():
                setattr( obj, np_key, np.asarray(d[np_key]) )

            # if
        # for


        return obj
    
    # from_dict

    @classmethod
    def from_json(cls, jsonfile: str):
        with open(jsonfile, 'r') as file:
            data_dict = json.load(file)

        # with

        return cls.from_dict(data_dict)
    
    # from_json

    @classmethod
    def from_parsed_arguments(cls, args: ap.Namespace):
        args_kwargs: dict = dict(args._get_kwargs())

        opts = cls()

        # ROIs
        for roi_name in [
            "",
            "_needle",
            "_fiducials",
        ]:
            roi_bnds = args_kwargs.get(f"roi{roi_name}", None)
            if roi_bnds is not None:
                roi = ROI3D(
                    top_left=roi_bnds[:2],
                    bottom_right=roi_bnds[2:],
                )
                setattr(opts, f"roi{roi_name}", roi)

            # if
        # for

        # blackout regions
        for bor_name in [
            "",
            "_needle",
            "_fiducials",
        ]:
            bor_bounds = args_kwargs.get(f"blackout_regions{bor_name}", None)
            if bor_bounds is not None:
                assert len(bor_bounds) % 4 == 0

                for i in range(0, len(bor_bounds), 4):
                    bor_bnd = bor_bounds[i:i+4]
                    
                    getattr(opts, f"blackout_regions{bor_name}").append(
                        ROI3D(
                            top_left=bor_bnd[:2],
                            bottom_right=bor_bnd[2:],
                        )
                    )
                # for
            # if


        # for

        for kw in [
            "threshold",
            "bspline_order",
            "interpolation_ds",
        ]:
            if args_kwargs.get(kw, None) is not None:
                setattr(opts, kw, args_kwargs[kw])

            # if
        # for

        if args_kwargs.get("fiducial_locations", None) is not None:
            opts.fiducial_locations = np.reshape(
                args_kwargs["fiducial_locations"],
                (-1, 3),
            )

        # if


        return opts
    
    # from_parsed_arguments

    def save(self, outfile: str):
        """ Save the options to a json file"""
        with open(outfile, 'w') as writer:
            json.dump(
                self.to_dict(),
                writer,
                indent=4,
            )

        # with

    # save 

    def to_dict(self):
        return {
            "roi"                       : self.roi.to_dict(),
            "roi_needle"                : self.roi_needle.to_dict(),
            "roi_fiducials"             : self.roi_fiducials.to_dict(),
            "blackout_regions"          : [ bor.to_dict() for bor in self.blackout_regions ],
            "blackout_regions_needle"   : [ bor.to_dict() for bor in self.blackout_regions_needle ],
            "blackout_regions_fiducials": [ bor.to_dict() for bor in self.blackout_regions_fiducials ],
            "threshold"                 : self.threshold,
            "bspline_order"             : self.bspline_order,
            "interpolation_ds"          : self.interpolation_ds,
            "fiducial_locations"        : self.fiducial_locations.tolist() if self.fiducial_locations is not None else None,
        }

    # to_dict

# dataclass: CTNeedleReconstructionOptions

class CTNeedleReconstruction:
    def __init__(self, options: CTNeedleReconstructionOptions = None) -> None:
        self.dicom_image3d: dicom.Image3D = None
        
        self.options = options if options is not None else CTNeedleReconstructionOptions()


        # post-processed run
        self.needle_shape      : np.ndarray = None
        self.fiducial_locations: np.ndarray = None
        self.fiducial_pose     : np.ndarray = None


    # __init__

    def _reset_results(self):
        self.needle_shape       = None
        self.fiducial_locations = None
        self.fiducial_pose      = None

    # _reset_data

    def determine_fiducial_locations(
        self, 
        mask: npt.NDArray[np.bool8],
    ) -> npt.NDArray[np.float64]:
        pts_fiducials = (
            np.argwhere(mask).astype(np.float_)
            * self.dicom_image3d.image_axis_scaling.reshape(1, -1)
        )

        clf_ = KMeans(
            n_clusters=self.options.num_fiducials,
            init='k-means++',
            n_init='auto',
        )
        clf_.fit(pts_fiducials)
        fiducial_locations = clf_.cluster_centers_

        return fiducial_locations

    # determine_fiducial_locations

    def load_ct_scan(self, npz_file: str) -> dicom.Image3D:
        self.dicom_image3d = dicom.Image3D.from_npz_file(npz_file)

        # reset the processed results
        self._reset_results()

        return self.dicom_image3d
    
    # load_ct_scan

    def load_options(self, opts_json_file: str) -> CTNeedleReconstructionOptions:
        opts = CTNeedleReconstructionOptions.from_json(opts_json_file)

        self.options = opts

        return self.options
    
    # load_options

    def reconstruct_scene(self, **kwargs) -> Union[
        Tuple[npt.NDArray, npt.NDArray, npt.NDArray],
        Tuple[npt.NDArray, npt.NDArray, npt.NDArray, Dict[str, npt.NDArray]]
    ]:
        """ Reconstructs the CT scene
            
            kwargs:
                debug: return the debug images used for processing

            Returns:
                - reconstructed needle shape (N, 3)
                - 3D locations of the fiducials (in CT coordinates) (M, 3)
                - Pose of CT coordinates to home fiducial locations (4, 4)
                - (optional if debug=True) dictionary of images used in the processing
        
        """
        assert self.dicom_image3d is not None, "You need to load the dicom .npz CT scan files!"

        debug        = kwargs.get("debug", False)
        debug_images = dict()

        # get the masks required for processing
        thresh_mask = self.dicom_image3d.image >= self.options.threshold
        roi_mask    = self.options.roi.get_mask(
            thresh_mask.shape,
            invert=False
        )
        bo_mask     = np.ones_like(roi_mask)
        for bor in self.options.blackout_regions:
            bo_mask &= bor.get_mask(
                thresh_mask.shape,
                invert=True
            )

        # for

        seg_mask = thresh_mask & roi_mask & bo_mask
        
        debug_images["image_thresh"]     = thresh_mask
        debug_images["image_thresh_roi"] = seg_mask

        # segment the needle
        seg_needle_mask = seg_mask & self.options.roi_needle.get_mask(seg_mask.shape, invert=False)
        for bor in self.options.blackout_regions_needle:
            seg_needle_mask &= bor.get_mask(
                seg_needle_mask.shape,
                invert=True,
            )

        # for
        debug_images["needle_image_roi"] = seg_needle_mask

        # - connected component analysis
        lbl_needle_mask, num_labels = skimage.measure.label(
            seg_needle_mask,
            connectivity=kwargs.get("connected_components_connectivity", 1),
            return_num=True,
        )
        lbl_max = np.argmax(
            [
                np.sum(lbl_needle_mask == lbl)
                for lbl in range(1, num_labels)
            ]
        ) + 1
        seg_needle_mask = lbl_needle_mask == lbl_max
        debug_images["needle_image_roi_conncomp"] = seg_needle_mask

        seg_needle_skel_mask = skimage.morphology.skeletonize_3d(seg_needle_mask)
        debug_images["needle_image_roi_skeleton"] = seg_needle_skel_mask

        # segment the fiducials
        seg_fiducials_mask = seg_mask & self.options.roi_fiducials.get_mask(seg_mask.shape, invert=False)
        seg_fiducials_mask &= np.logical_not(seg_needle_mask)
        for bor in self.options.blackout_regions_fiducials:
            seg_fiducials_mask &= bor.get_mask(
                seg_fiducials_mask.shape,
                invert=True,
            )

        # for
        debug_images["fiducials_image_roi"] = seg_fiducials_mask

        # get CT fiducial locations
        self.fiducial_locations = self.determine_fiducial_locations(
            mask=seg_fiducials_mask,
        )

        #  get pose of fiducial
        fiducial_pose_init = point_cloud_registration(
            self.options.fiducial_locations,
            self.fiducial_locations,
            rotation_about_idx=None,
        )
        self.fiducial_pose = icp(
            self.options.fiducial_locations, 
            self.fiducial_locations,
            max_correspondence_distance=1e-2, # mm
            init=fiducial_pose_init,
        )

        # get the needle shape points
        needle_idx_pts = np.argwhere(seg_needle_skel_mask)  # (N, 3)
        needle_pts     = (
            needle_idx_pts.astype(np.float64)
            * self.dicom_image3d.image_axis_scaling.reshape(1, -1)
        )

        sort_idx   = np.argsort(needle_pts[:, 2])
        needle_pts = needle_pts[sort_idx]
        
        self.needle_shape = needle_pts

        # smoothing and interpolation
        if self.options.bspline_order > 0:
            L, _, s = stereo_needle_proc.arclength(needle_pts)
            bspline = BSplineND.fit(s, needle_pts, order=self.options.bspline_order)

            s_interp          = L - np.arange(0, L, self.options.interpolation_ds)
            self.needle_shape = bspline(s_interp, der=0)

        # if

        if debug:
            return (
                self.needle_shape, 
                self.fiducial_locations,
                self.fiducial_pose,
                debug_images,
            )
        
        # if
        
        return (
                self.needle_shape, 
                self.fiducial_locations,
                self.fiducial_pose,
        )

    # reconstruct_scene

    def save_results(self, odir: str, outfile_base: str = None) -> str:
        """ Save the results to an output directory 

            Returns:
                the output file name if the results are saved, otherwise None
        
        
        """
        saved = False

        outfile = os.path.join(
            odir,
            outfile_base if outfile_base is not None else "ct_scan_results.xlsx"
        )

        with pd.ExcelWriter(outfile, engine="xlsxwriter") as xl_writer:
            if self.needle_shape is not None:
                saved = True
                needle_shape_df = pd.DataFrame(self.needle_shape)

                needle_shape_df.to_excel(
                    xl_writer,
                    sheet_name="needle shape",
                    header=False,
                    index=False,
                )

            # if

            if self.fiducial_locations is not None:
                saved = True

                fiducial_locations_df = pd.DataFrame(self.fiducial_locations)

                fiducial_locations_df.to_excel(
                    xl_writer,
                    "fiducial locations",
                    header=False,
                    index=False,
                )

            # if

            if self.fiducial_pose is not None:
                saved = True

                fiducial_pose_df = pd.DataFrame(self.fiducial_pose)

                fiducial_pose_df.to_excel(
                    xl_writer,
                    "fiducial pose",
                    header=False,
                    index=False,
                )

            # if

        # with

        return outfile if saved else None

    # save_results

# class: CTNeedleReconstruction

def __parse_args(args=None):
    parser = ap.ArgumentParser("Reconstruct the CT needle shape of an experiment")

    # standard arguments
    parser.add_argument(
        "datafile",
        type=str,
        help="The 'ct_scan.npz' file to be processed"
    )

    parser.add_argument(
        "--odir",
        type=str,
        required=False,
        default=None,
        help="The output directory to save the results to. If not provided, results will not be saved."
    )

    parser.add_argument(
        "--debug-images",
        action="store_true",
        help="Display the debug images"
    )

    parser.add_argument(
        "--debug-image-units-in-voxels",
        action="store_true",
        help="Display the debug images with the units of the original voxel locations"
    )

    parser.add_argument(
        "--show-images",
        action="store_true"
    )

    parser.add_argument(
        '--save-images',
        action="store_true"
    )

    # hyper-parameter arguments
    hyparam_arg_grp = parser.add_argument_group("Hyperparameters")
    hyparam_arg_grp.add_argument(
        "--options-json",
        type=str,
        required=False,
        default=None,
        help="The CT reconstruction options json file to load in."
    )

    CTNeedleReconstructionOptions._add_arguments_to_parser(hyparam_arg_grp)

    ARGS = parser.parse_args(args)

    return ARGS

# __parse_args

def plot_3d_mask(
        mask: npt.NDArray[np.bool8], 
        fig: plt.Figure = None,
        ax: plt.Axes = None,
        point_scaling: npt.NDArray[np.float64] = None,
        **kwargs
    ) -> Tuple[plt.Figure, plt.Axes]:
    """ Plots the points found in a 3D mask 

        Returns:
            - The figure used for the plot
            - The axes plotted on the figure
    
    """
    if fig is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

    # if

    pts = np.argwhere(mask).astype(np.float64)
    if point_scaling is not None:
        pts *= np.reshape(point_scaling, (1, -1))

    # if

    if ax is None:
        ax = fig.add_subplot(111, projection='3d')

    # if

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        **kwargs
    )

    stereo_needle_proc.axisEqual3D(ax)

    return fig, ax

# plot_3d_mask

def main(args=None):
    ARGS = __parse_args(args)

    opts = CTNeedleReconstructionOptions.from_parsed_arguments(ARGS)
    if ARGS.options_json is not None:
        opts = CTNeedleReconstructionOptions.from_json(ARGS.options_json)

    ct_needle_reconstructor = CTNeedleReconstruction(options=opts)

    # load the data
    print("Loading the data...")
    ct_needle_reconstructor.load_ct_scan(ARGS.datafile)

    # handle the data
    print("Reconstructing the scene...")
    results = ct_needle_reconstructor.reconstruct_scene(debug=ARGS.debug_images)

    # plot the debug images
    if ARGS.debug_images:
        debug_images = results[-1]
        results      = results[:-1]

        print("Handling debug images...")
        for kw, mask_img in debug_images.items():
            fig, ax = plot_3d_mask(
                mask_img,
                fig=None,
                ax=None,
                point_scaling=(
                    ct_needle_reconstructor.dicom_image3d.image_axis_scaling
                    if not ARGS.debug_image_units_in_voxels else
                    None
                ),
            )
            ax.set_title(f"Debug Image: {kw}")

            units = "voxels" if ARGS.debug_image_units_in_voxels else "mm"
            ax.set_xlabel(f"x ({units})")
            ax.set_ylabel(f"y ({units})")
            ax.set_zlabel(f"z ({units})")
            
            if (ARGS.odir is not None) and ARGS.save_images:
                outfigfile = os.path.join(ARGS.odir, f"ct_results_debug_image_{kw}.png")
                fig.savefig(outfigfile)


                print(f"Saved figure for {kw} to: {outfigfile}")

            # if

        # for
        if ARGS.show_images:
            plt.show()

        # if
        plt.close("all")

    # if

    if ARGS.odir is not None:
        outfile = ct_needle_reconstructor.save_results(
            odir=ARGS.odir,
        )

        assert outfile is not None, "Results did not properly save! Check your data sources."

        print("Saved CT scan reconstructed results to:", outfile)

    # if


    print("Program Completed.")


# main

if __name__ == "__main__":
    main()

# if __main__
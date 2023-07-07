from dataclasses import (
    dataclass,
    field,
)
import os

from typing import (
    Any,
    Dict,
    List,
    Tuple,
)

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

import dicom

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
    bspline_order: int = -1

    # experiment options
    fiducial_locations: npt.NDArray[np.float64] = None

    @property
    def num_fiducials(self):
        return self.fiducial_locations.shape[0]
    
    # num_fiducials

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
            "fiducial_locations"        : self.fiducial_locations.tolist(),
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


    # __init__

    def load_ct_scan(self, npz_file: str) -> dicom.Image3D:
        self.dicom_image3d = dicom.Image3D.from_npz_file(npz_file)

        return self.dicom_image3d
    
    # load_ct_scan

    def reconstruct_needle(self, **kwargs) -> npt.NDArray[np.float64]:
        """ Reconstructs the needle from the needle shape
            
        
        """
        assert self.dicom_image3d is not None, "You need to load the dicom .npz CT scan files!"

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

        # TODO: simple binary cleaning up (morphological operators)
        corrected_mask = seg_mask # FIXME: change the masked used to updated one

        # TODO: connected component analysis
        fiducial_mask = corrected_mask # FIXME: change the masked used to updated one

        # TODO: get CT fiducial locations
        self.fiducial_locations = self.segment_fiducials(
            mask=fiducial_mask,
        )

        # get the needle shape points
        needle_idx_pts = np.argwhere(corrected_mask)  # (N, 3)
        needle_pts     = needle_idx_pts * np.reshape(
            [
                self.dicom_image3d.pixel_spacing[0],
                self.dicom_image3d.pixel_spacing[1],
                self.dicom_image3d.slice_thickness,
            ],
            (1, -1)
        )
        self.needle_shape = needle_pts

        # TODO: smoothing and interpolation
        if self.options.bspline_order > 0:
            pass

        return self.needle_shape

    # reconstruct_needle

    def segment_fiducials(
        self, 
        mask: npt.NDArray[np.bool8],
    ) -> npt.NDArray[np.float64]:
        fiducial_locations = np.nan * np.ones_like(self.options.fiducial_locations)
        
        # TODO

        return fiducial_locations

    # segment_fiducials

# class: CTNeedleReconstruction

def main(args=None):
    pass

# main

if __name__ == "__main__":
    main()

# if __main__
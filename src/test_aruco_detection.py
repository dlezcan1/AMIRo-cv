import os

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from stereo_needle_proc import load_stereoparams_matlab, roi

pathjoin = lambda *args: os.path.normpath( os.path.join( *args ) )


def draw_axis( img, corners, rvec, tvec, cameraMatrix, distCoeffs, scale=1 ):
    img = img.copy()
    axis = scale * np.diag( [ 1, 1, -1 ] ).astype( np.float32 )
    corner = tuple( corners[ 0 ].ravel().astype( int ) )

    colors = 255 * np.eye( 3 )[ [ 2, 1, 0 ] ]

    imgpts, _ = cv.projectPoints( axis, rvec, tvec, cameraMatrix, distCoeffs )

    cv.line( img, corner, tuple( imgpts[ 0 ].ravel().astype( int ) ), (255, 0, 0), 5 )
    cv.line( img, corner, tuple( imgpts[ 1 ].ravel().astype( int ) ), (0, 255, 0), 5 )
    cv.line( img, corner, tuple( imgpts[ 2 ].ravel().astype( int ) ), (0, 0, 255), 5 )

    return img


# draw_axis

def main():
    # load the stereo parameters
    stereo_param_dir = "calibration/Stereo_Camera_Calibration_02-08-2021/6x7_5mm"
    stereo_param_file = pathjoin( stereo_param_dir, 'calibrationSession_params-error_opencv-struct.mat' )
    stereo_params = load_stereoparams_matlab( stereo_param_file )

    # load ARUCO dictionary and parameters
    aruco_dir = "aruco/aruco_paper_install"
    aruco_dict = cv.aruco.getPredefinedDictionary( cv.aruco.DICT_4X4_50 )
    aruco_params = cv.aruco.DetectorParameters_create()
    # aruco_params.adaptiveThreshConstant = 3
    show_rejected = False

    clahe = cv.createCLAHE( tileGridSize=(50, 50), clipLimit=8 )
    aruco_length = 20  # mm

    for i in range( 9 ):
        img_l = cv.imread( pathjoin( aruco_dir, f'left-{i:04d}.png' ), cv.IMREAD_COLOR )
        img_r = cv.imread( pathjoin( aruco_dir, f'right-{i:04d}.png' ), cv.IMREAD_COLOR )

        roi_l = [ [ 0, 0 ], [ 170, -1 ] ]
        roi_r = roi_l

        # img_l = roi( img_l, roi_l, full=True )
        # img_r = roi( img_r, roi_r, full=True )

        gray_l = cv.cvtColor( img_l, cv.COLOR_BGR2GRAY ).astype( float )
        gray_r = cv.cvtColor( img_r, cv.COLOR_BGR2GRAY ).astype( float )

        # image processing
        scale = 5  # paper - 7 | transparent - 1
        baseline = 0
        gray_enh_l = np.maximum( np.minimum( scale * gray_l + baseline, 255 ), 0 ).astype( np.uint8 )
        gray_enh_r = np.maximum( np.minimum( scale * gray_r + baseline, 255 ), 0 ).astype( np.uint8 )

        # # CLAHE
        # gray_enh_l = clahe.apply(gray_l)
        # gray_enh_r = clahe.apply(gray_r)

        # # auto Correction
        # cv.intensity_transform.autoscaling( gray_l, gray_enh_l )
        # cv.intensity_transform.autoscaling( gray_r, gray_enh_r )

        # # log correction - good for transparent, but not perfect
        # cv.intensity_transform.logTransform( gray_l, gray_enh_l )
        # cv.intensity_transform.logTransform( gray_r, gray_enh_r )

        # linear transform - great for paper
        cv.intensity_transform.contrastStretching(gray_l.astype(np.uint8), gray_enh_l, 0, baseline, int(255/scale), 255)
        cv.intensity_transform.contrastStretching(gray_r.astype(np.uint8), gray_enh_r, 0, baseline, int(255/scale), 255)


        # gamma correction - good for transparent, but not perfect
        # gamma = 1/2
        # cv.intensity_transform.gammaCorrection( gray_l.astype(np.uint8), gray_enh_l, gamma=gamma )
        # cv.intensity_transform.gammaCorrection( gray_r.astype(np.uint8), gray_enh_r, gamma=gamma )

        thresh = 50
        _, thresh_l = cv.threshold( gray_enh_l, thresh, 255, cv.THRESH_BINARY )
        _, thresh_r = cv.threshold( gray_enh_r, thresh, 255, cv.THRESH_BINARY )

        # filter out black
        gray_enh_l *= (thresh_l > 0)
        gray_enh_r *= (thresh_r > 0)

        # plt.figure( figsize=(12, 8) )
        # plt.imshow( np.concatenate( (thresh_l, thresh_r), axis=1 ), cmap='gray' )

        plt.figure( figsize=(12, 8) )
        plt.imshow( np.concatenate( (gray_enh_l, gray_enh_r), axis=1 ), cmap='gray' )

        # detect aruco
        corners_l, ids_l, rejected_l = cv.aruco.detectMarkers( gray_enh_l, aruco_dict, parameters=aruco_params )
        corners_r, ids_r, rejected_r = cv.aruco.detectMarkers( gray_enh_r, aruco_dict, parameters=aruco_params )

        # id_keep = 43
        # idx_keep_l = np.argwhere(ids_l == id_keep)
        # idx_keep_r = np.argwhere(ids_r == id_keep)
        #
        # ids_l = ids_l[idx_keep_l[0][0]] if len(idx_keep_l) > 0 else None
        # ids_r = ids_r[idx_keep_r[0][0]] if len(idx_keep_r) > 0 else None
        #
        # corners_l = [corners_l[idx_keep_l[0][0]]] if len(idx_keep_l) > 0 else []
        # corners_r = [corners_r[idx_keep_r[0][0]]] if len(idx_keep_r) > 0 else []

        print( f'{i:04d} IDs | left:', ids_l, 'right:', ids_r )

        if ids_l is not None:
            cv.aruco.drawDetectedMarkers( img_l, corners_l, ids=ids_l, borderColor=(0, 255, 0) )
            for corners in corners_l:
                rvecs, tvecs, objpoints = cv.aruco.estimatePoseSingleMarkers( corners, aruco_length,
                                                                              stereo_params[ 'cameraMatrix1' ],
                                                                              stereo_params[ 'distCoeffs1' ] )

                img_l = cv.aruco.drawAxis( img_l, stereo_params[ 'cameraMatrix1' ],
                                           stereo_params[ 'distCoeffs1' ],
                                           rvecs, tvecs, aruco_length )

            # for

        # if
        if len( rejected_l ) > 0 and show_rejected:
            for cs in rejected_l:
                cs = np.expand_dims( np.vstack( (cs.squeeze(), cs[ 0, 0 ]) ), axis=0 )
                cv.polylines( img_l, cs.astype( int ), False, (0, 0, 255), 4 )

            # for
        # elif

        if ids_r is not None:
            cv.aruco.drawDetectedMarkers( img_r, corners_r, ids=ids_r, borderColor=(0, 255, 0) )

            for corners in corners_r:
                rvecs, tvecs, objpoints = cv.aruco.estimatePoseSingleMarkers( corners, aruco_length,
                                                                              stereo_params[ 'cameraMatrix2' ],
                                                                              stereo_params[ 'distCoeffs2' ] )

                img_r = cv.aruco.drawAxis( img_r, stereo_params[ 'cameraMatrix2' ], stereo_params[ 'distCoeffs2' ],
                                           rvecs, tvecs, aruco_length )

            # for
        # if
        if len( rejected_r ) > 0 and show_rejected:
            for cs in rejected_r:
                cs = np.expand_dims( np.vstack( (cs.squeeze(), cs[ 0, 0 ]) ), axis=0 )
                cv.polylines( img_r, cs.astype( int ), False, (0, 0, 255), 4 )

            # for
        # elif

        img_lr = np.concatenate( (img_l, img_r), axis=1 )
        img_lr = cv.cvtColor( img_lr, cv.COLOR_BGR2RGB )

        plt.figure( figsize=(15, 8) )
        plt.imshow( img_lr, cmap='gray' )
        plt.title( f"Image {i:04d}" )
        plt.show()

    # for


# main


if __name__ == "__main__":
    main()

# if __main__

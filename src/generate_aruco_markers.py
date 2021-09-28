import argparse

import cv2 as cv


def __get_parser():
    parser = argparse.ArgumentParser( description='Script to generate ARUCO markers.' )
    parser.add_argument( 'N', type=int, help='N x N size of the marker' )
    parser.add_argument( 'ID', type=int, help='The ID of the aruco marker' )
    parser.add_argument( '-s', '--savefile', default=None, type=str, help='The file to save the marker to' )
    parser.add_argument('-V', '--view', action='store_true', help="View the ARUCO marker.")
    parser.add_argument( '-M', default=50, type=int, help='Number of unique aruco markers' )
    parser.add_argument( '--pixel-size', default=100, type=int, help="The side size of the aruco marker" )

    return parser


# __get_parser

def main( args=None ):
    parser = __get_parser()
    args = parser.parse_args( args )

    # argument checking
    if args.N not in range( 4, 8 ):  # valid arguments are 4-7
        raise ValueError( "'N' must be between [4,7]" )

    # if

    if args.M not in [ 50, 100, 250, 1000 ]:
        raise ValueError( "'M' must be either 50, 100, 250, or 1000." )

    # if

    # get the aruco board
    aruco_dicts = { k: vars( cv.aruco )[ k ] for k in
                    list( filter( lambda x: x.startswith( 'DICT' ), vars( cv.aruco ) ) ) }
    aruco_selection_key = "DICT_{0:d}X{0:d}_{1:d}".format( args.N, args.M )
    aruco_selection = aruco_dicts[ aruco_selection_key ]  # the specific aruco pattern we would like to use

    aruco_dict = cv.aruco.getPredefinedDictionary( aruco_selection )

    aruco_img = aruco_dict.drawMarker( args.ID, args.pixel_size )

    # view the marker
    if args.view:
        while True:
            aruco_img = aruco_dict.drawMarker( args.ID, args.pixel_size )
            cv.imshow(f'{aruco_selection_key} | ID: {args.ID}', aruco_img)

            print("Press 'n' for the next marker and 'q' to stop...", end=' ')

            key = cv.waitKey(0)
            if key & 0xFF == ord('q'):
                print('Closing.')
                break

            elif key & 0xFF == ord('n'):
                print('Next marker.')
                args.ID = (args.ID + 1) % args.M
                cv.destroyAllWindows()

            else:
                print('Invalid Key.')
                continue

        # while
        cv.destroyAllWindows()
    # if

    # output the aruco image
    if args.savefile is not None:
        args.savefile = args.savefile.format(args.N, args.M, args.ID)
        cv.imwrite(args.savefile, aruco_img)
        print(f"Saved {aruco_selection_key} ARUCO marker ID {args.ID} to: '{args.savefile}'")

    # if


# main

if __name__ == "__main__":
    main()

# if __main__

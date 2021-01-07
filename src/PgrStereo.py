# Python Class for Camera integration
#
# author: Dimitri Lezcano
# date: 10-23-2020


import PyCapture2 as pycap
import numpy as np
import matplotlib.pyplot as plt
import cv2

class PgrStereo():

    def __init__(self, cam_left_idx: int = -1, cam_right_idx: int = -1):
        ''' Constructor '''
        # bus manager
        self.bus = pycap.BusManager()

        # left and right cameras
        self.cam_left = pycap.Camera()
        self.cam_right = pycap.Camera()

        # connect the cameras if applicable
        if (cam_left_idx >= 0) and (cam_right_idx >= 0):
            self.connect(cam_left_idx, cam_right_idx)

        # if
        
    # __init__

    def __del__(self):
        ''' Destructor '''
        # disconnect the cameras
        try:
            self.stopCapture()
            
        except Exception as e:
            pass
            
        self.disconnect()

    # __del__

    def connect(self, cam_left_idx: int = 0, cam_right_idx: int = 1):
        if cam_left_idx == cam_right_idx:
            raise IndexError("Need 2 different camera indices for stereo")

        if (self.bus.getNumOfCameras() < cam_left_idx + 1) or (0 > cam_left_idx):
            raise IndexError("Left camera index is out of range")

        if (self.bus.getNumOfCameras() < cam_right_idx + 1) or (0 > cam_right_idx):
            raise IndexError("Right camera index is out of range")

        self.cam_left.connect(self.bus.getCameraFromIndex(cam_left_idx))
        self.cam_right.connect(self.bus.getCameraFromIndex(cam_right_idx))

        if self.cam_left.isConnected:
            print('[LEFT]: Camera connected')

        if self.cam_right.isConnected:
            print('[RIGHT]: Camera connected')

        return (self.cam_left.isConnected, self.cam_right.isConnected)

    # connect

    def disconnect(self):
        self.cam_left.disconnect()
        print('[LEFT]: Capture disconnected')
        self.cam_right.disconnect()
        print('[RIGHT]: Capture disconnected')
                               
    # disconnect

    def enable_embedded_timestamp(self, enable_timestamp):
        for k, cam in zip(['Left', 'Right'], [self.cam_left, self.cam_right]):
            embedded_info = cam.getEmbeddedImageInfo()
            if embedded_info.available.timestamp:
                cam.setEmbeddedImageInfo(timestamp = enable_timestamp)
                if enable_timestamp :
                    print('[{:s}]: TimeStamp is enabled.'.format(k.upper()))
                else:
                    print('[{:s}]: TimeStamp is disabled.'.format(k.upper()))

            # if
        # for
    # enable_embedded_timestamp

    def isConnected(self):
        return self.cam_left.isConnected and self.cam_right.isConnected

    # isConnected    

    def grab_image_pair(self):
        ''' Grabs an image pair from each camera. Returns None if there is an error '''
        # capture left image
        try:
            img_left = self.cam_left.retrieveBuffer()
            img_left_arr = img_left.getData().reshape(img_left.getRows(), img_left.getCols(), -1)

        except pycap.Fc2error as fc2Err:
            print('Left Camera: Error Retrieving buffer: %s' % fc2Err)
            img_left_arr = None

        # capture right image
        try:
            img_right = self.cam_right.retrieveBuffer()
            img_right_arr = img_right.getData().reshape(img_right.getRows(), img_right.getCols(), -1)

        except pycap.Fc2error as fc2Err:
            print('Right Camera: Error Retrieving buffer: %s' % fc2Err)
            img_right_arr = None

        return (img_left_arr, img_right_arr)

    # grab_image_pair

    
    def print_camera_info(self):
        print("PgrStereo Camera:")
        for k, cam in zip(['Left', 'Right'], [self.cam_left, self.cam_right]):
            cam_info = cam.getCameraInfo()
            print('\n*** [{:s}] CAMERA INFORMATION ***\n'.format(k.upper()))
            print('Serial number - %d' % cam_info.serialNumber)
            print('Camera model - %s' % cam_info.modelName)
            print('Camera vendor - %s' % cam_info.vendorName)
            print('Sensor - %s' % cam_info.sensorInfo)
            print('Resolution - %s' % cam_info.sensorResolution)
            print('Firmware version - %s' % cam_info.firmwareVersion)
            print('Firmware build time - %s' % cam_info.firmwareBuildTime)
            print()
            
        # for
    # print_camera_info

    def startCapture(self):
        self.cam_left.startCapture()
        print('[LEFT]: Capture started')
                              
        self.cam_right.startCapture()
        print('[RIGHT]: Capture started')
                              
    # startCapture

    def stopCapture(self):
        self.cam_left.stopCapture()
        print('[LEFT]: Capture stopped')
                              
        self.cam_right.stopCapture()
        print('[RIGHT]: Capture stopped')

    # stopCapture


# class: PgrStereo


def prompted_capture():
    pgr_stereo = PgrStereo()
    pgr_stereo.connect()

    pgr_stereo.startCapture()

    # iterate through prompts
    counter = 0
    file_save = "{:s}-{:04d}.png"
    f1 = plt.figure(1)
    f2 = plt.figure(2)
    while True:
        try:
            input('Press [ENTER] to take an image. Press [CTRL-C] to end program.')
            img_left, img_right = pgr_stereo.grab_image_pair()

            plt.imsave(file_save.format('left', counter), img_left)
            print("Saved image:", file_save.format('left', counter))
                  
            plt.imsave(file_save.format('right', counter), img_right)
            print("Saved Image:", file_save.format('right', counter))

            plt.figure(1)
            plt.imshow(img_left)
            plt.title('Left')

            plt.figure(2)
            plt.imshow(img_right)
            plt.title('Right')

            plt.pause(1)
            # input('Press [ENTER] to continue.')
            # plt.close(f1)
            # plt.close(f2)

            counter += 1
            print()
            
        except KeyboardInterrupt:
            print("[INTERRUPT] Requested termination...")
            break

        except Exception as e:
            print("[ERROR]:", e)
            break

    # while

    plt.close('all')
    pgr_stereo.stopCapture()
    pgr_stereo.disconnect()
    
# prompted_capture

def live_capture():
    pgr_stereo = PgrStereo()
    
    pgr_stereo.connect(0,1) # where to change the stereo camera order

    pgr_stereo.startCapture()

    counter = 0
    file_base = "{}-{:04d}.png"    
    while True:
        try:
            img_left, img_right = pgr_stereo.grab_image_pair()

            img_cat = np.concatenate((img_left, img_right), axis=1)
            cv2.putText(img_cat, 'LEFT', (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,0], 4)
            cv2.putText(img_cat, 'RIGHT', (img_left.shape[1]+20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, [0,0,0], 4)

            # cv2.imshow('left', img_left[:,:,::-1])
            # cv2.imshow('right', img_right[:,:,::-1])
            cv2.imshow('left-right', img_cat[:,:,::-1])
          

            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

            # if
            
            elif key & 0xFF == ord('c'):
                cv2.imwrite(file_base.format("left", counter), cv2.cvtColor(img_left, cv2.COLOR_RGB2BGR))
                print("Captured image: ", file_base.format('left', counter))
                
                cv2.imwrite(file_base.format("right", counter), cv2.cvtColor(img_right, cv2.COLOR_RGB2BGR))
                print("Captured image: ", file_base.format('right', counter))

                counter += 1
            # elif
            

        except:
            break

    # while
    
    cv2.destroyAllWindows()
    pgr_stereo.stopCapture()
    pgr_stereo.disconnect()


# live_capture

    

def test_PgrStereo():
    pgr_stereo = PgrStereo()

    stereo_connected = pgr_stereo.connect()
    if all(stereo_connected):
        print("Stereo cameras are connected")

    # if
    pgr_stereo.enable_embedded_timestamp(True)

    pgr_stereo.print_camera_info()
    
    pgr_stereo.startCapture()

    img_left, img_right = pgr_stereo.grab_image_pair()
    
    if not isinstance(img_left, type(None)):
        plt.figure()
        plt.imshow(img_left)
        plt.title('Left Camera')

    # if
                              
    if not isinstance(img_right, type(None)):
        plt.figure()
        plt.imshow(img_right)
        plt.title('Right Camera')

    # if

    plt.show()
    
    pgr_stereo.stopCapture()
    print('Capture stopped')

    del(pgr_stereo)

# test_PgrStereo

def stereo_viewer():
    pgr_stereo = PgrStereo()

    pgr_stereo.connect()

    pgr_stereo.startCapture()
    
    while True:
        try:
            img_left, img_right = pgr_stereo.grab_image_pair()

            cv2.imshow('left', img_left[:,:,::-1])
            cv2.imshow('right', img_right[:,:,::-1])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            

        except:
            break

    # while
    
    cv2.destroyAllWindows()
    pgr_stereo.stopCapture()
    pgr_stereo.disconnect()

# stereo_viewer
            
        
#========================== MAIN =================================

if __name__ == "__main__":
    # test_PgrStereo()
    # prompted_capture()
    # stereo_viewer()
    live_capture()
# if: main

    

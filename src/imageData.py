import cv2
import numpy as np
import threading
import re
import matplotlib.pyplot as plt

class ImageData(object):
    def __init__(self,im_path,int_path,dpth_path,name):
        '''
        Image data class holds important info for each RGB and D image pair
        implments compute methods as threads to speed up computation
        each class instance is hashed in the IndoorMapping class for easy lookup
        '''
        #bookkeeping data
        self.name = name
        self.im_path = im_path
        self.int_path = int_path
        self.dpth_path = dpth_path

        #thread data
        self.feature_pipeline_thread = None
        self.feature_pipeline_success = False

        #image data
        self.distorted_rgb_image = None
        self.distorted_dpth_image = None
        self.undistorted_rgb_image = None
        self.undistorted_dpth_image = None
        self.im_width = None
        self.im_height = None
        self.intrinsic = None
        self.distortion_params = None

        #computed data
        self.kp_cv = None
        self.des_cv = None
        self.dpth_kp = None

    def open_images(self) -> bool:
        success = True
        self.distorted_rgb_image = cv2.imread(self.im_path,cv2.IMREAD_GRAYSCALE)
        if self.distorted_rgb_image is None:
            return False
        self.distorted_dpth_image = cv2.imread(self.dpth_path,cv2.IMREAD_GRAYSCALE)
        if self.distorted_dpth_image is None:
            return False
        return success

    def open_intrinsics(self) -> bool:
        success = True
        file = open(self.int_path,'r')
        line = file.readline()
        line = line.strip('\n')
        int_list = re.split(" ",line)

        if(not len(int_list) == 11):
            return False

        self.im_width = float(int_list[0])
        self.im_height = float(int_list[1])
        self.intrinsic = np.array([
            [float(int_list[2]),0,float(int_list[4])],
            [0,float(int_list[3]),float(int_list[5])],
            [0,0,1]])
        self.distortion_params = np.array([float(int_list[6]),
                                  float(int_list[7]),
                                  float(int_list[8]),
                                  float(int_list[9])])
        return success

    def undistort_images(self) -> bool:
        success = True
        self.undistorted_rgb_image = cv2.undistort(self.distorted_rgb_image,
                                               self.intrinsic,
                                               self.distortion_params,
                                               None, None)
        self.undistorted_dpth_image = cv2.undistort(self.distorted_dpth_image,
                                               self.intrinsic,
                                               self.distortion_params,
                                               None, None)
        return success

    def run_feature_extraction(self) -> bool:
        success = True
        sift = cv2.SIFT_create()
        self.kp_cv, self.des_cv = sift.detectAndCompute(self.undistorted_rgb_image,None)
        return success


    def feature_pipeline(self) -> None:
        self.feature_pipeline_success = self.open_images()
        if not self.feature_pipeline_success:
            print("error opening image {}".format(self.im_path))
            return
        self.feature_pipeline_success = self.open_intrinsics()
        if not self.feature_pipeline_success:
            print("error opening intrinsics {}".format(self.int_path))
            return
        self.feature_pipeline_success = self.undistort_images()
        if not self.feature_pipeline_success:
            print("error undistorting features {}".format(self.name))
            return
        self.feature_pipeline_success = self.run_feature_extraction()
        if not self.feature_pipeline_success:
            print("error runnning feature extraction {}".format(self.name))
            return

    def compute_features(self, threads: bool) -> bool:
        """
        the idea here is run all of our preprocessing as some pipeline that can
        be threaded if desired, threading will allow us to iterate faster on the
        large dataset
        """
        if self.feature_pipeline_thread:
            print("feature pipeline thread already exists for {}".format(self.name))
            return False

        self.feature_pipeline_thread = threading.Thread(target=self.feature_pipeline)
        self.feature_pipeline_thread.start()

        if not threads:
            self.feature_pipeline_thread.join()
            self.feature_pipeline_thread = None
            return self.feature_pipeline_success
        else:
            return True

    def join_feature_thread(self) -> bool:
        self.feature_pipeline_thread.join()
        self.feature_pipeline_thread = None
        return self.feature_pipeline_success

    def press(self,event):
        print('press', event.key)

    def renderImage(self) -> None:
        fig, ax = plt.subplots()
        fig.canvas.mpl_connect("image debugger", self.press)
        plt.show()

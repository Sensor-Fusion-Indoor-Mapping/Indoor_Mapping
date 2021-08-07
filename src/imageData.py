import cv2
import numpy as np
import threading
import re

class ImageData(object):
    def __init__(self,im_path,int_path,img_info):
        self.name = img_info[0]+img_info[1]+img_info[2]
        self.pose_name = img_info[0]
        self.im_path = im_path
        self.int_path = int_path
        self.feature_pipeline_thread = None
        self.feature_pipeline_success = False

        self.distorted_image = None
        self.undistorted_image = None
        self.im_width = None
        self.im_height = None
        self.intrinsic = None
        self.distortion_params = None

        self.kp_cv = None
        self.des_cv = None

    def open_image(self) -> bool:
        success = True
        self.distorted_image = cv2.imread(self.im_path+".jpg",cv2.IMREAD_GRAYSCALE)
        if self.distorted_image is None:
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
        self.distortion_params = [float(int_list[6]),
                                  float(int_list[7]),
                                  float(int_list[8]),
                                  float(int_list[9])]
        return success

    def undistort_image(self) -> bool:
        success = True
        self.undistorted_image = self.distorted_image
        return success

    def run_feature_extraction(self) -> bool:
        success = True
        sift = cv2.SIFT_create()
        self.kp_cv, self.des_cv = sift.detectAndCompute(self.undistorted_image,None)
        return success


    def feature_pipeline(self) -> None:
        self.feature_pipeline_success = self.open_image()
        if not self.feature_pipeline_success:
            print("error opening image {}".format(self.im_path))
            return
        self.feature_pipeline_success = self.open_intrinsics()
        if not self.feature_pipeline_success:
            print("error opening intrinsics {}".format(self.int_path))
            return
        self.feature_pipeline_success = self.undistort_image()
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
            return self.feature_pipeline_success
        else:
            return True

    def join_feature_thread(self) -> bool:
        self.feature_pipeline_thread.join()
        return self.feature_pipeline_success

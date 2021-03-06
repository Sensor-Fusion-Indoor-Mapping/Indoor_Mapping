#!/usr/bin/python3
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import cv2
import sys, getopt, os, re
import time
from imageData import ImageData


true = ["true","True","y","Y"]
false = ["false","False","n","N"]


class IndoorMapping():
    def __init__(self):
        self.image_intrinsics_directory = "../data/camera_intrinsics/"
        self.color_image_directory = "../data/color_images/"
        self.depth_image_directory = "../data/depth_images/"
        self.single_image_name = ""
        self.load_single_image = False
        self.run_with_threads = True
        self.compute_block_size = 14
        self.image_dict = {}
        self.image_list = []

    def set_arguments(self,argv):
        try:
            opts, args = getopt.getopt(argv,"hi:t:b:",["img=", "thread=","blocksize="])
        except getopt.GetoptError:
            print('main.py -i <inputName>')
            sys.exit(2)
        for opt, arg in opts:
            if opt == '-h':
                print('main.py -i <inputfile>')
                sys.exit()
            elif opt in ("-i", "--img"):
                self.single_image_name = arg
                self.load_single_image = True
                print("loading files only from camera {}".format(arg))
            elif opt in ("-t","--thread"):
                if arg in false:
                    self.run_with_threads = False
                    print("running program without threads")
            elif opt in ("-b","--blocksize"):
                self.compute_block_size = int(arg)
                print("running program with compute block size of {}".format(arg))

    def run(self):
        self.load_image_data()
        start = time.process_time()
        self.compute_features()
        end = time.process_time()
        print("computed features in {} sec (process time)".format(end-start))

        #TODO add more to compute pipelines here/ ie BA RANSAC methods...

        run = True
        while run:
            run = self.terminal_debugger()

    def load_image_data(self):
        """
        read all files from directory, paths are known
        if load single is enabled, check for name of path
        load data into an image processor class
        """
        image_dir = os.listdir(self.color_image_directory)
        for file in image_dir:
            #extract file name
            file = os.path.splitext(file)[0]
            file_info = re.split('_',file)
            color_path = self.color_image_directory + file + ".jpg"

            #if load name then cont. unless desired files
            if(self.load_single_image and not file_info[0] == self.single_image_name):
                continue

            #match to intrisic file
            intrinsic_file = file_info[0] + "_intrinsics_" + file_info[1][1] + ".txt"
            intrinsic_path = self.image_intrinsics_directory + intrinsic_file

            #match to depth file
            depth_file = file_info[0] + "_d" + file_info[1][1] + "_" + file_info[2] + ".png"
            depth_path = self.depth_image_directory + depth_file

            #generate hashable name
            #basically the same as a file name but type x instead of i or d
            name = file_info[0] + "_x" + file_info[1][1] + "_" + file_info[2]

            #load into class
            #save index of name in a dict for queries
            self.image_dict[name] = len(self.image_list)
            self.image_list.append(ImageData(color_path,intrinsic_path,
                                              depth_path,name))

    def compute_features(self):
        thread_count = 0
        thread_count_start = 0
        for imgData in self.image_list:
            success = imgData.compute_features(self.run_with_threads)
            if not success:
                print("something went wrong computing featues on image {}".format(imgData.name))
                return False
            thread_count += 1
            #run with thread count to avoid running out of memory by batching threads into compute groups
            if(self.run_with_threads and thread_count >= self.compute_block_size):
                for imgData in self.image_list[thread_count_start:(thread_count+thread_count_start)]:
                    success = imgData.join_feature_thread()
                    if not success:
                        print("something went wrong joining feature thread of image {}".format(imgData.name))
                        return False
                thread_count_start += thread_count
                thread_count = 0
                print("computed {} images".format(thread_count_start))
            elif(not self.run_with_threads):
                print("computed {} images".format(thread_count))
        return True

    def terminal_debugger(self):
        print("""
              ============================
              |     debugging options    |
              ============================
              | [i] -> render img of name|
              | [q] -> quit              |
              | [tbd]...                 |
              ============================
              """)
        value = input(">>> ")

        if value == "i":
            print("enter name of image to render")
            print("name example <hash>_x<cam index>_<yaw index>")
            name = input(">>> ")
            if not name in self.image_dict:
                print("{} is a not valid name".format(name))
                return True
            #self.image_list[self.image_dict[name]].renderImage()

        if value == "q":
            print("quitting...")
            return False
        return True

if __name__ == "__main__":
    imapp = IndoorMapping()
    imapp.set_arguments(sys.argv[1:])
    imapp.run()

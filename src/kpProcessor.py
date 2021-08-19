import numpy as np
import cv2

def match_sift_points(desA,kpA,desB,kpB):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(desA,desB,k=2)
    # Need to draw only good matches, so create a mask

    goodMatches = []
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            goodMatches.append(m)

    src_pts = np.asarray([kpA[m.queryIdx].pt for m in goodMatches])
    des_pts = np.asarray([kpB[m.trainIdx].pt for m in goodMatches])

    return src_pts.T, des_pts.T

def match_depth(kpA,depth_imageA,kpB,depth_imageB):
    get_depth = lambda pnt : depth_imageA[int(pnt[1]), int(pnt[0])]/255
    src_dpth = np.asarray([get_depth(pnt) for pnt in kpA])
    get_depth = lambda pnt : depth_imageB[int(pnt[1]), int(pnt[0])]/255
    des_dpth = np.asarray([get_depth(pnt) for pnt in kpB])
    return src_dpth, des_dpth

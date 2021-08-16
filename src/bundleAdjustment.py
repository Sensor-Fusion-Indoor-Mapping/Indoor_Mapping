import numpy as np
from scipy.optimize import least_squares

'''
Read BA_notes for more information
the goal of the following methods is to perform bundle adjustment on a given set
of keypoints and images with a custom error function that calculates the 2D proj
error and the 3D projection error of the given points
'''

def bundle_adjust(P1,P2,K1,K2,RGB_points,D_points):
    """
    entry point, perform bundle adjustment on the given parameters
    point lists are provieded in an (img,m,n) format where m is the point size and
    n is the number of points

    steps:
        format inital guess data
        calculate inital error
        least squares
        calculate final error
    """
    cam_num = 2
    #format inital guesses
    camera_poses = np.zeros((3,2,cam_num))
    pose_list = np.asarray([P1,P2])
    """
    looks like
    ------
    |rx x|
    |ry y|
    |rz z|
    ------
    """
    print(pose_list.shape)
    for i in range(cam_num):
        camera_poses[:,0,0] = rotationMatrix2angleaxis(pose_list[i,:,0:3])
        camera_poses[:,1,0] = pose_list[i,:,3]

    #point list
    """
    looks like
    RGB = (img,3,n)
    D = (img,1,n)
    """
    RGB_adjust = RGB_points
    D_adjust = D_points
    px, py, = 0,0

    #compute initial error

    #least squares

    #compute final error

def pack_data(poses,rgb,d):
    """
    packs data into a single vector for least squares operation
    """
    a = poses.flatten(order='F')
    b = rgb.flatten()
    c = d.flatten()
    return np.concatenate((a,b,c))

def unpack_data(data, pose_s, rgb_s, d_s):
    """
    unpacks data from concatenated flattened array for testing
    """
    pose_len = 1
    for x in pose_s:
        pose_len *= x
    poses = np.reshape(data[:pose_len],pose_s,order="F")
    rgb_len = 1
    for x in rgb_s:
        rgb_len *= x
    rgb = np.reshape(data[pose_len:pose_len+rgb_len],rgb_s)
    d = np.reshape(data[pose_len+rgb_len:],d_s)
    return poses,rgb,d

def rotationMatrix2angleaxis(R):
    #for very small values
    ax = [0,0,0]
    ax[0] = R[2,1] - R[1,2]
    ax[1] = R[0,2] - R[2,0]
    ax[2] = R[1,0] - R[0,1]
    ax = np.array(ax)


    costheta = max( (R[0,0] + R[1,1] + R[2,2] - 1.0) / 2.0 , -1.0)
    costheta = min(costheta, 1.0)

    sintheta = min(np.linalg.norm(ax) * 0.5 , 1.0)
    theta = np.arctan2(sintheta, costheta)

    #TODO (MatLab had pression problems, I am not sure about python)

    kthreshold = 1e-12
    if (sintheta > kthreshold) or (sintheta < -kthreshold):
        r = theta / (2.0 *sintheta)
        ax = r * ax
        return ax
    else:
        if (costheta > 0.0):
            ax = ax *0.5
            return ax
    inv_one_minus_costheta = 1.0 / (1.0 - costheta)

    for i in range(3):
        ax[i] = theta * np.sqrt((R(i, i) - costheta) * inv_one_minus_costheta)
        cond1 = ((sintheta < 0.0) and (ax[i] > 0.0))
        cond2 = ((sintheta > 0.0) and (ax[i] < 0.0))
        if cond1 or cond2:
            ax[i] = -ax[i]
            return ax
    pass

def xprodmat(a):
    assert(a.shape[0] == 1  and a.shape[1] ==3)
    ax=a[0,0]
    ay=a[0,1]
    az=a[0,2]
    A=np.array([[0, -az,ay],[az,0,-ax],[-ay,ax,0]])
    return A


def AngleAxisRotatePts(validMot, validStr):
    validStr=np.transpose(validStr)
    angle_axis = np.reshape(validMot[0:3], (1,3))
    theta2 = np.inner(angle_axis, angle_axis)

    if (theta2 > 0.0):
        theta = np.sqrt(theta2)
        w = (1.0/theta) * angle_axis

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        w_cross_pt = np.dot(xprodmat(w),validStr)

        w_dot_pt = np.dot(w,validStr)
        t1= (validStr * costheta)
        t2 = (w_cross_pt * sintheta)
        t3 = np.dot( (1 - costheta) * np.transpose(w),w_dot_pt)
        result = t1 + t2 + t3

    else:
        w_cross_pt = np.dot(xprodmat(angle_axis),validStr)
        result = validStr + w_cross_pt
    return np.transpose(result)

def AngleAxis2RotationMatrix(angle_axis):
    R= np.zeros((3,3))
    theta2 = np.inner(angle_axis, angle_axis)
    if (theta2 > 0.0):
        theta = np.sqrt(theta2)
        wx = angle_axis[0] / theta
        wy = angle_axis[1] / theta
        wz = angle_axis[2] / theta
        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        R[0,0] = costheta + wx * wx * (1 - costheta)
        R[1,0] = wz * sintheta + wx * wy * (1 - costheta)
        R[2,0] = -wy * sintheta + wx * wz * (1 - costheta)
        R[0,1] = wx * wy * (1 - costheta) - wz * sintheta
        R[1,1] = costheta + wy * wy * (1 - costheta)
        R[ 2, 1] = wx * sintheta + wy * wz * (1 - costheta)
        R[0,2] = wy * sintheta + wx * wz * (1 - costheta)
        R[1,2] = -wx * sintheta + wy * wz * (1 - costheta)
        R[2,2] = costheta + wz * wz * (1 - costheta)
    else:
        R[0, 0] = 1
        R[1, 0] = -angle_axis[2]
        R[2, 0] = angle_axis[1]
        R[0, 1] = angle_axis[2]
        R[1,1] = 1
        R[2,1] = -angle_axis[0]
        R[0,2] = -angle_axis[1]
        R[1,2] = angle_axis[0]
        R[2,2] = 1
    return R

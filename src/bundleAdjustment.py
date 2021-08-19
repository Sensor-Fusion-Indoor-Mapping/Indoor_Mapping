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
    camera_poses = np.zeros((cam_num,3,2))
    pose_list = np.asarray([P1,P2])
    """
    looks like
    ------
    |rx x|
    |ry y|
    |rz z|
    ------
    """
    for i in range(cam_num):
        camera_poses[i,:,0] = rotationMatrix2angleaxis(pose_list[i,:,0:3])
        camera_poses[i,:,1] = pose_list[i,:,3]

    #point list
    """
    looks like
    RGB = (img,3,n)
    D = (img,1,n)
    """
    Intrinsics = np.asarray([K1,K2])
    #RGB_adjust = RGB_points
    #D_adjust = D_points
    print(D_points.shape)
    px, py = 0,0

    x = pack_data(camera_poses,RGB_points,D_points)
    unpack_data(x,camera_poses.shape,RGB_points.shape,D_points.shape)


    #compute initial error
    error = compute_err_slam(pose_list,RGB_points,D_points,Intrinsics, RGB_points)
    error_func = lambda x : 2*np.sqrt(np.sum(np.power(x,2)) / x.shape[0])
    print("initial error {}".format(error_func(error)))
    #least squares
    func = lambda x : wrapper_func(x,Intrinsics,RGB_points,camera_poses.shape,RGB_points.shape,D_points.shape)
    sol = least_squares(func,pack_data(camera_poses,RGB_points,D_points), method="lm",max_nfev=1000)
    #compute final error
    print("final data points {}".format(sol.x))
    print("final error {}".format(error_func(sol.fun)))

def compute_error(pose_list,RGB_points,D_points,px,py,obs_indx,obs_val):
    '''
    least squares attempts to minimize the vector output by this function
    our error function is two fold, one part computes the usual 2D reprojection error of each point
    the second error function computes the 2D reprojection error of each 3D poin in the camera frame
    '''
    err = compute_err_slam(pose_list,RGB_points,D_points,px,py,obs_indx,obs_val)
    err += compute_err_depth(err)
    return err

def compute_err_slam(camera_poses, RGB_points, D_points, Intrinsics, observed_RGB):
    # points from camera A are 'world coordinates'
    # for each camera frame, project points into frame of other camera
    observed_error = np.zeros((0,3))
    for i in range(camera_poses.shape[0]):
        # get points and project onto opposite cam frame
        # each point is then compared to position of measured points in the
        # original system
        #for i onto !i points
        print(RGB_points[0])
        for x in range(camera_poses.shape[0]):
            if x!=i:
                pose = np.zeros((3,4))
                pose[:,0:3] = AngleAxis2RotationMatrix(camera_poses[i,:,0])
                pose[:,3] = camera_poses[i,:,1]
                for n in range(RGB_points.shape[2]):
                    Q_i = np.hstack([np.dot(D_points[x,n],np.dot(np.linalg.inv(Intrinsics[x]),RGB_points[x,:,n])),1])
                    projected_2D_pnt = np.dot(Intrinsics[i],np.dot(pose,Q_i))
                    pi_pnt = projected_2D_pnt/projected_2D_pnt[2]
                    if not np.isnan(np.sum(pi_pnt)):
                        observed_error = np.vstack([observed_error,observed_RGB[i,:,n] - pi_pnt])

        return observed_error.flatten()

def wrapper_func(x,Intrinsics,obs,pose_s,rgb_s,d_s):
    pose, rgb, d = unpack_data(x,pose_s,rgb_s,d_s)
    return compute_err_slam(pose,rgb,d,Intrinsics,obs)

def compute_err_depth(err):
    return np.array([0])

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

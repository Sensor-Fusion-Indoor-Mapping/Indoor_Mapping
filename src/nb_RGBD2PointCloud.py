
#################################################
### THIS FILE WAS AUTOGENERATED! DO NOT EDIT! ###
#################################################
## file to edit: dev_nb/RGBD2PointCloud.ipynb
import numpy as np
#assumes image is loaded in rgb format not opencv default of bgr example: cv2.imread(path)[:,:,::-1]
#assumes depth image is loaded via cv2.imread(fl_pth_d,cv2.IMREAD_ANYDEPTH)

class PointCloud():

    def __init__(self,img,z):
        self.z = z
        self.img = img
        self.pntCld()
        self.clr()

    def get_pntCld(self): return self.pntCld
    def get_pntCld_clr(self): return self.clr

    def clr(self):
        clr=self.img.reshape(self.img.shape[0] * self.img.shape[1], 3)
        clr=np.divide(clr, 255)
        self.clr=clr

    def pntCld(self):
        ''' Args:
        -    Depth map image in numpy array shape (N,M)
        Returns:
            x,y,z position in numpy array of shape (N*M,3)
        '''
        row, col = np.indices(self.z.shape)
        self.pntCld= np.column_stack((row.ravel(), col.ravel(), self.z.ravel()))

    def TransformPnt(self,T):
        '''Args:
        -   pntCld:  x,y,z, positions in a numpy array of shape (N, 3) where N is the number of 3D points
        -   T:  Homogeneous Transform in numpy array of shape (4, 4)
        Returns:
            Transformed x,y,z points in (N,3) numpy array
        '''
        # flattens T from (4,4) to (16,) then vectorizes it to length N
        T_shp=np.array([T.reshape(16)] * self.pntCld.shape[0])
        arr=np.c_[self.pntCld,T_shp,np.zeros(self.pntCld.shape)] # make array of shape (N,22) x,y,z (3,) +T (16,) + new x,y,z (3,) set to zero
        arr[:,19]=arr[:,3] *arr[:,0] + arr[:,4] *arr[:,1] + arr[:,5] *arr[:,2]+arr[:,6]  # new x
        arr[:,20]=arr[:,7] *arr[:,0] + arr[:,8] *arr[:,1] + arr[:,9] *arr[:,2]+arr[:,10] # new y
        arr[:,21]=arr[:,11]*arr[:,0] + arr[:,12]*arr[:,1] + arr[:,13]*arr[:,2]+arr[:,14] # new z
        self.pntCld=arr[:,19:]

    def plot3D(self):
        '''x,y,z 3D plot of image where points are colored with from the 2D image color'''
        from matplotlib import pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(self.pntCld[:,0], self.pntCld[:,1], self.pntCld[:,2], c=self.clr, marker='o', alpha=0.1, s=0.1)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')

        plt.show()
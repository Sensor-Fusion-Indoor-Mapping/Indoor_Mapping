import numpy as np
import kpProcessor
import structure
import processor
import bundleAdjustment

class ImageGroup(object):
    def __init__(self, imageList):
        '''
        read in a list of image names and begin constructing point
        correspondances and image orientation estimates. Extend this class to
        take in N images with the goal of implimenting bundle ajustment as the
        last step.

        general data format:
            any array of points is (m,n)
            example a 2D homogenous point list would be (3,n)

        '''
        #list of imageDatas or imageGroups, both contain nessary info to build
        self.imageList = imageList

        #class params
        self.ransac_T = 0.02

    def compute_essential_and_inlyer(self,imA,imB):
        '''
        given the indexes of imA and imB compute the essental matrix and number
        of inlyers for a given solution
        algo:
            get KP
            match KP
            compute F & inlyers
            E = K'T * F * K
            recompute inlyers?
        '''
        #extract data from imgs
        descA, kpA = self.imageList[imA].des_cv, self.imageList[imA].kp_cv
        descB, kpB = self.imageList[imB].des_cv, self.imageList[imB].kp_cv

        #compute match list
        pointsA, pointsB = kpProcessor.match_sift_points(descA,kpA,descB,kpB)

        #convert to homogenous coords
        pointsA = processor.cart2hom(pointsA)
        pointsB = processor.cart2hom(pointsB)

        #compute F and inlyers from homography
        if len(pointsA.shape) != 2 and len(pointsB.shape) != 2:
            return np.eye(3), np.array([])

        F, inlyersA, inlyersB = structure.ransac_fundamental_matrix(pointsA, pointsB,
                                                         800,self.ransac_T)
        if F.shape != (3,3):
            return np.eye(3), np.array([])

        inlyers = np.asarray((inlyersA,inlyersB))


        #get intrinsic data from imgs
        K_a = self.imageList[imA].intrinsic
        K_b = self.imageList[imB].intrinsic

        #solve for E
        E = np.dot(K_b.T, np.dot(F,K_a))
        U, S, V = np.linalg.svd(E)
        S[-1] = 0
        S = [1, 1, 0] # Force rank 2 and equal eigenvalues
        E = np.dot(U, np.dot(np.diag(S), V))

        #recompute inlyers?
        print(inlyers.shape, inlyersA.shape, inlyersB.shape)
        print(E)
        return E, inlyers

    def compute_pose(self, E, inlyers):
        """
        takes previously calculated essential matrix and attempts to reconstruct
        the pose of the camera.
        P = [R|t] => 4X4
            [0|1]

        x = P * x'
        """
        #compute normalized inlyer points
        points1n, T1 = structure.scale_and_translate_points(inlyers[0])
        points2n, T2 = structure.scale_and_translate_points(inlyers[1])

        #compute P2 from cands
        P2_cand = structure.compute_P_from_essential(E)
        P1 = np.hstack((np.identity(3),np.zeros((3,1))))
        indx = -1
        point_indx = 0
        for i, P2 in enumerate(P2_cand):
            #find the correct camera parameters
            d1 = structure.reconstruct_one_point(
                points1n[:,0], points2n[:,0], P1, P2)
            if(d1.shape != (4,)):
                continue
            # Convert P2 from camera to world view
            P2_homogenous = np.linalg.inv(np.vstack([P2, [0,0,0,1]]))
            d2 = np.dot(P2_homogenous[:3, :4], d1)

            if d1[2] > 0 and d2[2] > 0:
                indx = i

        if indx == -1:
            print("there is an issue with the p calculation, indx -1")
            return np.array([])

        P2 = P2_cand[indx]
        return P2

    def compute(self):
        for i in range(len(self.imageList)-2):
            E, inlyers = self.compute_essential_and_inlyer(i,i+1)
            if(len(inlyers.shape) == 1):
                print("error in compute essential, NEXT IMAGES +++++++++")
                continue
            P2 = self.compute_pose(E, inlyers)
            print(P2)
            if(len(P2.shape) != 2):
                print("ERROR generating p2")
                continue

            # bundle adjustment
            D_points = np.array([])
            P1 = np.hstack((np.identity(3),np.zeros((3,1))))
            K1 = self.imageList[i].intrinsic
            K2 = self.imageList[i+1].intrinsic

            bundleAdjustment.bundle_adjust(P1,P2,K1,K2,inlyers,D_points)


            return
            print("NEXT IMAGES +++++++++++++++")
    def debug(self):
        pass

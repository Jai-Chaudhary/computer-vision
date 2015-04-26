import numpy as np
import scipy as sp
import cv2
from skimage.measure import  ransac
from skimage.transform import warp, AffineTransform
from skimage import img_as_ubyte

class SIFT:
    MIN_MATCH_COUNT = 10
    extractor = cv2.SIFT()

    
    """
    returns list of DMatch Objects (matches) sorted by distance 
    that contains keypoints matched in images.

    :param keypoints1: list of keypoint objects for image 1
    :param descriptorSet1:  list of descriptor objects for image 1
    :param keypoints2:  list of keypoint objects for image 2
    :param descriptorSet2:  list of descriptor objects for image 2
    :returns: list of DMatch Objects
    """
    def findMatches(self,keypoints1, descriptorSet1, keypoints2, descriptorSet2):        
        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

        # Match descriptors.
        matches = bf.match(descriptorSet1,descriptorSet2)

        # Sort them in the order of their distance.
        matches = sorted(matches, key = lambda x:x.distance)

        # visualize the matches
        print '#matches:', len(matches)
        dist = [m.distance for m in matches]

        print 'distance: min: %.3f' % min(dist)
        print 'distance: mean: %.3f' % (sum(dist) / len(dist))
        print 'distance: max: %.3f' % max(dist)
        
        return matches

    """
     generate an image where the matches are visualizated by 
     connecting matching pixels by highlighted lines

    :param imageFilename1: path to image1
    :param imageFilename2:  path to image2
    :returns: void
    """
    def showImages(self, imageFilename1, imageFilename2):


        # find the keypoints and descriptors with SIFT
        img1 = cv2.cvtColor(cv2.imread(imageFilename1), cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(cv2.imread(imageFilename2), cv2.COLOR_BGR2GRAY)

        kp1, des1 = self.extractor.detectAndCompute(img1,None)
        kp2, des2 = self.extractor.detectAndCompute(img2,None)
        matches = self.findMatches(kp1, des1, kp2, des2)
        self.drawMatches(img1, kp1, img2, kp2, matches)
                

        
        
    """
     Visualization
     My own implementation of cv2.drawMatches as brew binary for MacOSX
     OpenCV 2.4.11 does not suppourt this function  but it's supported in
     OpenCV 3.0.0

    :param img1: image1 Object
    :param kp1: list of keypoint objects for image 1
    :param img2: image2 Object
    :param kp2: list of keypoint objects for image 2
    :param matches: list of DMatch Objects that contains keypoints matched in images.
    :returns: void
    """    
    def drawMatches(self, img1, kp1, img2, kp2, matches, name = 'DrawMatches'):

        
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        view = sp.zeros((max(h1, h2), w1 + w2, 3), sp.uint8)
        view[:h1, :w1, 0] = img1
        view[:h2, w1:, 0] = img2
        view[:, :, 1] = view[:, :, 0]
        view[:, :, 2] = view[:, :, 0]

        
        for m in matches:

            # Get the matching keypoints for each of the images
            img1_idx = m.queryIdx
            img2_idx = m.trainIdx

            # x - columns
            # y - rows
            (x1,y1) = kp1[img1_idx].pt
            (x2,y2) = kp2[img2_idx].pt
            color = tuple([sp.random.randint(0, 255) for _ in xrange(3)])
            cv2.line(view, (int(x1), int(y1)) , ((int(x2) + w1), int(y2)), color)
            cv2.circle(view, (int(x1),int(y1)), 4, (255, 0, 0), 1)   
            cv2.circle(view, (int(x2)+w1,int(y2)), 4, (255, 0, 0), 1)

        cv2.imwrite(name + ".png", view)

        
    def affineMatches(self, matches, kp1, kp2):
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ])
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ])
        model = AffineTransform()
        model.estimate(src_pts, dst_pts)
        
        model_robust, inliers = ransac((src_pts, dst_pts), AffineTransform, min_samples=3,
                               residual_threshold=30, max_trials=500)
        
        print("Least Square Model(scale, translation. rotation) ", model.scale, model.translation, model.rotation)
        print("RANSAC Least Square Model(scale, translation. rotation) ", model_robust.scale, model_robust.translation, model_robust.rotation)

        inlier_idxs = np.nonzero(inliers)[0]
        
        affineMatches = [matches[idx] for idx in inlier_idxs]
        print ("AFFINE MATCHES: ", len(affineMatches))

        return affineMatches, model_robust

    def alignAffine(self, imageFilename1, imageFilename2, affineTransform):
        img1 = cv2.imread(imageFilename1)
        img2 = cv2.imread(imageFilename2)
        img_warped = warp(img2, affineTransform, output_shape=img1.shape)

        b1,g1,r1 = cv2.split(img1)
        
        b_warped,g_warped,r_warped = cv2.split(img_warped)
        img_merged = cv2.merge((img_as_ubyte(b_warped), g1, r1 ))
        cv2.imwrite("AffineorigImage.png", img1)
        cv2.imwrite("AffinewarpedImage.png", img_as_ubyte(img_warped))
        cv2.imwrite("AffinemergedImage.png", img_merged)

        print "Affine Error: ", self.alignmentError(img1, img_warped)
        
    def alignmentError(self, img1, img2):
        hist1 = cv2.calcHist([img1.astype('float32')],[0],None,[256],[0,256])
        hist2 = cv2.calcHist([img2.astype('float32')],[0],None,[256],[0,256])
        return cv2.compareHist(hist1, hist2, cv2.cv.CV_COMP_BHATTACHARYYA)
        
    def homographyMatches(self, matches, kp1, kp2):
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1,1,2)

        transformMatrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,30.0)
        
        inliers = mask.ravel().tolist()
        
        inlier_idxs = np.nonzero(inliers)[0]
        
        homographyMatches = [matches[idx] for idx in inlier_idxs]
        
        print ("Homography Matches", len(homographyMatches))
        return homographyMatches, transformMatrix
    
    
    def alignHomography(self, imageFilename1, imageFilename2, homographyTransform):
        img1 = cv2.imread(imageFilename1)
        img2 = cv2.imread(imageFilename2)
        img_warped = cv2.warpPerspective(img2, homographyTransform, img1.shape[:2], (1000,1000), flags = cv2.WARP_INVERSE_MAP)

        b1,g1,r1 = cv2.split(img1)
        
        b_warped,g_warped,r_warped = cv2.split(img_warped)

        img_merged = cv2.merge((b_warped,g1,r1))
        cv2.imwrite("HomographyorigImage.png", img1)
        cv2.imwrite("HomographywarpedImage.png", img_warped)
        cv2.imwrite("HomographymergedImage.png", img_merged)

        print "Homography Error: ", self.alignmentError(img1, img_warped)
        

if __name__ == "__main__":
    sift = SIFT()
    imageFilename1 = 'StopSign1.jpg'
    imageFilename2 = 'StopSign2.jpg'

    sift.showImages(imageFilename1, imageFilename2)

    
    img1 = cv2.cvtColor(cv2.imread(imageFilename1), cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(cv2.imread(imageFilename2), cv2.COLOR_BGR2GRAY)

    kp1, des1 = sift.extractor.detectAndCompute(img1,None)
    kp2, des2 = sift.extractor.detectAndCompute(img2,None)
    matches = sift.findMatches(kp1, des1, kp2, des2)
    affineMatches, affineTransform =  sift.affineMatches(matches, kp1, kp2)

    sift.drawMatches(img1, kp1, img2, kp2, affineMatches, 'affineMatches')

    sift.alignAffine( imageFilename1, imageFilename2, affineTransform)

    homographyMatches,  homographyTransform = sift.homographyMatches(matches, kp1, kp2)

    sift.drawMatches(img1, kp1, img2, kp2, homographyMatches, 'homographyMatches')

    sift.alignHomography( imageFilename1, imageFilename2, homographyTransform)





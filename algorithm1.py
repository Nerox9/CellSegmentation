import os
import cv2
import numpy as np
from matplotlib import pyplot
# Todo: check slir, random walker


class Algorithm1:
    def __init__(self):
        self.parameters = {
            "MeanShift": {'max':54, 'termcrit':10},
            "Hough": {'dp':1, 'minDist':20, 'param1':50, 'param2':30, 'minRadius':0, 'maxRadius':0},
            "Gauss": {'x':5, 'y':5, 'SigmaX':0},
            "Gauss2": {'x': 7, 'y': 5, 'SigmaX': 0},

        }

        self.training_path = r".\training"
        self.testing_path = r".\test"

        self.train_set = os.listdir(self.training_path)
        self.manual_set = list()
        self.orig_set = list()

        self.show = False
        self.image_path = ""
        self.manual_path = ""
        self.win = ""
        self.img = ""


        for file in self.train_set:
            if file.startswith("manseg_"):
                self.manual_set.append(file)
            else:
                self.orig_set.append(file)

    def preprocess(self, imageid=0, show=False):
        self.show = show
        self.image_path = os.path.join(self.training_path, self.orig_set[imageid])
        self.manual_path = os.path.join(self.training_path, self.manual_set[imageid])
        self.win = ""

        image = cv2.imread(self.image_path, 0)
        mask = cv2.imread(self.manual_path, 0)

        img = np.zeros((image.shape[0], image.shape[1], 3), np.uint8)
        img[:, :, 0] = image
        img[:, :, 1] = image
        img[:, :, 2] = image

        self.img = img
        # print(img.shape, img.dtype)

    def process(self, imageid=0, show=False):
        self.preprocess(imageid, show)
        #self.Controller()
        return self.segmentation()

    def segmentation(self):
        img = self.img
        originalImg = img

        img = self.MeanShift(img)
        cv2.imwrite("meanshift.jpg", img)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        cv2.imwrite("gray.jpg", gray)
        img = self.GaussianFilter(gray, self.parameters['Gauss'])
        cv2.imwrite("gauss.jpg", img)

        hough = self.Hough(img, originalImg)
        cv2.imwrite("hough.jpg", hough)

        laplacian = self.Contour(gray)
        #laplacian = self.GaussianFilter(laplacian)
        #self.Show(self.win,laplacian)


        min = np.min(laplacian)
        laplacian = laplacian - min  # to have only positive values
        max = np.max(laplacian)
        div = max / float(255)  # calculate the normalize divisor
        laplacian_8u = np.uint8(np.round(laplacian / div))
        cv2.imwrite("laplacian.jpg", laplacian_8u)
        self.Show(self.win, laplacian_8u)
        laplacian_8u = self.GaussianFilter(laplacian_8u, self.parameters['Gauss2'])

        print("dilate")
        kernel = np.ones((2, 2), np.uint8)
        dilation = cv2.dilate(laplacian, kernel, iterations=1)
        self.Show(self.win, dilation)
        #blobs = self.BlobDetection(img)
        #cv2.imwrite("blobs.jpg", blobs)
        #img = self.BG2Blk(img)

        img = self.HistEq(laplacian_8u)

        #self.PrintHist(laplacian_8u)

        #dist = cv2.distanceTransform(gray, cv2.DIST_L2, 5)
        #if self.show:
        #    self.Show(win, dist)

        img = self.Threshold(laplacian_8u)
        self.Show(self.win, img)

        #img = self.kMeans(img)
        #cv2.imwrite("kmenas.jpg", img)
        #img = self.Canny(img)
        #cv2.imwrite("canny.jpg", img)
        finalImg = self.Morph(img)
        cv2.imwrite("morph.jpg", finalImg)

        if self.show:
            self.CatShow(self.win, originalImg[:, :, 0], finalImg)
        return finalImg
        # """

    @staticmethod
    def CatShow(win, img1, img2):
        conc = np.concatenate((img1, img2), axis=1)
        cv2.imshow(win, conc)
        cv2.waitKey()

    @staticmethod
    def Show(win, image):
        cv2.imshow(win, image)
        cv2.waitKey()

    def test(self, image):
        manual_segmented = cv2.imread(self.manual_path, 0)

        image = image.astype('bool')
        manual_segmented = manual_segmented.astype('bool')
        dice = np.sum(manual_segmented[image == 1])*2.0 / (np.sum(manual_segmented) + np.sum(image))
        print('Dice similarity score is %.3f'% dice)
        return dice

    def termcrit(self,x):
        self.parameters["MeanShift"]["termcrit"] = x
        self.control = self.MeanShift(self.img)
        cv2.imshow(self.win, self.control)
        pass

    def max(self,x):
        self.parameters["MeanShift"]["max"] = x
        self.control = self.MeanShift(self.img)
        cv2.imshow(self.win, self.control)
        pass

    def Controller(self):
        cv2.namedWindow(self.win)
        cv2.createTrackbar("MeanShift - MaxLevel", self.win, 0, 100, self.termcrit)
        cv2.createTrackbar("MeanShift - TermCrit", self.win, 0, 100, self.max)

        max_old = 0
        termcrit_old = 0

        while (1):
            max = cv2.getTrackbarPos("MeanShift - MaxLevel", self.win)
            termcrit = cv2.getTrackbarPos("MeanShift - TermCrit", self.win)
            cv2.waitKey(1)

    def MeanShift(self, img):
        # Mean Shift
        params = self.parameters['MeanShift']
        self.shifted = cv2.pyrMeanShiftFiltering (img, params['max'], params['termcrit'])

        if self.show:
            self.CatShow(self.win, img, self.shifted)

        return self.shifted
    
    def GaussianFilter(self, img, params):
        # Gaussian Filter
        gauss = cv2.GaussianBlur(img, (params['x'], params['y']), params['SigmaX'])
        if self.show:
            self.Show(self.win, gauss)
        return gauss

    def Hough(self, img, originalImg):
        houghImg = originalImg
        params = self.parameters['Hough']
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, params['dp'], params['minDist'],
                                   param1=params['param1'], param2=params['param2'], minRadius=params['minRadius'],
                                   maxRadius=params['maxRadius'])

        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            # draw the outer circle
            cv2.circle(houghImg, (i[0], i[1]), i[2], (0, 255, 0), 2)
            # draw the center of the circle
            cv2.circle(houghImg, (i[0], i[1]), 2, (0, 0, 255), 3)
        if self.show:
            self.Show(self.win, houghImg)

        return houghImg

    def BlobDetection(self, img):
        # Blob Detection
        params = cv2.SimpleBlobDetector_Params()
        params.minThreshold = 0
        params.maxThreshold = 255
        params.filterByColor = True
        params.minArea = 0
        params.filterByArea = True

        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0

        # Filter by Inertia
        params.filterByInertia = True
        params.minInertiaRatio = 0
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0

        params.minDistBetweenBlobs = 0

        detector = cv2.SimpleBlobDetector_create(params)
        keypoints = detector.detect(img)

        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(img, keypoints, np.array([]), (0, 0, 255),
                                              cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        if self.show:
            self.Show(self.win, im_with_keypoints)

        return im_with_keypoints

    def BG2Blk(self, img):
        # Convert BG to black
        black_mask = cv2.inRange(img, 70, 87)
        if self.show:
            self.Show(self.win, black_mask)

        img[np.where((black_mask == [255]))] = [0]
        if self.show:
            self.Show(self.win, img)

        return img

    def HistEq(self, img):
        # Histogram equalization
        img = cv2.equalizeHist(img)
        if self.show:
            self.Show(self.win, img)
        return img

    def PrintHist(self,img):
        # Histogram Print
        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist(img, [i], None, [256], [0, 256])
            pyplot.plot(histr, color=col)
            pyplot.xlim([0, 256])
        pyplot.show()

    def Threshold(self,img):
        # Thresholding
        ret, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        if self.show:
            self.Show(self.win, thresh)

        return thresh

    def kMeans(self,img):
        # K-Means
        # define criteria, number of clusters(K) and apply kmeans()
        Z = img.reshape((-1, 3))
        Z = np.float32(Z)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 3
        ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

        # Now convert back into uint8, and make original image
        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        if self.show:
            self.Show(self.win, res2)
        return res2

    def Canny(self, img):
        # Canny Edge
        cannyimg = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        cannyimg = cv2.Canny(cannyimg, 20, 100)

        if self.show:
            self.Show(self.win, cannyimg)
        return cannyimg

    def Contour(self, img):
        grad_x = cv2.Sobel(img, cv2.CV_8U, 1, 0, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
        # Gradient-Y
        # grad_y = cv.Scharr(gray,ddepth,0,1)
        grad_y = cv2.Sobel(img, cv2.CV_8U, 0, 1, ksize=3, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

        abs_grad_x = cv2.convertScaleAbs(grad_x)
        abs_grad_y = cv2.convertScaleAbs(grad_y)

        sobel = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)
        laplacian = cv2.Laplacian(img, cv2.CV_8U)
        if self.show:
            #laplacian = cv2.GaussianBlur(laplacian, (5, 5), 0)
            self.Show(self.win, laplacian)
        return laplacian

    def Morph(self, img):
        print("morph")
        kernel = np.ones((3, 3), np.uint8)
        closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel,iterations=5)
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

        if self.show:
            #self.Show(self.win, opening)
            self.Show(self.win, closing)

        kernel = np.ones((2, 2), np.uint8)
        dilation = cv2.dilate(img, kernel, iterations=5)

        if self.show:
            self.Show(self.win, dilation)

        kernel = np.ones((2, 2), np.uint8)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

        if self.show:
            self.Show(self.win, closing)

        erosion = cv2.erode(closing, kernel, iterations=10)

        return erosion

    def ChangeParams(self):
        for func in self.parameters.keys():
            for param in func.keys():
                p = self.parameters[func][param]
                self.parameters[func][param] = p
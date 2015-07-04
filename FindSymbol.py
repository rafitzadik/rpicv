import cv2
import numpy as np

MIN_MATCH_COUNT = 5

class findSymbol(object):
    def __init__(self, symbol_img_filename):
        self.img = cv2.imread(symbol_img_filename, 0)
        self.orb = cv2.ORB_create()
        #kp = self.orb.detect(img, None)
        self.kp1, self.des1 = self.orb.detectAndCompute(self.img,None)

    def find(self,img2):
        #kp = self.orb.detect(img2,None)
        kp2, des2 = self.orb.detectAndCompute(img2, None)
        if (des2 == None):
            return None, img2
        
        FLANN_INDEX_LSH = 6
        
        index_params = dict(algorithm = FLANN_INDEX_LSH,
                            table_number = 6,
                            key_size = 12,
                            multi_probe_level = 1)
        search_params = dict(checks = 50)

        flann = cv2.FlannBasedMatcher(index_params, search_params)

        #print len(self.des1), len(des2)
        matches = flann.knnMatch(self.des1, des2, k=2)
        #print len(matches)
        good = []
        if len(matches) < MIN_MATCH_COUNT:
            return None, img2 #not enough matches, forget about "good"

        #print matches[0]
        for m in matches:
            if len(m) == 2 and m[0].distance < 0.7 * m[1].distance:
                good.append(m[0])
        print 'matches: ', len(matches), 'good: ', len(good)
        #cv2.drawKeypoints(img2, kp2, img2, color=(0,255,0),flags=0)
##        bf = cv2.BFMatcher(cv2.NORM_HAMMIN, crossCheck=True)
##        matches = bf.match(self.des1, des2)
##        matches = sorted(matches, key = lambda x:x.dstance)
##        good = matches [:MIN_MATCH_COUNT]

        if len(good) > MIN_MATCH_COUNT:
            src_pts = np.float32([ self.kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            if (mask != None):
                matchesMask = mask.ravel().tolist()
                h,w = self.img.shape
                pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
                dst = cv2.perspectiveTransform(pts,M)
                img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
                return np.int32(dst), img2
        return None, img2

if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    finder = findSymbol("rpi-logo.jpg")

##    frame = cv2.imread("rpi-alt.jpg")
##    area, img = finder.find(frame)
##    cv2.imshow('frame', img)
##    cv2.waitKey(0)
    
    while (1):
        _, frame = cap.read()
        area, img = finder.find(frame)
        cv2.imshow('frame', img)
        k = cv2.waitKey(5) & 0xff
        if k == 27:
            break

    cv2.destroyAllWindows()

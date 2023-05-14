import cv2
import numpy as np
from matplotlib import pyplot as plt

imgL = cv2.imread('./Camera_data/Task_2/imgL.jpg')
imgR = cv2.imread('./Camera_data/Task_2/imgR.jpg')
imgL = cv2.resize(imgL, dsize=None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR_EXACT)
imgR = cv2.resize(imgR, dsize=None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR_EXACT)

imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY)

def get_keypoints_and_descriptors(imgL, imgR):

    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(imgL, None)
    kp2, des2 = orb.detectAndCompute(imgR, None)

    ############## Using FLANN matcher ##############
    FLANN_INDEX_LSH = 6
    index_params = dict(
        algorithm=FLANN_INDEX_LSH,
        table_number=6,  # 12
        key_size=12,  # 20
        multi_probe_level=1,
    )  # 2
    search_params = dict(checks=50) 
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    flann_match_pairs = flann.knnMatch(des1, des2, k=2)
    return kp1, des1, kp2, des2, flann_match_pairs

def filter(matches, ratio_threshold=0.6):
    filtered_matches = []
    for m, n in matches:
        if m.distance < ratio_threshold * n.distance:
            filtered_matches.append(m)
    return filtered_matches

def draw_matches(imgL, imgR, kp1, des1, kp2, des2, flann_match_pairs):
    """Draw the first 8 mathces between the left and right images."""
    # https://docs.opencv.org/4.2.0/d4/d5d/group__features2d__draw.html
    # https://docs.opencv.org/2.4/modules/features2d/doc/common_interfaces_of_descriptor_matchers.html
    img = cv2.drawMatches(
        imgL,
        kp1,
        imgR,
        kp2,
        flann_match_pairs[:8],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
    )
    plt.figure(figsize=(10,5))
    plt.imshow(img)
    plt.title('Matches')
    plt.show()
    cv2.imwrite("ORB_FLANN_Matches.png", img)
    cv2.waitKey(0)


kp1, des1, kp2, des2, flann_match_pairs = get_keypoints_and_descriptors(imgL, imgR)
good_matches = filter(flann_match_pairs, 0.6)
draw_matches(imgL, imgR, kp1, des1, kp2, des2, good_matches)   

def compute_fundamental_matrix(matches, kp1, kp2, method=cv2.FM_RANSAC):

    pts1, pts2 = [], []
    fundamental_matrix, inliers = None, None
    for m in matches[:8]:
        pts1.append(kp1[m.queryIdx].pt)
        pts2.append(kp2[m.trainIdx].pt)
    if pts1 and pts2:
        fundamental_matrix, inliers = cv2.findFundamentalMat(
            np.float32(pts1),
            np.float32(pts2),
            method=method,
        )
    return fundamental_matrix, inliers, pts1, pts2
F, I, points1, points2 = compute_fundamental_matrix(good_matches, kp1, kp2)
h1, w1 = imgL.shape
h2, w2 = imgR.shape
thresh = 0
_, H1, H2 = cv2.stereoRectifyUncalibrated(  
    np.float32(points1), np.float32(points2), F, imgSize=(w1, h1), threshold=thresh,
)
imgL_undistorted = cv2.warpPerspective(imgL, H1, (w1, h1))
imgR_undistorted = cv2.warpPerspective(imgR, H2, (w2, h2))

plt.subplot(121); plt.imshow(imgL_undistorted, cmap='gray'); plt.title('Left camera')
plt.subplot(122); plt.imshow(imgR_undistorted, cmap='gray'); plt.title('Right camera')
plt.show()

imgL = imgL_undistorted
imgR = imgR_undistorted
window_size = 5 
nDisp = 16*8

left_matcher = cv2.StereoSGBM_create(
    minDisparity=-1, # the disparity only goes in one direction
    numDisparities=nDisp, # each disparity value is multiplied with 16 to create better precision.
    blockSize=window_size, # the size of the block to compare
    P1=8 * 3 * window_size,
    P2=32 * 3 * window_size,
    disp12MaxDiff=12,
    uniquenessRatio=1,
    speckleWindowSize=5,
    speckleRange=5,
    preFilterCap=63,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
)
right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
displ = left_matcher.compute(imgL, imgR) #.astype(np.float32)/16
dispr = right_matcher.compute(imgR, imgL)  #.astype(np.float32)/16
lmbda = 5000
sigma = 3

wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
wls_filter.setLambda(lmbda)
wls_filter.setSigmaColor(sigma)
displ = np.int16(displ)
dispr = np.int16(dispr)
filteredImg = wls_filter.filter(displ, imgL, None, dispr)
mean = np.mean(filteredImg[filteredImg != 0])
filteredImg[filteredImg == 0] = mean
filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=1, alpha=255, norm_type=cv2.NORM_MINMAX)
plt.imshow(filteredImg.astype(np.float32)/16, cmap='gray')
plt.colorbar(fraction=0.026, pad=0.04)
plt.title('Refined disparity map')
plt.show()
cv2.imwrite('undistorted_L.png', imgL_undistorted)
cv2.imwrite('undistorted_R.png', imgR_undistorted)
cv2.imwrite('result_calibrated.jpg', filteredImg)

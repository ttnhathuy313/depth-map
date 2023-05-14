import cv2
import numpy as np
from matplotlib import pyplot as plt
import sys


cam_matrix = np.array([[1379.4102426941595, 0.00000000e+00, 648.173298],
                       [0.00000000e+00, 1255.63713, 257.079780],
                       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

def process(imgL, imgR):
    # imgL = cv2.resize(imgL, (640 >> 1, 480 >> 1), interpolation = cv2.INTER_LINEAR_EXACT)
    # imgR = cv2.resize(imgR, (640 >> 1, 480 >> 1), interpolation = cv2.INTER_LINEAR_EXACT) 
    
    # imgL = cv2.resize(imgL, dsize = None, fx = 0.5, fy = 0.5)
    # imgR = cv2.resize(imgR, dsize = None, fx = 0.5, fy = 0.5)

    imgL = cv2.cvtColor(imgL, cv2.COLOR_BGR2GRAY)
    imgR = cv2.cvtColor(imgR, cv2.COLOR_BGR2GRAY) 
    imgL = cv2.resize(imgL, dsize = None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_LINEAR_EXACT)
    imgR = cv2.resize(imgR, dsize = None, fx = 0.5, fy = 0.5, interpolation = cv2.INTER_LINEAR_EXACT)
    width = imgL.shape[1]
    height = imgL.shape[0]


    # plt.subplot(121); plt.imshow(imgL, cmap='gray'); plt.title('Left')
    # plt.subplot(122); plt.imshow(imgR, cmap='gray'); plt.title('Right')
    # plt.show()

    window_size = 5  # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
    nDisp = 16*8

    left_matcher = cv2.StereoSGBM_create(
        minDisparity=-1,
        numDisparities=nDisp,  # max_disp has to be dividable by 16 f. E. HH 192, 256
        blockSize=window_size,
        P1=8 * 3 * window_size,
        # wsize default 3; 5; 7 for SGBM reduced size image; 15 for SGBM full size image (1300px and above); 5 Works nicely
        P2=32 * 3 * window_size,
        disp12MaxDiff=12,
        uniquenessRatio=1,
        speckleWindowSize=5,
        speckleRange=5,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
    # FILTER Parameters
    lmbda = 5000
    sigma = 3

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)

    wls_filter.setSigmaColor(sigma)
    displ = left_matcher.compute(imgL, imgR) #.astype(np.float32)/16
    dispr = right_matcher.compute(imgR, imgL)  #.astype(np.float32)/16
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filteredImg = wls_filter.filter(displ, imgL, None, dispr) / 16  # important to put "imgL" here!!!

    filteredImg = np.uint8(filteredImg)
    mean = np.mean(filteredImg[filteredImg != 0])
    filteredImg[filteredImg == 0] = mean
    
    base_line = 0.078
    f = cam_matrix[0][0] * 0.25
    depth = np.zeros_like(filteredImg)
    depth = base_line * f / filteredImg
    
    filteredImg = cv2.normalize(src=filteredImg, dst=filteredImg, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
    
    # ### 3D plot
    # coordinates = []
    # baseline = 30.5
    # f = 0.8
    # for i in range(displ.shape[0]):
    #     for j in range(displ.shape[1]):
    #         disp = displ[i, j]/16
    #         coordinates.append([i, j, 170.0/disp])
    # coordinates = np.array(coordinates)
    ### 3D plot end
    
    distance = np.array(depth[(height >> 1)-20:(height >> 1) + 20,(width >> 1)-20:(width >> 1) + 20][(depth[(height >> 1)-20:(height >> 1) + 20,(width >> 1)-20 :(width >> 1) + 20] != np.inf)]).mean()
    
    
    return (filteredImg, distance)


for i in range(1, 2):
    nDigit = len(str(i))
    id = '0' * (6 - nDigit) + str(i)
    imgL = cv2.imread('../part1/part1/' + id + '/left.png')
    imgR = cv2.imread('../part1/part1/' + id + '/right.png')
    id = str(100)
    imgL = cv2.imread('./Camera_data/Task_1/L_' + id + '.jpg')
    imgR = cv2.imread('./Camera_data/Task_1/R_' + id + '.jpg')
    result, distance = process(imgL, imgR)
    print(distance)
    plt.figure(figsize=(20,10))
    plt.subplot(133); plt.imshow(result, cmap='gray'); plt.title('Disparity Map')
    plt.subplot(131); plt.imshow(imgL[:,:,::-1]); plt.title('Left camera')
    plt.subplot(132); plt.imshow(imgR[:,:,::-1]); plt.title('Right camera')
    plt.show()
    
    
cam1 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cam2 = cv2.VideoCapture(1, cv2.CAP_DSHOW)

while (True):
    ret, frame1 = cam1.read()
    ret, frame2 = cam2.read()
    result, distance = process(frame1, frame2)
    cv2.putText(frame1, f'Distance: {distance} m', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    if (distance < 0.8):
        cv2.putText(frame1, 'WARNING', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.imshow('frame1', frame1)
    cv2.imshow('frame2', frame2)
    cv2.imshow('disparity', result)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
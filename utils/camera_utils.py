import cv2
import numpy as np
from PIL import Image

# Logitech C920 Camera calibration and distortion parameters (OpenCV)
# Camera.fx: 1394.6027293299926
# Camera.fy: 1394.6027293299926
# Camera.cx: 995.588675691456
# Camera.cy: 599.3212928484164

# Camera.k1: 0.11480806073904032
# Camera.k2: -0.21946985653851792
# Camera.p1: 0.0012002116999769957
# Camera.p2: 0.008564577708855225
# Camera.k3: 0.11274677130853494

# K_rgb = np.array([[1394.6027293299926, 0, 995.588675691456], [0, 1394.6027293299926, 599.3212928484164], [0,0,1]])
# D_rgb = np.array([[0.11480806073904032],[-0.21946985653851792], [0.0012002116999769957], [0.008564577708855225], [0.11274677130853494]])

K_th = np.array([[527.5158969253331, 0.0, 327.8057399], [0.0, 527.3384216620225, 259.7357201275458], [0.0, 0.0, 1.0]])
D_th = np.array([[-0.343267017336368],[0.168923177132998],[-0.001271279542797],[-0.048877613602393]])

H_1 = np.array([[ 9.96674965e-01, -7.65072006e-02,  1.93124333e+01],
       [-5.05151544e-03,  1.27628456e+00, -1.01210839e+01],
       [-3.04540412e-05, -2.08364467e-04,  1.00000000e+00]])

H_3 = np.array([[ 9.60834096e-01,  2.47301114e-01,  8.23394373e+00],
       [-1.00833994e-01,  1.57852285e+00, -4.99828409e+01],
       [-4.99586936e-04,  8.78631225e-04,  1.00000000e+00]])

H_5 = np.array([[ 9.30997052e-01, -5.80885702e-01,  9.06581187e+01],
       [ 2.05941603e-02,  8.63955671e-01, -8.39343876e+00],
       [ 1.15808854e-04, -1.34273776e-03,  1.00000000e+00]])

H = H_5

def get_camera_rgb(cam='logitech_c920'):
    if cam == 'logitech_c920':
        #camera_rgb = cv2.VideoCapture("v4l2src device=/dev/video0 ! image/jpeg,framerate=30/1,width=1920, height=1080,type=video ! jpegdec ! videoconvert ! video/x-raw ! appsink", cv2.CAP_GSTREAMER)
        camera_rgb = cv2.VideoCapture("v4l2src device=/dev/video0 ! image/jpeg,framerate=30/1,width=1920, height=1080,type=video ! jpegdec ! videoconvert ! video/x-raw ! appsink", cv2.CAP_GSTREAMER)
        #camera_rgb = cv2.VideoCapture(0)
        #camera_rgb.set(cv2.CAP_DSHOW + 0); // instead of capture.set(0);
        #camera_rgb = WebcamVideoStream(src=0).start()
        #camera_rgb.set(cv2.CAP_PROP_FPS, 60)
        K_rgb = np.array([[1394.6027293299926, 0, 995.588675691456], [0, 1394.6027293299926, 599.3212928484164], [0,0,1]])
        D_rgb = np.array([[0.11480806073904032],[-0.21946985653851792], [0.0012002116999769957], [0.008564577708855225], [0.11274677130853494]])
    return camera_rgb, K_rgb, D_rgb

def frame_to_pil(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = Image.fromarray(frame)
    return frame

def pil_to_frame(pil_image):
    open_cv_image = np.array(pil_image) 
    # Convert RGB to BGR 
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image

# def pil_to_frame(frame):
#     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     frame = Image.fromarray(frame)
#     return frame

def capture_raw_thermal(w=640, h=480, thermal_file='/dev/shm/imgRAW.bin'):
    img = np.fromfile(thermal_file, dtype='uint16')
    if img.shape[0] == h*w:
        img = np.uint8(255*(img.reshape((h,w))/((2**16)-1.0)))
        #img = cv2.equalizeHist(img)
        img3 = np.zeros((h,w,3))    
        img3[:,:,0] = img 
        img3[:,:,1] = img 
        img3[:,:,2] = img
        return True, np.uint8(img3)
    else:
        return False, None

# def capture_raw_thermal(w=640, h=480, l=2):
#     img = np.fromfile('/home/heliaus/customer_640/examples/imgRAW.bin', dtype='uint16')
#     if img.shape[0] == h*w:

#         img = img.reshape((h,w))
#         mean = img.mean()
#         std = img.std()
#         img_norm = (img - mean)/std
#         img = np.clip((img_norm + l) / 2*l, 0, 1)
#         img = np.uint8(img*255)
        
#         #img = np.uint8(255*(img.reshape((h,w))/((2**16)-1.0)))
#         img = cv2.equalizeHist(img)

#         img3 = np.zeros((h,w,3))    
#         img3[:,:,0] = img 
#         img3[:,:,1] = img 
#         img3[:,:,2] = img
        
#         return True, np.uint8(img3)
#     else:
#         return False, None

def chessboard_calib(frame, nline=3, ncol=3):
    ## termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ## processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (ncol, nline), None)
    # If found, add object points, image points (after refining them)
    print(ret)
    if ret == True:
        #objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners, (ncol, nline), (-1,-1), criteria)
        #imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(frame, (ncol, nline), corners, ret)
    return frame

eps = 0.00001

def otsu_t(frame):
    # Convert to grayscale and Otsu's threshold
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    return thresh

def otsu_calib(image):
    # Remove border
    kernel_vertical = cv2.getStructuringElement(cv2.MORPH_RECT, (1,50))
    temp1 = 255 - cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel_vertical)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
    temp2 = 255 - cv2.morphologyEx(image, cv2.MORPH_CLOSE, horizontal_kernel)
    temp3 = cv2.add(temp1, temp2)
    result = cv2.add(temp3, image)

    # Convert to grayscale and Otsu's threshold
    gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

    # Find contours and filter using contour area
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(result, (x, y), (x + w, y + h), (36,255,12), 2)
    return result
    
def undistort(img, K, D, fisheye=False): 
    h,w = img.shape[:2]
    DIM = (w, h)
    if fisheye:
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    else:
        map1, map2 = cv2.initUndistortRectifyMap(K, D, np.eye(3), K, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    return undistorted_img

def mser_calib(frame):
    vis = frame.copy()
    #mser = cv2.MSER_create(_min_area=600, _max_area=5000, _max_variation=1, _delta=10)
    mser = cv2.MSER_create(_min_area=600, _max_area=5000)

    #mser = cv2.MSER_create()

    regions, bbox = mser.detectRegions(frame)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
    #cv2.polylines(vis, hulls, 1, (0, 0, 255))

    # compute the center of the contours
    centers = []
    corners = []
    boxes = []
    for c in hulls:
        M = cv2.moments(c)    
        cX = int(M["m10"] / (eps+M["m00"]))
        cY = int(M["m01"] / (eps+M["m00"]))
        x,y,w,h = cv2.boundingRect(c)
        if w < h*2 and w > h/2:
            patch_std = np.std(frame[y:y+h,x:x+w,:])
            patch_mean = np.mean(frame[y:y+h,x:x+w,:])
            if patch_std < 120 and patch_mean > 60:
                #vis = cv2.circle(vis, (cX,cY), radius=0, color=(0, 255, 0), thickness=4)
                #cv2.rectangle(vis,(x,y),(x+w,y+h),(200,0,0),2)
                centers.append((cX, cY))
                boxes.append([x,y,w,h])

    # filter out overlapping boxes
    #cv2.groupRectangles(boxes,1,0)
    centers_filt = []
    for i in range(len(boxes)):
        x,y,w,h = boxes[i]
        filter_out = False
        for j in range(len(boxes)):
            x_n,y_n,w_n,h_n = boxes[j]
            if j != i and (abs(x-x_n) <= 2 or abs(y-y_n) <= 2):
                if w*h <= w_n*h_n:
                    filter_out = True
        if not filter_out:
            cv2.rectangle(vis,(x,y),(x+w,y+h),(200,0,0),2)
            corners.append((x,y))
            centers_filt.append((x,y))
                      
    centers = np.array(centers_filt)
    corners = np.array(corners)

    # sort points
    try:
        x, y = centers[:,0], centers[:,1]
        sort_idx = np.argsort(x)
        centers[:,0] = x[sort_idx]
        centers[:,1] = y[sort_idx]
    except:
        pass
    
    return vis, centers

def orb_calib(img1, img2):
    FLANN_INDEX_KDTREE = 1
    # Initiate SIFT detector
    sift = cv2.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks=50)   # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0,0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i,(m,n) in enumerate(matches):
        if m.distance < 0.7*n.distance:
            matchesMask[i]=[1,0]
    draw_params = dict(matchColor = (0,255,0),
                       singlePointColor = (255,0,0),
                       matchesMask = matchesMask,
                       flags = cv2.DrawMatchesFlags_DEFAULT)
    img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,matches,None,**draw_params)
    return img3

def zoom(frame, zoom_size):
    # Resizes the image/video frame to the specified amount of "zoomSize".
    # A zoomSize of "2", for example, will double the canvas size
    frame = cv2.resize(frame, (0,0), fx=zoom_size, fy=zoom_size) 

    #cv2Object = imutils.resize(cv2Object, width=(zoomSize * cv2Object.shape[1]))
    # center is simply half of the height & width (y/2,x/2)
    center = (frame.shape[0]//2, frame.shape[1]//2)
    # cropScale represents the top left corner of the cropped frame (y/x)
    crop = (int(center[0]/zoom_size), int(center[1]/zoom_size))
    # The image/video frame is cropped to the center with a size of the original picture
    # image[y1:y2,x1:x2] is used to iterate and grab a portion of an image
    # (y1,x1) is the top left corner and (y2,x1) is the bottom right corner of new cropped frame.
    frame = frame[crop[0]:(center[0] + crop[0]), crop[1]:(center[1] + crop[1])]
    return frame
    

# found homography                                                                                                                                                                                                  │
# [[ 9.75018798e-01 -3.33039231e-02  2.42402091e+01]                                                                                                                                                                │
#  [-3.33672191e-02  1.26041925e+00  6.55453385e+01]                                                                                                                                                                │
#  [-9.01801291e-05 -1.65166210e-04  1.00000000e+00]]    

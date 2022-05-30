import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_filtered_image(image, action):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    filtered = None
    if action == 'NO_FILTER':
        filtered = image

# 色彩空間轉換
    # GRAY
    elif action == 'GRAY':
        filtered = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # XYZ
    elif action == 'XYZ':
        filtered = cv2.cvtColor(img, cv2.COLOR_BGR2XYZ)
    # YCrCb
    elif action == 'YCrCb':
        filtered = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    # HSV
    elif action == 'HSV':
        filtered = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # HLS
    elif action == 'HLS':
        filtered = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    # CIEL *a*b*
    elif action == 'CIElab':
        filtered = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    # CIEL *u*v*
    elif action == 'CIEluv':
        filtered = cv2.cvtColor(img, cv2.COLOR_BGR2Luv)

# 設定值處理
    # THRESH_BINARY
    elif action == 'BINARY':
        def BINARY(k):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, filtered = cv2.threshold(gray, k+ratio, 255, cv2.THRESH_BINARY)
            cv2.imshow("binary demo", filtered)
        lowThreshold = 0
        max_lowThreshold = 150
        ratio = 1
        cv2.namedWindow('binary demo')
        cv2.createTrackbar('Min threshold', 'binary demo',
                           lowThreshold, max_lowThreshold, BINARY)
        BINARY(0)  # initialization
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()

    # THRESH_BINARY_INV
    elif action == 'BINARY_INV':
        def BINARY_INV(k):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, filtered = cv2.threshold(
                gray, k+ratio, 255, cv2.THRESH_BINARY_INV)
            cv2.imshow("binary_inv demo", filtered)
        lowThreshold = 0
        max_lowThreshold = 150
        ratio = 1
        cv2.namedWindow('binary_inv demo')
        cv2.createTrackbar('Min threshold', 'binary_inv demo',
                           lowThreshold, max_lowThreshold, BINARY_INV)
        BINARY_INV(0)  # initialization
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()

    # THRESH_TRUNC 截斷這定值化處理
    elif action == 'TRUNC':
        def TRUNC(k):
            _, filtered = cv2.threshold(img, k+ratio, 255, cv2.THRESH_TRUNC)
            cv2.imshow("trunc demo", filtered)
        lowThreshold = 0
        max_lowThreshold = 150
        ratio = 1
        cv2.namedWindow('trunc demo')
        cv2.createTrackbar('Min threshold', 'trunc demo',
                           lowThreshold, max_lowThreshold, TRUNC)
        TRUNC(0)  # initialization
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()

    # THRESH_TOZERO 低設定值零處理
    elif action == 'TOZERO':
        def TOZERO(k):
            _, filtered = cv2.threshold(img, k+ratio, 255, cv2.THRESH_TOZERO)
            cv2.imshow("tozero demo", filtered)
        lowThreshold = 0
        max_lowThreshold = 150
        ratio = 1
        cv2.namedWindow('tozero demo')
        cv2.createTrackbar('Min threshold', 'tozero demo',
                           lowThreshold, max_lowThreshold, TOZERO)
        TOZERO(0)  # initialization
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
    # THRESH_TOZERO_INV 超設定值零處理
    elif action == 'TOZERO_INV':
        def TOZERO_INV(k):
            _, filtered = cv2.threshold(
                img, k+ratio, 255, cv2.THRESH_TOZERO_INV)
            cv2.imshow("tozero_inv demo", filtered)
        lowThreshold = 0
        max_lowThreshold = 150
        ratio = 1
        cv2.namedWindow('tozero_inv demo')
        cv2.createTrackbar('Min threshold', 'tozero_inv demo',
                           lowThreshold, max_lowThreshold, TOZERO_INV)
        TOZERO_INV(0)  # initialization
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()

    # THRESH_OTSU 大津二值化
    elif action == 'OTSU':
        def OTSU(k):
            image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, filtered = cv2.threshold(
                image, k+ratio, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            cv2.imshow("otsu demo", filtered)
        lowThreshold = 0
        max_lowThreshold = 150
        ratio = 1
        cv2.namedWindow('otsu demo')
        cv2.createTrackbar('Min threshold', 'otsu demo',
                           lowThreshold, max_lowThreshold, OTSU)
        OTSU(0)  # initialization
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
# 影像平滑處理
    # 均值濾波
    elif action == 'BLURRED':
        def BLURRED(k):
            filtered = cv2.blur(img, (k+ratio, k+ratio))
            cv2.imshow("blur demo", filtered)
        lowThreshold = 0
        max_lowThreshold = 100
        ratio = 1
        k = 0
        cv2.namedWindow('blur demo')
        cv2.createTrackbar('Min threshold', 'blur demo',
                           lowThreshold, max_lowThreshold, BLURRED)
        BLURRED(0)  # initialization
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
    # 方框濾波
    elif action == 'BOX_FILTER':
        def BOX_FILTER(k):
            filtered = cv2.boxFilter(img, -1, (k+ratio, k+ratio))
            cv2.imshow("boxFilter demo", filtered)
        lowThreshold = 0
        max_lowThreshold = 100
        ratio = 1
        k = 0
        cv2.namedWindow('boxFilter demo')
        cv2.createTrackbar('Min threshold', 'boxFilter demo',
                           lowThreshold, max_lowThreshold, BOX_FILTER)
        BOX_FILTER(0)  # initialization
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
    # 高斯濾波
    elif action == 'GAUSSIANBLUR':
        def MEDIANBLUR(k):
            filtered = cv2.GaussianBlur(img, (k+ratio, k+ratio), 0)
            cv2.imshow("GaussianBlur demo", filtered)
        lowThreshold = 0
        max_lowThreshold = 100
        ratio = 1
        k = 0
        cv2.namedWindow('GaussianBlur demo')
        cv2.createTrackbar('Min threshold', 'GaussianBlur demo',
                           lowThreshold, max_lowThreshold, MEDIANBLUR)
        MEDIANBLUR(0)  # initialization
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
    # 中值濾波
    elif action == 'MEDIANBLUR':
        def MEDIANBLUR(k):
            filtered = cv2.medianBlur(img, k+ratio)
            cv2.imshow("medianBlur demo", filtered)
        lowThreshold = 0
        max_lowThreshold = 100
        ratio = 1
        k = 0
        cv2.namedWindow('medianBlur demo')
        cv2.createTrackbar('Min threshold', 'medianBlur demo',
                           lowThreshold, max_lowThreshold, MEDIANBLUR)
        MEDIANBLUR(0)  # initialization
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
    # 雙邊濾波
    elif action == 'BILATERALFILTER':
        def BILATERALFILTER(k):
            filtered = cv2.bilateralFilter(img, k+ratio, 100, 100)
            cv2.imshow("bilateralFilter demo", filtered)
        lowThreshold = 0
        max_lowThreshold = 100
        ratio = 1
        k = 0
        cv2.namedWindow('bilateralFilter demo')
        cv2.createTrackbar('Min threshold', 'bilateralFilter demo',
                           lowThreshold, max_lowThreshold, BILATERALFILTER)
        BILATERALFILTER(0)  # initialization
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
# 形態學操作
    # 腐蝕
    elif action == 'EROSION':
        def EROSION(k):
            kernel = np.ones((k+ratio, k+ratio), np.uint8)
            filtered = cv2.erode(img, kernel)
            cv2.imshow("eposion demo", filtered)
        lowThreshold = 0
        max_lowThreshold = 100
        ratio = 1
        k = 0
        cv2.namedWindow('eposion demo')
        cv2.createTrackbar('Min threshold', 'eposion demo',
                           lowThreshold, max_lowThreshold, EROSION)
        EROSION(0)  # initialization
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
    # 膨脹
    elif action == 'DILATION':
        def DILATION(k):
            kernel = np.ones((k+ratio, k+ratio), np.uint8)
            filtered = cv2.dilate(img, kernel)
            cv2.imshow("dilation demo", filtered)
        lowThreshold = 0
        max_lowThreshold = 100
        ratio = 1
        k = 0
        cv2.namedWindow('dilation demo')
        cv2.createTrackbar('Min threshold', 'dilation demo',
                           lowThreshold, max_lowThreshold, DILATION)
        DILATION(0)  # initialization
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
    # 開運算
    elif action == 'MORPH_OPEN':
        def MORPH_OPEN(k):
            kernel = np.ones((k+ratio, k+ratio), np.uint8)
            filtered = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            cv2.imshow("open demo", filtered)
        lowThreshold = 0
        max_lowThreshold = 100
        ratio = 1
        k = 0
        cv2.namedWindow('open demo')
        cv2.createTrackbar('Min threshold', 'open demo',
                           lowThreshold, max_lowThreshold, MORPH_OPEN)
        MORPH_OPEN(0)  # initialization
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
    # 閉運算
    elif action == 'MORPH_CLOSE':
        def MORPH_CLOSE(k):
            kernel = np.ones((k+ratio, k+ratio), np.uint8)
            filtered = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            cv2.imshow("close demo", filtered)
        lowThreshold = 0
        max_lowThreshold = 100
        ratio = 1
        k = 0
        cv2.namedWindow('close demo')
        cv2.createTrackbar('Min threshold', 'close demo',
                           lowThreshold, max_lowThreshold, MORPH_CLOSE)
        MORPH_CLOSE(0)  # initialization
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
    # 頂帽運算
    elif action == 'MORPH_TOPHAT':
        def MORPH_TOPHAT(k):
            kernel = np.ones((k+ratio, k+ratio), np.uint8)
            filtered = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
            cv2.imshow("tophat demo", filtered)
        lowThreshold = 0
        max_lowThreshold = 100
        ratio = 1
        k = 0
        cv2.namedWindow('tophat demo')
        cv2.createTrackbar('Min threshold', 'tophat demo',
                           lowThreshold, max_lowThreshold, MORPH_TOPHAT)
        MORPH_TOPHAT(0)  # initialization
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
    # 黑帽運算
    elif action == 'MORPH_BLACKHAT':
        def MORPH_BLACKHAT(k):
            kernel = np.ones(k+ratio, np.uint8)
            filtered = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
            cv2.imshow("blackhat demo", filtered)
        lowThreshold = 0
        max_lowThreshold = 100
        ratio = 1
        k = 0
        cv2.namedWindow('blackhat demo')
        cv2.createTrackbar('Min threshold', 'blackhat demo',
                           lowThreshold, max_lowThreshold, MORPH_BLACKHAT)
        MORPH_BLACKHAT(0)  # initialization
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
# 影像梯度
    # sobel
    elif action == 'SOBEL':
        def SobelThreshold(k):
            x = cv2.Sobel(img, cv2.CV_16S, 1, 0)
            y = cv2.Sobel(img, cv2.CV_16S, 0, 1)
            absX = cv2.convertScaleAbs(x)  # 轉回uint8
            absY = cv2.convertScaleAbs(y)
            filtered = cv2.addWeighted(absX, k+ratio, absY, k+ratio, 0)
            cv2.imshow("sobel demo", filtered)
        lowThreshold = 0
        max_lowThreshold = 10
        ratio = 1
        k = 0
        cv2.namedWindow('sobel demo')
        cv2.createTrackbar('Min threshold', 'sobel demo',
                           lowThreshold, max_lowThreshold, SobelThreshold)
        SobelThreshold(0)  # initialization
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
    # scharr
    elif action == 'SCHARR':
        def ScharrThreshold(k):
            scharrx = cv2.Scharr(img, cv2.CV_64F, 1, 0)
            scharry = cv2.Scharr(img, cv2.CV_64F, 0, 1)
            scharrx = cv2.convertScaleAbs(scharrx)   # 转回uint8
            scharry = cv2.convertScaleAbs(scharry)
            filtered = cv2.addWeighted(scharrx, k+ratio, scharry, k+ratio, 0)
            cv2.imshow("scharr demo", filtered)
        lowThreshold = 0
        max_lowThreshold = 10
        ratio = 1
        k = 0
        cv2.namedWindow('scharr demo')
        cv2.createTrackbar('Min threshold', 'scharr demo',
                           lowThreshold, max_lowThreshold, ScharrThreshold)
        ScharrThreshold(0)  # initialization
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()
    # LAPLACIAN
    elif action == 'LAPLACIAN':
        def LaplacianThreshold(k):
            gray_lap = cv2.Laplacian(img, cv2.CV_16S, ksize=k+ratio)
            filtered = cv2.convertScaleAbs(gray_lap)
            cv2.imshow('laplacian demo', filtered)
        lowThreshold = 0
        max_lowThreshold = 10
        ratio = 1
        k = 1
        cv2.namedWindow('laplacian demo')
        cv2.createTrackbar('Min threshold', 'laplacian demo',
                           lowThreshold, max_lowThreshold, LaplacianThreshold)
        LaplacianThreshold(0)  # initialization
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()

    # canny
    elif action == 'CANNY':
        def CannyThreshold(lowThreshold):
            detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)
            detected_edges = cv2.Canny(
                detected_edges, lowThreshold, lowThreshold * ratio, apertureSize=kernel_size)
            # just add some colours to edges from original image.
            filtered = cv2.bitwise_and(img, img, mask=detected_edges)
            cv2.imshow('canny demo', filtered)

        lowThreshold = 0
        max_lowThreshold = 100
        ratio = 3
        kernel_size = 3
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.namedWindow('canny demo')
        cv2.createTrackbar('Min threshold', 'canny demo',
                           lowThreshold, max_lowThreshold, CannyThreshold)
        CannyThreshold(0)  # initialization
        if cv2.waitKey(0) == 27:
            cv2.destroyAllWindows()

    return filtered

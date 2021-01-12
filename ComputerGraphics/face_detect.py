import cv2
import numpy as np
from time import sleep
from matplotlib import pyplot as plt
import keyboard as key


"""
def Quantization():
    target = cv2.imread('TrippleFace.png')
    hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)  ## RGB--> GRAY

    equ = cv2.equalizeHist(hsvt)
    dst = np.hstack((hsvt, equ))

    print("quantizaion")
    
    cv2.imshow('quantization',dst)
    cv_wait()
    
    return dst
"""
def Quantization3():
    img = cv2.imread('TrippleFace.png')
    Z = img.reshape((-1, 3))

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 16              # K값을 높일수록 시간이 오래 걸림
    Z = np.float32(Z)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape(img.shape)

    cv2.imshow('quantization', res2)
    cv_wait()
    return res2

"""
def Quantization3():
    img = cv2.imread('TrippleFace.png')          
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 256        # K값을 높일수록 시간이 오래 걸림
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    cv2.imshow('res2',res2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return res2
"""
#양자화 작업 : RGB를 Gray로 바꾼 뒤 해야함
def Quantization2():
    src = cv2.imread('TrippleFace.png', cv2.IMREAD_COLOR)
    hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)    #채널 분리

    h = cv2.inRange(h, 0, 30)  #검사할 이미지, 하한값, 상한값
    face = cv2.bitwise_and(hsv, hsv, mask = h)
    face = cv2.cvtColor(face, cv2.COLOR_HSV2BGR)
    print("quantization")
    cv2.imshow("TrippleFace", face)
    cv_wait()
    return face

def Show_2Dhistogram(hsvt):
    print("H,S 요소에 대한 히스토그램입니다")
    target = cv2.imread('TrippleFace.png')
    #hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV) ## RGB-->HSV 포맷변환
    #cv2.imshow('hsv Image', hsvt)
    #cv2.waitKey(0)
    histt = cv2.calcHist([hsvt], [0, 1], None, [255, 255], [0, 180, 0, 256]) # row: H(색조) / col: S(채도)
    #'his'togram of 't'arget

    cv2.imshow('histogram', histt)
    cv_wait() #원하는 H,S 값을 구하기 힘든 -->  1) 흑백으로 나옴

    key.wait('Space bar')
    plt.imshow(histt, interpolation='nearest')  #  2) 이렇게 하면 색상의 구분이 가능해져

    close_plt()


#역투영 (신뢰도 맵을 구하는 과정)
def CalcBack(target):
    print('CalcBack 역투영')
    roi = cv2.imread('total_skin.jpg') # 원하는 곳의 이미지
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    #target = cv2.imread('TrippleFace.png') # 타겟이미지_목표
    hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)
    roihist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256]) # 원하는 곳의 이미지의 히스토그램
    cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX) # 해당 히스토그램을 0,255 로 정규화

    calc = cv2.calcBackProject([hsvt], [0,1], roihist, [0,180,0,256],1) #타겟이미지에서 원하는 이미지의 히스토그램을 이용해서 추출
    cv2.imshow('calc', calc)
    cv2.waitKey(0)

    cv_wait()
    return calc

#calc를 반환하는 CalcBack 함수와 같은 기능, 이미지 출력이 겹쳐서 imshow는 구현X / "return calc"만 구현
def CalcBack_no_image(target):
    # 역투영
    roi = cv2.imread('total_skin.jpg') # 원하는 곳의 이미지
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    #target = cv2.imread('TrippleFace.png') # 타겟이미지_목표
    hsvt = cv2.cvtColor(target, cv2.COLOR_BGR2HSV)

    roihist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256]) # 원하는 곳의 이미지의 히스토그램
    cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX) # 해당 히스토그램을 0,255 로 정규화
    calc = cv2.calcBackProject([hsvt], [0,1], roihist, [0,180,0,256],1) #타겟이미지에서 원하는 이미지의 히스토그램을 이용해서 추출
    #cv2.imshow('calc',calc)
    #cv2.waitKey(0)
    return calc


def Show_1Dhistogram(calc):
    # calc으로 1차원 히스토그램을 구현해 사용자가 임의로 임계값을 설정할 수 있다.
    print('Show 1D histogram')
    hist, bins = np.histogram(calc.flatten(), 256, [0, 256])
    plt.hist(calc.flatten(), 32, color='r')  # 2nd parameter --> bin <ex)256 -> 0~255 / 16 -> 0~15 & 16~31....
    #cv2.imshow('calc', calc)                          # 최적화한 bin의 값 ==> 32 / 32로 해야 눈으로 보기에 계산이 편해 어느 구간이 어느값인지..
    plt.xlim([0, 256])

    plt.ion()       # make blocking function to non blocking function
    plt.show()
    close_plt()

#이진화
def Thres(calc):
    #임계값을 1로 하고 그 이상은 흰색 그 아래는 검은색으로 설정.
    ret, binary_image = cv2.threshold(calc, 1, 255, cv2.THRESH_BINARY) #최적화된 임계값은 반복된 시도로 구하는게...
    cv2.imshow('threshold', binary_image)                               #제생각에는 1이 제일 뚜렷하게 구분되지 않나...
    print("이진화")
    cv2.waitKey(0)      # 얘를 빼면 안되네????
    cv_wait()


def OTSU(calc): #오츄 오리지널
    ret, binary_image = cv2.threshold(calc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('OTSU', binary_image)

    print("Threshold in OTSU algorithm: ", ret) #오츄알고리즘에서의 임계값 50
    cv_wait()

def Otsu_Gaussian(calc): #오츄 가우시안
    gau_calc = cv2.GaussianBlur(calc, (5,5), 0)
    ret3, binary_image = cv2.threshold(gau_calc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    print("Threshold in OTSU algorithm + Gaussian: ", ret3) # 가우시안 오츄 알고리즘에서의 임계값 34
    cv2.imshow('OTSU + Gaussian', binary_image)

    cv_wait()


def cv_wait():
    wait = cv2.waitKey(0)
    while (wait != 32):
        wait = cv2.waitKey(0)
        print(wait)
    cv2.destroyAllWindows()

def close_plt():
    plt.pause(0.1)
    key.wait('Space bar')
    #plt.clf()
    plt.close()

def main():

    print("press spacebar")
    QuantizedImage = Quantization3()    #모델 영상을 양자화
    # 양자화 작업 먼저 한뒤에 히스토그램 출력
    key.wait('Space bar')
    Show_2Dhistogram(QuantizedImage)
    key.wait('Space bar')
    CalcBack(QuantizedImage)
    key.wait('Space bar')
    Show_1Dhistogram(CalcBack_no_image(QuantizedImage))
    key.wait('Space bar')
    Thres(CalcBack_no_image(QuantizedImage))
    key.wait('Space bar')
    OTSU(CalcBack_no_image(QuantizedImage))
    key.wait('Space bar')
    Otsu_Gaussian(CalcBack_no_image(QuantizedImage))

if __name__=="__main__":
    main()
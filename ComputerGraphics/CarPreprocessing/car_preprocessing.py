import cv2, os
import numpy as np
from matplotlib import pyplot as plt
from time import sleep

clicked = []
# mouse callback handler

def mouse_handler(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONUP:
        clicked.append([x, y])
        print(clicked)

    img = cv2.imread('car.jpg')
    cv2.imshow('img', img)


def show_template():
    ori_img = cv2.imread('car.jpg')
    ##if event == cv2.EVENT_LBUTTONUP:
    # img = ori_img.copy()

    src_np = np.array([[125, 216], [340, 266], [340, 343], [125, 283]], dtype=np.float32)
    # perspective transform

    # width = max(np.linalg.norm(src_np[0] - src_np[1]), np.linalg.norm(src_np[2] - src_np[3]))
    # height = max(np.linalg.norm(src_np[0] - src_np[3]), np.linalg.norm(src_np[1] - src_np[2]))
    dst_np = np.array([[0, 0], [242, 0], [242, 77], [0, 77]], dtype=np.float32)


    M = cv2.getPerspectiveTransform(src=src_np,
                                    dst=dst_np)  # 영상의 자동차 번호판을 template에 맞추는 투영행렬(M, 변환행렬) 을 구해준다. 직선 의 성질은 유지하되, 평행은 유지 x
    result = cv2.warpPerspective(ori_img, M=M, dsize=(242, 77))  # 투영행렬(변환행렬)에 의해 결과물 출력

    ##resized = cv2.resize(result, dsize=(600, 300), interpolation=cv2.INTER_AREA)
    cv2.imshow('result', result)
    cv2.imwrite("result.jpg", result)

    return result

def show_template_no_image():
    ori_img = cv2.imread('car.jpg')

    ##if event == cv2.EVENT_LBUTTONUP:
    # img = ori_img.copy()

    src = []
    src.append([125.0, 216.0])
    src.append([340.0, 266.0])
    src.append([340.0, 343.0])
    src.append([125.0, 283.0])

    #print(src)
    # perspective transform
    src_np = np.array(src, dtype=np.float32)

    # width = max(np.linalg.norm(src_np[0] - src_np[1]), np.linalg.norm(src_np[2] - src_np[3]))
    # height = max(np.linalg.norm(src_np[0] - src_np[3]), np.linalg.norm(src_np[1] - src_np[2]))

    dst_np = np.array([
        [0, 0],
        [242, 0],
        [242, 77],
        [0, 77]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src=src_np,
                                    dst=dst_np)  # 영상의 자동차 번호판을 template에 맞추는 투영행렬(M, 변환행렬) 을 구해준다. 직선 의 성질은 유지하되, 평행은 유지 x
    result = cv2.warpPerspective(ori_img, M=M, dsize=(242, 77))  # 투영행렬(변환행렬)에 의해 결과물 출력

    return result


def wait():
    wait = cv2.waitKey(0)
    while (wait != 32):
        wait = cv2.waitKey(0)
        print(wait)


def Otsu_Gaussian(calc): #오츄 가우시안

    calc = cv2.cvtColor(calc, cv2.COLOR_BGR2GRAY)
    gau_calc = cv2.GaussianBlur(calc, (5,5), 0)
    ret3, binary_image = cv2.threshold(gau_calc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    print("Threshold in OTSU algorithm + Gaussian: ", ret3) # 가우시안 오츄 알고리즘에서의 임계값 34
    cv2.imshow('OTSU + Gaussian', binary_image)

def Otsu_Gaussian_no_image(calc): #오츄 가우시안

    calc = cv2.cvtColor(calc, cv2.COLOR_BGR2GRAY)
    gau_calc = cv2.GaussianBlur(calc, (5,5), 0)
    ret3, binary_image = cv2.threshold(gau_calc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    #print("Threshold in OTSU algorithm + Gaussian: ", ret3) # 가우시안 오츄 알고리즘에서의 임계값 34
    #cv2.imshow('OTSU + Gaussian', binary_image)
    return binary_image

# 이 부분에서 저희는 kernel 3X3의 eroding이 가장 좋은 결과라고 생각하는데 혹시 몰라서 여러 경우를 다 짜놨어요 (참고* 4-B 적절한 크기와 형태의 SE, 연산의 종류를 선택)
def Morphology(binary_image):

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3)) # kernel의 크기를 크게 할수록 이진 모폴로지의 효과는 뚜렷해져
    eroding = cv2.erode(binary_image, kernel, iterations=1)  # 침식
    dilating = cv2.dilate(binary_image, kernel, iterations=1)  # 팽창
    opening =cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=1) #열기 == 침식 연산 후 팽창
    closing = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel, iterations=1) #닫기 == 팽창 연산 후 침식
    close_after_open = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=1) #열기 후 닫기
    open_after_close = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=1) #닫기 후 열기

    # iteration = 이 함수를 몇번 반복하느냐 --> 결과적으로 한번만 실행하는것이 가장 선명한 화질을 보여
    cv2.imshow("eroding", eroding)
    cv2.imshow("dilating", dilating)
    cv2.imshow("opening", opening)
    cv2.imshow("closing", closing)
    cv2.imshow("close after open", close_after_open)
    cv2.imshow("open after close", open_after_close)



def main():
    #cv2.namedWindow('img')
    #cv2.setMouseCallback('img', mouse_handler)
    #cv2.waitKey(0)

    ori_img = cv2.imread('car.jpg')
    cv2.imshow('img', ori_img)
    wait()
    show_template()
    wait()
    Otsu_Gaussian(show_template_no_image())
    wait()
    Morphology(Otsu_Gaussian_no_image(show_template_no_image()))
    wait()



if __name__ == "__main__":
    main()

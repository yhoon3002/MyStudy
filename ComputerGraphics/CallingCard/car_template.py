import cv2, os
import numpy as np

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

    src = []
    src.append([125.0, 216.0])
    src.append([340.0, 266.0])
    src.append([340.0, 343.0])
    src.append([125.0, 283.0])

    print(src)
    # perspective transform
    src_np = np.array(src, dtype=np.float32)

    # width = max(np.linalg.norm(src_np[0] - src_np[1]), np.linalg.norm(src_np[2] - src_np[3]))
    # height = max(np.linalg.norm(src_np[0] - src_np[3]), np.linalg.norm(src_np[1] - src_np[2]))

    dst_np = np.array([
        [0, 0],
        [520, 0],
        [520, 110],
        [0, 110]
    ], dtype=np.float32)

    M = cv2.getPerspectiveTransform(src=src_np,
                                    dst=dst_np)  # 영상의 자동차 번호판을 template에 맞추는 투영행렬(M, 변환행렬) 을 구해준다. 직선 의 성질은 유지하되, 평행은 유지 x
    result = cv2.warpPerspective(ori_img, M=M, dsize=(520, 110))  # 투영행렬(변환행렬)에 의해 결과물 출력

    ##resized = cv2.resize(result, dsize=(600, 300), interpolation=cv2.INTER_AREA)
    cv2.imshow('result', result)


def wait():
    wait = cv2.waitKey(0)
    while (wait != 32):
        wait = cv2.waitKey(0)
        print(wait)


def main():
    #cv2.namedWindow('img')
    #cv2.setMouseCallback('img', mouse_handler)
    #cv2.waitKey(0)

    ori_img = cv2.imread('car.jpg')
    cv2.imshow('img', ori_img)
    wait()
    show_template()
    wait()


if __name__ == "__main__":
    main()

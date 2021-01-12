import numpy as np
import cv2
from scipy.spatial import distance as dist

def mouse_handler(event, x, y, flags, param): #마우스로 좌표 알아내기
    if event == cv2.EVENT_LBUTTONUP:
        clicked = [x, y]
        print(clicked)

def grab_cut(resized):
    mask_img = np.zeros(resized.shape[:2], np.uint8) #초기 마스크를 만든다.

    #grabcut에 사용할 임시 배열을 만든다.
    bgdModel = np.zeros((1,65), np.float64)
    fgdModel = np.zeros((1,65), np.float64)

    #rect = (130, 51, 885-130, 661-51) #mouse_handler로 알아낸 좌표 / card1일때
    rect = (150, 150, 800, 750) #card2 일 때
    cv2.grabCut(resized, mask_img, rect, bgdModel, fgdModel, 3, cv2.GC_INIT_WITH_RECT) #grabcut 실행
    mask_img = np.where((mask_img==2)|(mask_img==0), 0, 1).astype('uint8') #배경인 곳은 0, 그 외에는 1로 설정한 마스크를 만든다.
    img = resized*mask_img[:,:,np.newaxis] #이미지에 새로운 마스크를 곱해 배경을 제외한다.

    background = resized - img

    background[np.where((background >= [0, 0, 0]).all(axis = 2))] = [0, 0, 0]

    img_grabcut = background + img

    cv2.imshow('grabcut', img_grabcut)

    return img_grabcut


#에지 검출
def edge_detection(img_grabcut):

    gray = cv2.cvtColor(img_grabcut, cv2.COLOR_BGR2GRAY)
    _, gray = cv2.threshold(gray, 120, 255, cv2.THRESH_TOZERO)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    gray = cv2.edgePreservingFilter(gray, flags=1, sigma_s=45, sigma_r=0.2)

    edged = cv2.Canny(gray, 75, 200, True)

    #cv2.imshow("grayscale", gray)
    cv2.imshow("edged", edged)

    return edged


def contours(edged):
    (cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:5] #contourArea : contour가 그린 면적

    largest = max(cnts, key=cv2.contourArea)
    rect = cv2.minAreaRect(largest)
    r = cv2.boxPoints(rect)
    box = np.int0(r)

    size = len(box)

    # 2.A 원래 영상에 추출한 4 변을 각각 다른 색 선분으로 표시한다.
    cv2.line(resized, tuple(box[0]), tuple(box[size-1]), (255, 0, 0), 3)
    for j in range(size-1):
        color = list(np.random.random(size=3) * 255)
        cv2.line(resized, tuple(box[j]), tuple(box[j+1]), color, 3)

    #4개의 점 다른색으로 표시
    boxes = [tuple(i) for i in box]
    cv2.line(resized, boxes[0], boxes[0], (0, 0, 0), 15) #검
    cv2.line(resized, boxes[1], boxes[1], (255, 0, 0), 15) #파
    cv2.line(resized, boxes[2], boxes[2], (0, 255, 0), 15) #녹
    cv2.line(resized, boxes[3], boxes[3], (0, 0, 255), 15) #적


    cv2.imshow("With_Color_Image", resized)

    return boxes

def order_dots(pts):
    p = np.array(pts)
    xSorted = p[np.argsort(p[:, 0]), :]
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]

    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (a, c) = leftMost

    D = dist.cdist(a[np.newaxis], rightMost, "euclidean")[0]
    (b, d) = rightMost[np.argsort(D), :]
    return a, b, c, d

def transformation(resized, pts):
    p = np.array(pts)
    rect = np.zeros((4, 2), dtype="float32")

    s = p.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    a, b, c, d = rect

    w1 = abs(c[0] - d[0])
    w2 = abs(a[0] - b[0])
    h1 = abs(b[1] - c[1])
    h2 = abs(a[1] - d[1])

    w = max([w1, w2])
    h = max([h1, h2])

    dst = np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]])

    M = cv2.getPerspectiveTransform(rect, dst)
    result = cv2.warpPerspective(resized, M, (w, h))

    cv2.imshow("transformation", result)
    return result

def wait():
    wait = cv2.waitKey(0)
    while (wait != 32):
        wait = cv2.waitKey(0)
        print(wait)

class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __str__(self):
        return f"[{self.x} {self.y}]"

#y절편 구하기
def get_intercepts(ordered_dots):
    slopes = get_slopes(ordered_dots)

    a, b, c, d = [Point(*coord) for coord in ordered_dots]

    left = -a.x * slopes[0] + a.y
    right = -b.x * slopes[1] + b.y
    top = -a.x * slopes[2] + a.y
    bottom = -c.x * slopes[3] + c.y

    return left, right, top, bottom

#기울기 구하기
def get_slopes(ordered_dots):
    a, b, c, d = [Point(*coord) for coord in ordered_dots]

    left = (a.y - c.y) / (a.x - c.x)
    right = (b.y - d.y) / (b.x - d.x)
    top = (a.y - b.y) / (a.x - b.x)
    bottom = (c.y - d.y) / (c.x - d.x)

    return left, right, top, bottom

if __name__ == "__main__":
    filename = input('Enter Filename: ')
    img = cv2.imread(filename)

    if img.shape[0] * img.shape[1] > 1000 ** 2:
        resized = cv2.resize(img, dsize=(0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR)
    else:
        resized = cv2.resize(img, dsize=(0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    cut = grab_cut(resized)
    wait()
    edges = edge_detection(cut)
    wait()
    pts = contours(edges)
    wait()
    res = transformation(resized, pts)

    dots = order_dots(pts)
    a, b, c, d = dots

    # 각각의 점에 a,b,c,d 표시해놓았음.
    font = cv2.FONT_HERSHEY_COMPLEX
    fontScale = 1.2
    img_dots = cv2.line(resized, tuple(a), tuple(a), (0, 0, 255), 20)
    img_dots = cv2.putText(resized, 'a', tuple(a), font, fontScale, (0, 0, 255), 2)
    img_dots = cv2.line(resized, tuple(b), tuple(b), (0, 0, 255), 20)
    img_dots = cv2.putText(resized, 'b', tuple(b), font, fontScale, (0, 0, 255), 2)
    img_dots = cv2.line(resized, tuple(c), tuple(c), (0, 0, 255), 20)
    img_dots = cv2.putText(resized, 'c', tuple(c), font, fontScale, (0, 0, 255), 2)
    img_dots = cv2.line(resized, tuple(d), tuple(d), (0, 0, 255), 20)
    img_dots = cv2.putText(resized, 'd', tuple(d), font, fontScale, (0, 0, 255), 2)

    print("2-B	추출된 네변(선분), (즉, 좌, 우, 상, 하단 )의 기울기, y 절편, 양끝점의 좌표을 각각 출력할 것.\n")
    print("기울기 :")
    print(get_slopes(dots))
    print("\n")
    print("y절편 :")
    print(get_intercepts(dots))
    print("\n")
    print("양끝점 :")
    print(a, c)
    print(b, d)
    print(a, b)
    print(c, d)
    #cv2.imshow("resized", resized)

    print("\n")
    print("3-C.	네 꼭지점(좌상, 좌하, 우상, 우하 코너)의 좌표를 출력한다")
    print(a)
    print(b)
    print(c)
    print(d)

    wait()

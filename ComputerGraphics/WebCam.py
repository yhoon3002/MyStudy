import cv2
import numpy as np

#트랙바를 조정할 때 마다 실행되는 콜백 함수
def nothing(x):
    pass

def WebCapture():
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # hd 해상도의 절반
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 360) # hd 해상도의 절반

    roi = cv2.imread('total_skin.jpg')
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    roihist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(roihist, roihist, 0, 255, cv2.NORM_MINMAX)

    #트랙바 생성
    cv2.namedWindow("FaceDetected",) #얼굴이 출력될 윈도우 이름
    cv2.createTrackbar('threshold', "FaceDetected", 0, 255, nothing) #트랙바 생성하기
    cv2.setTrackbarPos('threshold', "FaceDetected", 127) #트랙바 초기값

    while True:
        ret, frame = capture.read()
        blur = cv2.GaussianBlur(frame, (5, 5), 0) # 가우시안 필터링 효과 (num, num) num이 더 클수록 blur효과 심해져
        cv2.imshow("blur_img", blur) # 1. blur처리된 얼굴 이미지 출력

        hsvB = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
        calc = cv2.calcBackProject([hsvB], [0, 1], roihist, [0, 180, 0, 256], 1)

        threshold = cv2.getTrackbarPos('threshold', "FaceDetected")
        ret, binary_image = cv2.threshold(calc, threshold, 255, cv2.THRESH_BINARY) #역투영된 신뢰도 맵을 이진화
        binary_image = cv2.merge((binary_image, binary_image, binary_image)) # 이진화된 이미지를 frame과 타입을 맞춰주기 위해 3차원으로 변형

        bit_img = cv2.bitwise_and(blur, binary_image) # 2. 기존 이미지 frame과 이진화된 이미지를 비트연산을 통해 얼굴 파트 출력

        #TrackBar 부분 : threshold값을 임계값으로 설정하기
        #cv2.getTrackbarPos('Trackbar','Face')

        #ret, binary_image = cv2.threshold(calc, threshold, 255, cv2.THRESH_BINARY)

        cv2.imshow("FaceDetected", bit_img)

        if cv2.waitKey(1) > 0: break # 아무키나 입력하면 loop 탈출

    capture.release()
    cv2.destroyAllWindows()


def main():
    print("1. blur image 출력")
    print("2. 비트연산을 통한 얼굴 이미지 출력")
    WebCapture()
if __name__ == "__main__":
    main()

    ####
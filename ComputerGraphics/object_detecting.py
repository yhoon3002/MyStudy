import cv2
import numpy as np

value = []
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)
white = (255, 255, 255)

isDragging = False
x0, y0, w, h = -1, -1, -1, -1

def onMouse(event, x, y, flags, param):
    global isDragging, x0, y0, img, value
    
    if event == cv2.EVENT_LBUTTONDOWN:
        isDragging = True
        x0 = x
        y0 = y

    elif event == cv2.EVENT_MOUSEMOVE:
        if isDragging:
            img_draw = img.copy()
            cv2.rectangle(img_draw,(x0,y0),(x,y),blue,2)
            cv2.imshow('img',img_draw)

    elif event == cv2.EVENT_LBUTTONUP:
        if isDragging:
            isDragging = False
            w = x - x0
            h = y - y0

            if w > 0 and h > 0:
                img_draw = img.copy()
                cv2.rectangle(img_draw,(x0,y0),(x,y),red,2)
                cv2.imshow('img', img_draw)
                roi = img[y0:y0+h, x0:x0+w]
                val = input("모델 객체의 이름을 지정해주세요: ")
                value.append(val)
                cv2.imwrite(val+'.png',roi)
                cv2.imshow(val,roi)
                cv2.moveWindow(val,0,0)
                print("\n>> 객체는 총 3가지 지정해주세요.")
                print("지정 작업을 계속해주시거나,")
                print("객체 지정이 끝났다면 img 창에서 스페이스바를 눌러주세요.\n")
                
            else:
                cv2.imshow('img',img)
                print('드래그 방향은 왼쪽위->오른쪽아래\n')


# queryImage는 jpg, trainImage는 png형식이라 가정
# 추적할 객체는 3가지로 한정, 입력순
def vidTrack():
    global frame, value
    MIN_MATCH_COUNT = 10
    skip = 0
    count1, count2, count3 = 0, 0, 0
    
    cap = cv2.VideoCapture('./video.mp4')
    
    while(cap.isOpened()):
        ret, frame = cap.read()
        frame = cv2.resize(frame,dsize=(0,0),fx=0.8,fy=0.8,interpolation=cv2.INTER_LINEAR)
        
        if ret:
            cv2.imwrite('frame{:d}.jpg'.format(skip), frame)
            skip += 6   # 최적화-프레임 건너뛰기
            cap.set(1, skip)
            cv2.imshow('video',frame)
            
            imgQuery = frame
            img1 = cv2.imread('./' +value[0]+'.png')
            img2 = cv2.imread('./' +value[1]+'.png')
            img3 = cv2.imread('./' +value[2]+'.png')

            sift = cv2.xfeatures2d.SIFT_create()

            kpQ, desQ = sift.detectAndCompute(imgQuery,None)
            kp1, des1 = sift.detectAndCompute(img1,None)
            kp2, des2 = sift.detectAndCompute(img2,None)
            kp3, des3 = sift.detectAndCompute(img3,None)

            FLANN_INDEX_KDTREE = 0
            index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
            search_params = dict(checks = 50)
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches1 = flann.knnMatch(desQ,des1,k=2)
            matches2 = flann.knnMatch(desQ,des2,k=2)
            matches3 = flann.knnMatch(desQ,des3,k=2)

            good1 = []
            good2 = []
            good3 = []
            for m,n in matches1:
                if m.distance < 0.4*n.distance:
                    good1.append(m)
            for m,n in matches2:
                if m.distance < 0.4*n.distance:
                    good2.append(m)
            for m,n in matches3:
                if m.distance < 0.4*n.distance:
                    good3.append(m)

            # 첫번째 객체
            if len(good1)>MIN_MATCH_COUNT:

                src_pts = np.float32([ kpQ[m.queryIdx].pt for m in good1 ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp1[m.trainIdx].pt for m in good1 ]).reshape(-1,1,2)
                
                src_pts_xy = np.float32([ kpQ[m.queryIdx].pt for m in good1 ]).reshape(-1,1,1)
                
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask1 = mask.ravel().tolist()
                
                h,w,c = img1.shape
                x = src_pts_xy[10][0]
                y = src_pts_xy[11][0]
                pts = np.float32([ [x-w/2,y-h/2],[x-w/2,y+h/2],[x+w/2,y+h/2],[x+w/2,y-h/2] ]).reshape(-1,1,2)
                name_loc = ((2*x+w)/2, (2*y+h)/2)
                
                frame = cv2.polylines(frame,[np.int32(pts)],True,green,2)
                cv2.putText(frame,value[0],name_loc,cv2.FONT_HERSHEY_SIMPLEX,1,green,2)
                cv2.imshow('video',frame)
                
            elif len(good1)<=MIN_MATCH_COUNT:
                #print(value[0]+"의 매칭점이 충분하지 않습니다. - %d/%d" % (len(good1),MIN_MATCH_COUNT))
                matchesMask1 = None

            # 두번째 객체
            if len(good2)>MIN_MATCH_COUNT:

                src_pts = np.float32([ kpQ[m.queryIdx].pt for m in good2 ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good2 ]).reshape(-1,1,2)
                
                src_pts_xy = np.float32([ kpQ[m.queryIdx].pt for m in good2 ]).reshape(-1,1,1)
                
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask2 = mask.ravel().tolist()
                
                h,w,c = img2.shape
                x = src_pts_xy[10][0]
                y = src_pts_xy[11][0]
                pts = np.float32([ [x-w/2,y-h/2],[x-w/2,y+h/2],[x+w/2,y+h/2],[x+w/2,y-h/2] ]).reshape(-1,1,2)
                name_loc = ((2*x+w)/2, (2*y+h)/2)
                
                frame = cv2.polylines(frame,[np.int32(pts)],True,red,2)
                cv2.putText(frame,value[1],name_loc,cv2.FONT_HERSHEY_SIMPLEX,1,red,2)
                cv2.imshow('video',frame)
                
            elif len(good2)<=MIN_MATCH_COUNT:
                #print(value[1]+"의 매칭점이 충분하지 않습니다. - %d/%d" % (len(good2),MIN_MATCH_COUNT))
                matchesMask2 = None

            # 세번째 객체
            if len(good3)>MIN_MATCH_COUNT:

                src_pts = np.float32([ kpQ[m.queryIdx].pt for m in good3 ]).reshape(-1,1,2)
                dst_pts = np.float32([ kp3[m.trainIdx].pt for m in good3 ]).reshape(-1,1,2)
                
                src_pts_xy = np.float32([ kpQ[m.queryIdx].pt for m in good3 ]).reshape(-1,1,1)
                
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask3 = mask.ravel().tolist()
                
                h,w,c = img3.shape
                x = src_pts_xy[10][0]+10
                y = src_pts_xy[11][0]+10
                pts = np.float32([ [x-w/2,y-h/2],[x-w/2,y+h/2],[x+w/2,y+h/2],[x+w/2,y-h/2] ]).reshape(-1,1,2)
                name_loc = ((2*x+w)/2, (2*y+h)/2)
                
                frame = cv2.polylines(frame,[np.int32(pts)],True,blue,2)
                cv2.putText(frame,value[2],name_loc,cv2.FONT_HERSHEY_SIMPLEX,1,blue,2)
                cv2.imshow('video',frame)
                
            elif len(good3)<=MIN_MATCH_COUNT:
                #print(value[2]+"의 매칭점이 충분하지 않습니다. - %d/%d" % (len(good3),MIN_MATCH_COUNT))
                matchesMask3 = None

            
            key = cv2.waitKey(1)
            if key == 32:
                break
            elif key == ord('p'):
                cv2.waitKey(-1)

        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def main():
    global img
    src = cv2.imread('./image.jpg')
    img = cv2.resize(src,dsize=(0,0),fx=0.4,fy=0.4,interpolation=cv2.INTER_LINEAR)
    cv2.imshow('img',img)
    cv2.moveWindow('img',300,0)
    while True:
        cv2.setMouseCallback('img',onMouse)
        if cv2.waitKey(1) == 32:
            cv2.destroyAllWindows()
            break
    # 객체지정 작업이 끝났다면 영상을 불러와서 트래킹
    vidTrack()


if __name__ == "__main__":
    main()
    



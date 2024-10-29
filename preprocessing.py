# preprocessing.py | 담당자: 신우협
# 이미지 하나를 입력받아 openCV를 이용해 직사각형 모서리를 감지, 답란을 추출하여 배열로 반환하는 함수.

import cv2
import numpy as np

# 입력: image
# 출력: image (cv2)의 array
 
def preprocess(image):
    # 이미지 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 이미지 블러 처리
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 엣지 감지
    edged = cv2.Canny(blurred, 50, 150)

    # 외곽선 찾기
    contours, _ = cv2.findContours(edged, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 잘라낸 직사각형 이미지를 배열
    images = []

    # 모든 외곽선 중에서 직사각형을 인식 후 이미지 저장
    for contour in contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 사각형은 4개의 꼭짓점을 가짐
        if len(approx) == 4:
            # 사각형의 외곽 영역을 계산
            x, y, w, h = cv2.boundingRect(approx)
            # 사각형 영역을 잘라서 저장
            cropped_image = image[y:y+h, x:x+w]
            images.append(cropped_image)

    return images

# 테스트를 위한 가짜 데이터.
if __name__ == "__main__":
  image = cv2.imread('source/dummy/example.jpg')
  print(preprocess(image))

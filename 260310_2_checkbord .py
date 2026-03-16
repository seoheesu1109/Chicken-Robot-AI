import numpy as np
import cv2
import glob

# 체커보드 설정 (교차점 개수: 가로 9, 세로 6)
CHECKERBOARD = (9, 6)
# 실제 사각형 한 변의 길이 (mm 단위, 모르면 1.0으로 두어도 무방)
SQUARE_SIZE = 25.0 

# 3D 실제 세계 좌표 준비 (0,0,0), (1,0,0), ..., (8,5,0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE

objpoints = [] # 3D 실제 좌표들
imgpoints = [] # 2D 이미지 좌표들

# 이미지 파일들 불러오기
images = glob.glob('calibration_images/*.jpg')

print(f"{len(images)}개의 이미지를 분석합니다...")

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 체커보드 코너 찾기
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    if ret:
        objpoints.append(objp)
        # 코너 정밀화 (Subpixel accuracy)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)
        print(f"성공: {fname}")
    else:
        print(f"실패: {fname} (패턴을 찾을 수 없음)")

# 캘리브레이션 수행
if len(objpoints) > 0:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    
    # 결과 저장 (나중에 메인 코드에서 불러와서 사용)
    np.savez('calibration_data.npz', mtx=mtx, dist=dist)
    
    print("\n✅ 캘리브레이션 완료!")
    print("Camera Matrix:\n", mtx)
    print("Distortion Coefficients:\n", dist)
    print("\n'calibration_data.npz' 파일이 생성되었습니다.")
else:
    print("❌ 유효한 사진이 없습니다. 패턴이 잘 보이게 다시 찍어주세요.")
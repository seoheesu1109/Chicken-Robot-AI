import cv2

def test_cameras():
    # 0, 1, 2번 인덱스를 순서대로 시도
    for i in [0, 1, 2]:
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"✅ 카메라 {i}번 감지 성공! 창을 확인하세요.")
            while True:
                ret, frame = cap.read()
                if not ret: break
                
                cv2.imshow(f"Camera Test - Index {i}", frame)
                # 'q' 누르면 다음 인덱스로 넘어가거나 종료
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            cap.release()
            cv2.destroyAllWindows()
        else:
            print(f"❌ 카메라 {i}번 없음")

if __name__ == "__main__":
    test_cameras()
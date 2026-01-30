#RX,RY,RZ 캘리하는거



def camera_worker(self):
        while not self.stop_event.is_set():
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame: continue

            img = np.asanyarray(color_frame.get_data())
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)

            if ids is not None:
                rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, 100, self.camera_matrix, self.dist_coeffs)
                
                # 1. 마커 위치 분류
                marker_map = {ids[i][0]: i for i in range(len(ids))}
                
                # [ID 1]이 추적 대상(따라가야 할 마커)인 경우
                if 1 in marker_map:
                    idx = marker_map[1]
                    # 오늘 맞춘 캘리브레이션 공식으로 ID 1의 로봇 좌표계 상 위치 계산
                    pos, rot = self.get_robot_pose_math(tvecs[idx][0], rvecs[idx][0])
                    
                    self.target_pose = pos + rot
                    
                    # 시각화 (추적 대상은 파란색 축으로 표시)
                    cv2.aruco.drawDetectedMarkers(img, corners, ids)
                    cv2.drawFrameAxes(img, self.camera_matrix, self.dist_coeffs, rvecs[idx][0], tvecs[idx][0], 80)
                    cv2.putText(img, "TARGET (ID:1)", (int(corners[idx][0][0][0]), int(corners[idx][0][0][1])), 
                                1, 1.5, (255, 0, 0), 2)

                    # 자동 추적 활성화 시 명령 전송
                    if self.is_tracking and self.conn:
                        try:
                            # 로봇이 따라갈 좌표 전송 (Z값에 적절한 안전 거리 추가 가능)
                            msg = f"MOVE,{pos[0]:.1f},{pos[1]:.1f},{pos[2]:.1f},{rot[0]:.1f},{rot[1]:.1f},{rot[2]:.1f}\n"
                            self.conn.sendall(msg.encode())
                        except: pass
                
                # [ID 0]은 화면에 표시만 함 (로봇 손 확인용)
                if 0 in marker_map:
                    idx_hand = marker_map[0]
                    cv2.putText(img, "ROBOT HAND (ID:0)", (int(corners[idx_hand][0][0][0]), int(corners[idx_hand][0][0][1])), 
                                1, 1.2, (0, 255, 0), 2)

            cv2.imshow("ArUco Tracking View", img)
            if cv2.waitKey(1) == ord('q'): break
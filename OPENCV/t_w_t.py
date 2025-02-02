import cv2
import numpy as np
import random
import pandas

"""LESSON ONE ---> INTRODUCTION"""
# img1 = cv2.imread('Assets/cat1.jpg', 1)
# img1_rescale = cv2.resize(img1, (0, 0), fx=.3, fy=.3)
# img1_rotated = cv2.rotate(img1_rescale, cv2.ROTATE_90_CLOCKWISE)
# cut_out = img1_rescale[0:360, 0:540]
# img1[0:360, 0:540] = cut_out
# print(img1.shape)
# print(img1_rescale.shape)
# # cv2.imwrite('processed_IMG.jpg', img1_rotated)
# cv2.imshow('cat1', img1_rotated)
# cv2.waitKey(0)
# cv2.destroyWindow('cat1')

"""LESSON TWO ---> PIXELS AS NUMPY ARRAY"""
# np_array = np.zeros((1000, 1000, 3), dtype='int32')
# for r in range(np_array.shape[0]):
#     for c in range(np_array.shape[1]):
#         np_array[r][c] = [random.randrange(255), random.randrange(255), random.randrange(255)]
#
# cv2.imshow('random_px', np_array.astype('uint8'))
# cv2.waitKey(0)
# cv2.destroyWindow('random_px')

"""LESSON THREE ---> VIDEO"""
# video = cv2.VideoCapture('Assets/catvid1.mp4')
#
# if not video.isOpened():
#     print("video was not opened !!!")
#     exit()
#
# fps = video.get(cv2.CAP_PROP_FPS)
# if fps == 0:
#     fps = 30
#
# while True:
#     returned, frame = video.read()
#
#     if not returned:
#         print("Reached the end of the video or failed to read the frame.")
#         break
#
#     big_frame = np.zeros(frame.shape, dtype='uint8')
#     frame_resized = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
#     big_frame[0:(big_frame.shape[0]) // 2, 0:(big_frame.shape[1]) // 2] = frame_resized
#     big_frame[0:(big_frame.shape[0]) // 2, (big_frame.shape[1]) // 2:] = frame_resized
#     big_frame[(big_frame.shape[0]) // 2:, 0:(big_frame.shape[1]) // 2] = frame_resized
#     big_frame[(big_frame.shape[0]) // 2:, (big_frame.shape[1]) // 2:] = frame_resized
#
#     cv2.imshow("video", big_frame)
#     if cv2.waitKey(int(1000/fps)) & 0xFF == ord('q'):
#         break
# video.release()
# cv2.destroyAllWindows()

"""LESSON FOUR ---> HSV AND MUSKING"""
video = cv2.VideoCapture('Assets/catvid1.mp4')
if not video.isOpened():
    print("video failed to load!")
    quit()
#
while True:
    ret, frame = video.read()
    if not ret:
        print("error reading the video!")

    hsv_image = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)

    # Define the color range for red (lower range)
    lower_red1 = np.array([0, 50, 50])
    upper_red1 = np.array([10, 255, 255])

    # Define the color range for red (upper range)
    lower_red2 = np.array([170, 50, 50])
    upper_red2 = np.array([180, 255, 255])

    # Create masks for the two red ranges
    mask1 = cv2.inRange(hsv_image, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_image, lower_red2, upper_red2)

    # Combine the two masks
    mask = cv2.bitwise_or(mask1, mask2)

    # Apply the mask to the original frame
    result = cv2.bitwise_and(frame, frame, mask=mask)

    # cv2.imshow('result', result)
    cv2.imshow('cat', hsv_image)

    if cv2.waitKey(1) == ord('q'):
        break

video.release()
cv2.destroyAllWindows()


"""LESSON FIVE ---> CORNERS"""
# board_image = cv2.imread('Assets/chessboard.png')
# board_image = cv2.resize(board_image, (0, 0), fx=0.5, fy=0.5)
# bnw_image = cv2.cvtColor(board_image, cv2.COLOR_BGR2GRAY)
#
# corners = cv2.goodFeaturesToTrack(bnw_image, 100, 0.01, 10)
#
# for corner in np.int64(corners):
#     x, y = corner.ravel()
#     cv2.circle(board_image, (x, y), 8, (0, 0, 0), -1)
#
#
# cv2.imshow("board", board_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

"""LESSON SIX ---> OBJECT DETECTION"""
# soccer_img = cv2.resize(cv2.imread("Assets/soccer_practice.jpg"), (0, 0), fx=0.4, fy=0.4)
# ball_img = cv2.resize(cv2.imread("Assets/ball.PNG"), (0, 0), fx=0.4, fy=0.4)
#
# bnw_soccer = cv2.cvtColor(soccer_img, cv2.COLOR_BGR2GRAY)
# bnw_ball = cv2.cvtColor(ball_img, cv2.COLOR_BGR2GRAY)
#
# h, w = bnw_ball.shape
#
# methods = [cv2.TM_CCOEFF, cv2.TM_CCOEFF_NORMED, cv2.TM_CCORR,
#            cv2.TM_CCORR_NORMED, cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]
#
# for method in methods:
#     img_copy = bnw_soccer.copy()
#     final_copy = soccer_img.copy()
#
#     result = cv2.matchTemplate(img_copy, bnw_ball, method)
#     min_value, max_value, min_location, max_location = cv2.minMaxLoc(result)
#
#     if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
#         location = min_location
#     else:
#         location = max_location
#
#     bottom_right = (location[0] + w, location[1] + h)
#     final = cv2.rectangle(final_copy, location, bottom_right, (0, 255, 0), 3)
#
#     cv2.imshow("final", final)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

"""LESSON SEVEN ---> FEATURE DETECTION WITH IN BUILT MODELS"""
# cap = cv2.VideoCapture("Assets/man_face1.mp4")
#
# if cap.isOpened():
#     fps = cap.get(cv2.CAP_PROP_FPS)
# else:
#     print("video no being read!!!")
#     exit()
#
# face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#
# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("frame not being loaded!!!")
#
#     frame = cv2.resize(frame, (0, 0), fx=0.3, fy=0.3)
#     bnw_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     detected_faces = face_cascade.detectMultiScale(bnw_frame, 1.4, 5)
#     print(f"faces in this frame: {len(detected_faces)}")
#
#     for (x, y, w, h) in detected_faces:
#         frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
#
#         face = frame[y:y+h, x:x+w]
#         bnw_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#
#         eyes_detected = eye_cascade.detectMultiScale(bnw_face, 1.4, 5)
#         print(f"eyes in this frame: {len(eyes_detected)}")
#         for (e_x, e_y, e_w, e_h) in eyes_detected:
#             face = cv2.rectangle(face, (e_x, e_y), (e_x+e_w, e_y+e_h), (0, 255, 0), 3)
#
#     cv2.imshow("man's face", frame)
#     if cv2.waitKey(int(1000//fps)) == ord("q"):
#         break
#
# cap.release()
# cv2.destroyAllWindows()

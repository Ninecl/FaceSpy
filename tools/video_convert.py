import cv2

video = cv2.VideoCapture("ours.mp4")

fps = 25
size = (1280, 720)

writer = cv2.VideoWriter("ours.avi", cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 25, size)

ret, frame = video.read()
while(ret):
    cv2.imshow("video", frame)
    cv2.waitKey(1)
    writer.write(frame)
    ret, frame = video.read()

video.release()
cv2.destoryAllWindows()

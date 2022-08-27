import cv2

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

class Video(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()

        faces = faceDetect.detectMultiScale(frame, 1.3, 5)

        for x, y, w, h in faces:
            x1, x2 = x, x + w
            y1, y2 = y, y + h
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 1)

            cv2.line(frame, (x1, y1), (x1, y1 + 30), (255, 0, 255), 6)
            cv2.line(frame, (x1, y1), (x1 + 30, y1), (255, 0, 255), 6)

            cv2.line(frame, (x1, y2), (x1, y2 - 30), (255, 0, 255), 6)
            cv2.line(frame, (x1, y2), (x1 + 30, y2), (255, 0, 255), 6)

            cv2.line(frame, (x2, y1), (x2 - 30, y1), (255, 0, 255), 6)
            cv2.line(frame, (x2, y1), (x2, y1 + 30), (255, 0, 255), 6)

            cv2.line(frame, (x2, y2), (x2, y2 - 30), (255, 0, 255), 6)
            cv2.line(frame, (x2, y2), (x2 - 30, y2), (255, 0, 255), 6)

        ret,jpg = cv2.imencode(".jpg", frame)
        return jpg.tobytes()

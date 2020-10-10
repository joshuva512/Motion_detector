import cv2, pandas
from datetime import datetime

dataframe = pandas.DataFrame(columns=["start", "end"])
first_frame = []
status = [None, None]
times = []
video = cv2.VideoCapture(0)
while True:
    frame = video.read()[1]
    sc = 0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)
    if first_frame == []:
        first_frame = gray
        continue
    delta_frame = cv2.absdiff(first_frame, gray)
    thresh_frame = cv2.threshold(delta_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)
    (con, _) = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contor in con:
        if cv2.contourArea(contor) < 1000:
            continue
        sc = 1
        (x, y, w, h) = cv2.boundingRect(contor)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    status.append(sc)
    status = status[-2:]
    if status[-1] == 1 and status[-2] == 0:
        times.append(datetime.now())
    if status[-1] == 0 and status[-2] == 1:
        times.append(datetime.now())

    cv2.imshow("color_frame", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        if sc == 1:
            times.append(datetime.now())
        break

for i in range(0, len(times), 2):
    dataframe = dataframe.append({"start": times[i], "end": times[i + 1]}, ignore_index=True)
dataframe.to_csv("time details.csv")
video.release()
cv2.destroyWindow(None)

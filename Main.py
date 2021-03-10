import numpy as np
import cv2
import time

cont = 0
min_conf = 0.4

nn = cv2.dnn.readNetFromCaffe('SSD_MobileNet_prototxt.txt', 'SSD_MobileNet.caffemodel')

video = cv2.VideoCapture("Video.mp4")    # Entrada de Video

# fourcc = cv2.VideoWriter_fourcc(*'XVID')     # Formato do video de saida
# saida = cv2.VideoWriter('resultado.avi', fourcc, 30.0, (int(video.get(3)), (int(video.get(4)))))
saidatxt = open('saida.txt', 'w')

labels = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
             "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
colors = np.random.uniform(0, 255, size=(len(labels), 3))
tempo = [0, 0]

while True:

    ret, frame = video.read()     # Leitura dos frames
    temp = time.time()
    if ret is True:

        # cv2.rectangle(frame, (500, 100), (1085, 500), (0, 0, 0), 2)
        roi = frame[100:500, 500:1085]

        (h, w) = roi.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(roi, (416, 416)), 0.013843, (416, 416), (110, 110, 110), swapRB=True)

        nn.setInput(blob)
        detections = nn.forward()

        for i in np.arange(0, detections.shape[2]):

            confidence = detections[0, 0, i, 2]

            if confidence > min_conf:

                idx = int(detections[0, 0, i, 1])
                if idx == 10:
                    cont = cont + 1
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    label = "{} {}: {:.2f}%".format(labels[idx], i, confidence * 100)
                    cv2.rectangle(roi, (startX, startY), (endX, endY), colors[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(roi, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)
                    if (i == 0) and (tempo[i] == 0):
                        tempo[i] = time.time()
                    elif (i == 1) and (tempo[i] == 0):
                        tempo[i] = time.time()

                print(" ", cont)

        # saida.write(frame)
        cv2.imshow('frame', frame)
        # cv2.imshow('roi', roi)
        cont = 0

        if cv2.waitKey(1) & 0xFF == ord('q'):
            tempo[0] = time.time() - tempo[0]
            if tempo[1] > 0:
                tempo[1] = time.time() - tempo[1]
            print("Vaca 1: ", tempo[0],"s", "Vaca 2: ", tempo[1],"s", file=saidatxt)
            break
    else:
        tempo[0] = time.time() - tempo[0]
        if tempo[1] > 0:
            tempo[1] = time.time() - tempo[1]
        print("Vaca 1: ", tempo[0],"s", "Vaca 2: ", tempo[1],"s", file=saidatxt)
        break

saidatxt.close()
video.release()
# saida.release()
cv2.destroyAllWindows()

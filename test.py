import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob
import time

def get_output_layers(net):
    layer_names = net.getLayerNames()

    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    # label_conf = str(classes[class_id]) + ': ' + str(int(confidence*100)) + '%'
    label_conf = "{} [{:.2f}]".format(classes[class_id], float(confidence))
    # color = COLORS[class_id]
    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color=(0,0,255), thickness=2)
    # cv2.putText(img, label_conf , (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    cv2.putText(img, label_conf , (x - 10, y - 10), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,0,0), 1)



def drawBox(image, points):
    height, width = image.shape[:2]
    for (label, xi,yi, wi, hi) in points:
        center_x = int(xi * width)
        center_y = int(yi * height)
        w = int(wi * width)
        h = int(hi * height)
        # Rectangle coordinates
        x = int(center_x - w / 2)
        y = int(center_y - h / 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), color=COLORS,thickness=10)
    return
def savePredict(name, text):
    textName = name + '.txt'
    with open(textName, 'w+') as groundTruth:
        groundTruth.write(text)
        groundTruth.close()



conf_threshold = 0.2
nms_threshold = 0.4


with open("./model/yolov4_hole.names", 'r') as f: # Edit CLASS file
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(1, 3))

net = cv2.dnn.readNet("./model/yolov4_hole_best.weights", "./model/yolov4_hole.cfg") # Edit WEIGHT and CONFIC file


# ########### NHAN DIEN O GA BANG ANH THUONG #############

def detectImage(img_path):
    start = time.time()

    image = cv2.imread(img_path)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)

    net.setInput(blob)
    outs = net.forward(get_output_layers(net))
    #print(outs)
    class_ids = []
    confidences = []
    boxes = []


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)

            confidence = scores[class_id]
            if confidence > 0.5:
                print(confidence)
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                #print(w,h,x,y)
                class_ids.append(class_id)
                """if confidence < 0.6:
                    class_ids.append(2)""" #change
                confidences.append(float(confidence))

                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    # Result = ""
    for i in indices:
        i = i[0]
        box = boxes[i]
        x, y, w, h = boxes[i]

        draw_prediction(image, class_ids[i],confidences[i], round(x), round(y), round(x + w), round(y + h))
        

    print("Ran in {} seconds".format(time.time() - start))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()

#################################################################################

################ NHAN DIEN O GA BANG VIDEO #################################
def detectVideo(vid_path):
    
    cap = cv2.VideoCapture(vid_path)
    scale = 0.00392
    # scale = 1
    while True:
        time.sleep(1)
        # time = time()
        _,image = cap.read()
        
        Width = image.shape[0]
        Height = image.shape[1]

        blob = cv2.dnn.blobFromImage(image, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        outs = net.forward(get_output_layers(net))

        #print(outs)
        class_ids = []
        confidences = []
        boxes = []

        # start = time.time()
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)

                confidence = scores[class_id]
                if confidence > 0.5:
                    print(confidence)
                    center_x = int(detection[0] * Width)
                    center_y = int(detection[1] * Height)
                    w = int(detection[2] * Width)
                    h = int(detection[3] * Height)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    """if confidence < 0.6:
                        class_ids.append(2)""" #change
                    confidences.append(float(confidence))

                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        # Result = ""
        for i in indices:
            i = i[0]
            box = boxes[i]
            x, y, w, h = boxes[i]

            draw_prediction(image, class_ids[i],confidences[i], round(x), round(y), round(x + w), round(y + h))

        cv2.imshow('image', image)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()



detectImage('./images/a.jpg')
# detectVideo('./video/video_tn.mp4')
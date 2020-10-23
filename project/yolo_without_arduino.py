import cv2
import numpy as np
import firebase_admin
from firebase_admin import credentials, initialize_app, storage
from uuid import uuid4
import os

# Init firebase with your credentials
cred = credentials.Certificate("key/push-app-no-mask-firebase-adminsdk-g0sa7-ca7092f84e.json")
firebase_admin.initialize_app(cred, {'storageBucket': 'push-app-no-mask.appspot.com'})

#Load YOLO
net = cv2.dnn.readNet("yolov3_15000.weights","yolov3.cfg")
classes = []
with open("coco.names","r") as f:
    classes = [line.strip() for line in f.readlines()]

#Loading Webcam
outputFile = "yolo_out_py.avi"
cap = cv2.VideoCapture(0)

vid_writer = cv2.VideoWriter(outputFile,cv2.VideoWriter_fourcc('M','J','P','G'), 30, (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

global index
index=0
            
def push_firebase(outputFile):
    # Put your local file path
    bucket = storage.bucket()
    blob = bucket.blob(outputFile)

    # Create new token
    new_token = uuid4()

    # Create new dictionary with the metadata
    metadata = {"firebaseStorageDownloadTokens": new_token}

    # Set metadata to blob
    blob.metadata = metadata

    blob.upload_from_filename(outputFile, content_type='image/jpeg')

    # Opt : if you want to make public access from the URL
    # blob.make_public()

    print("your file url", blob.public_url)

    # result = os.popen('php android_push.php no-mask').read().strip()
    # print(result)


while True:

    # get frame from the video
    hasFrame, frame = cap.read()
    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv2.waitKey(3000)
        break

    cv2.imshow("VideoFrame",frame)
    if cv2.waitKey(1)>0: break

    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(frame, 1/255, (416,416), (0,0,0),1,crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Get the names of the output layers
    layer_names = net.getLayerNames()
    outputlayers = [layer_names[i[0]-1] for i in net.getUnconnectedOutLayers()]

    # Runs the forward pass to get output of the output layers
    outs = net.forward(outputlayers)

    # Remove the bounding boxes with low confidence
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []

    no_mask_frame = False

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box`s class label as the class with the highest score.

    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > 0.6:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.6, 0.4)
    for i in indices:
        i = i[0]
        left, top, width, height = boxes[i]
        classId = classIds[i]
        conf = confidences[i]
        right = left+width
        bottom = top+height
        cv2.rectangle(frame, (left,top), (right,bottom), (0,0,255))

        label = '%.2f' % conf

        # Get the label for the class name and its confidence
        if classes:
            if classId >= len(classes):
                print("Not valid.")
            else :
                label = '%s:%s' % (classes[classId], label)

        # Display the label at the top of the bounding box
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        top = max(top, labelSize[1])
        cv2.putText(frame, label, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255))
        
        if classes:
            if classId >= len(classes):
                print("Not valid.")
            else :
                if classId == 0:
                    print("mask")
                if classId == 1:
                    print("no-mask")
                    no_mask_frame=True

    if no_mask_frame==True :
        outputFile = "img/yoloPython" + str(index)+".jpg"
        index+=1
        print(outputFile)
        cv2.imwrite(outputFile,frame.astype(np.uint8))
        push_firebase(outputFile)


    # Put efficiency information. The function get PerfProfile returns the
    # overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
    cv2.putText(frame,label,(0,15),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))

    # Write the frame with the detection boxes
    vid_writer.write(frame.astype(np.uint8))

    # outputFile = "img/yolo_python_" + str(index)+".jpg"
    # index+=1
    # print(outputFile)
    # cv2.imwrite(outputFile,frame.astype(np.uint8))

cap.release()
cv2.destroyAllWindows()

################################################################################################
# END YOLO #####################################################################################
# START FIREBASE ###############################################################################
################################################################################################

# import firebase_admin
# from firebase_admin import credentials, initialize_app, storage
# from uuid import uuid4

# # Init firebase with your credentials
# cred = credentials.Certificate("key/push-app-no-mask-firebase-adminsdk-g0sa7-ca7092f84e.json")
# firebase_admin.initialize_app(cred, {'storageBucket': 'push-app-no-mask.appspot.com'})

# # Put your local file path
# fileName = "img/eunha2.jpeg"
# bucket = storage.bucket()
# blob = bucket.blob(fileName)

# # Create new token
# new_token = uuid4()

# # Create new dictionary with the metadata
# metadata = {"firebaseStorageDownloadTokens": new_token}

# # Set metadata to blob
# blob.metadata = metadata

# blob.upload_from_filename(fileName, content_type='image/jpeg')

# # Opt : if you want to make public access from the URL
# blob.make_public()

# print("yout file url", blob.public_url)

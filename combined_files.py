import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
import math

# original edges
# KEYPOINT_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (0, 5), (0, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

KEYPOINT_EDGES = [(0, 1), (0, 2), (1, 3), (2, 4), (5, 7), (7, 9), (6, 8), (8, 10), (5, 6), (5, 11), (6, 12), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16)]

PARTS = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle"
}

EXERCISES = {
    # given in (body part, body part, body part, ideal angle)
    # margin of error for angle will be constant for now (10 degrees?)
    # order: shoulder to knee, hip to ankle, shoulder to elbow to wrist
    "pushup": [(6, 12, 14, 180), (12, 14, 16, 180), (6, 8, 10, 180)]
}

def angle(p1, p2, p3):
    a = math.dist(p1, p2)
    b = math.dist(p2, p3)
    c = math.dist(p1, p3)

    c_angle = math.acos((a ** 2 + b ** 2 - c ** 2) / (2 * a * b))
    return math.degrees(c_angle)


model_path = "movenet_lightning_fp16.tflite"
interpreter = tf.lite.Interpreter(model_path)
interpreter.allocate_tensors()

cap = cv2.VideoCapture(0) # 0 can be switched out for mp4
while cap.isOpened():
    ret, frame = cap.read()
    img = frame.copy()

    input_image = tf.expand_dims(img, axis=0)
    input_image = tf.image.resize_with_pad(input_image, 192, 192)

    input_image = tf.cast(input_image, dtype=tf.uint8)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
    interpreter.invoke()
    keypoints = interpreter.get_tensor(output_details[0]['index'])

    width = 640
    height = 640

    input_image = tf.expand_dims(img, axis=0)
    input_image = tf.image.resize_with_pad(input_image, width, height)
    input_image = tf.cast(input_image, dtype=tf.uint8)

    image_np = np.squeeze(input_image.numpy(), axis=0)
    image_np = cv2.resize(image_np, (width, height))
    #image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    body_part = 0
    parts = {}

    for keypoint in keypoints[0][0]:
        x = int(keypoint[1] * width)
        y = int(keypoint[0] * height)

        body_part += 1
        parts[body_part] = (x, y)
        # if body_part == 6 or body_part == 12 or body_part == 14:
            
        # else:
        #     cv2.circle(image_np, (x, y), 4, (0, 0, 255), -1)


    shoulder_knee = angle(parts[6], parts[12], parts[14])
    hip_ankle = angle(parts[12], parts[14], parts[16])
    shoulder_wrist = angle(parts[6], parts[8], parts[10])

    if 140 < shoulder_wrist < 180:
        cv2.line(image_np, (parts[6][0], parts[6][1]), (parts[8][0], parts[8][1]), (0, 255, 0), 2)
        cv2.line(image_np, (parts[8][0], parts[8][1]), (parts[10][0], parts[10][1]), (0, 255, 0), 2)
    else:
        cv2.line(image_np, (parts[6][0], parts[6][1]), (parts[8][0], parts[8][1]), (0, 0, 255), 2)
        cv2.line(image_np, (parts[8][0], parts[8][1]), (parts[10][0], parts[10][1]), (0, 0, 255), 2)


    if 140 < hip_ankle < 180:
        cv2.line(image_np, (parts[12][0], parts[12][1]), (parts[14][0], parts[14][1]), (0, 255, 0), 2)
        cv2.line(image_np, (parts[14][0], parts[14][1]), (parts[16][0], parts[16][1]), (0, 255, 0), 2)
    else:
        cv2.line(image_np, (parts[12][0], parts[12][1]), (parts[14][0], parts[14][1]), (0, 0, 255), 2)
        cv2.line(image_np, (parts[14][0], parts[14][1]), (parts[16][0], parts[16][1]), (0, 0, 255), 2)
    
    if 140 < shoulder_knee < 180:
        cv2.line(image_np, (parts[6][0], parts[6][1]), (parts[12][0], parts[12][1]), (0, 255, 0), 2)
        cv2.line(image_np, (parts[12][0], parts[12][1]), (parts[14][0], parts[14][1]), (0, 255, 0), 2)
    else:
        cv2.line(image_np, (parts[6][0], parts[6][1]), (parts[12][0], parts[12][1]), (0, 0, 255), 2)
        cv2.line(image_np, (parts[12][0], parts[12][1]), (parts[14][0], parts[14][1]), (0, 0, 255), 2)


    # for edge in KEYPOINT_EDGES:
    #     x1 = int(keypoints[0][0][edge[0]][1] * width)
    #     y1 = int(keypoints[0][0][edge[0]][0] * height)

    #     x2 = int(keypoints[0][0][edge[1]][1] * width)
    #     y2 = int(keypoints[0][0][edge[1]][0] * height)
    
    #     cv2.line(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("pose estimation", image_np)
    
    if cv2.waitKey(10) & 0xFF==ord('q'):
        break
        
cap.release()
cv2.destroyAllWindows()
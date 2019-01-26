

import time

import cv2
import numpy as np
import argparse
import gluoncv as gcv
import gluoncv.utils as utils
import mxnet as mx
import camera

# Load the model
net = gcv.model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)

def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_device", dest="video_device",
                        help="Video device # of USB webcam (/dev/video?) [-1 for Jetson]",
                        default=-1, type=int)
    arguments = parser.parse_args()
    return arguments


def read_cam(video_capture):
    if video_capture.isOpened():
        windowName = "ssdObject"
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName, 1280, 720)
        cv2.moveWindow(windowName, 0, 0)
        cv2.setWindowTitle(windowName, "SSD Object Detection")
        while True:
            # Check to see if the user closed the window
            if cv2.getWindowProperty(windowName, 0) < 0:
                # This will fail if the user closed the window; Nasties get printed to the console
                break
            ret_val, frame = video_capture.read()

            frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
            rgb_nd, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=512, max_size=700)

            # # Run frame through network
            class_IDs, scores, bounding_boxes = net(rgb_nd)

            for i in range(len(scores[0])):
              cid = int(class_IDs[0][i].asnumpy())
              cname = net.classes[cid]
              score = float(scores[0][i].asnumpy())
              if score < 0.5:
                break
              x,y,w,h = bbox =  bounding_boxes[0][i].astype(int).asnumpy()
              print(cid, score, bbox)
              tag = "{}; {:.4f}".format(cname, score)
              cv2.rectangle(frame, (x,y), (w, h), (0, 255, 0), 2)
              cv2.putText(frame, tag, (x, y-20),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 1)

            # print(bounding_boxes[0][0][2], bounding_boxes[0][0][3])
            # cv2.rectangle(frame,(384,0),(510,128),(0,255,0),3)
            displayBuf = frame
            cv2.imshow(windowName, displayBuf)
            cv2.waitKey(0)

    else:
        print("camera open failed")


if __name__ == '__main__':
    arguments = parse_cli_args()
    print("Called with args:")
    print(arguments)
    print("OpenCV version: {}".format(cv2.__version__))
    print("Device Number:", arguments.video_device)
    if arguments.video_device == -1:
        video_capture = camera.open_camera(device_number=None)
    else:
        video_capture = camera.open_camera(
            device_number=arguments.video_device)
    read_cam(video_capture)
    video_capture.release()
    cv2.destroyAllWindows()

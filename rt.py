

import time

import cv2
import numpy as np
import argparse
import gluoncv as gcv
import mxnet as mx
import camera

# Load the model
net = gcv.model_zoo.get_model('ssd_512_mobilenet1.0_voc', pretrained=True)

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

            # frame = mx.nd.array(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).astype('uint8')
            # rgb_nd, frame = gcv.data.transforms.presets.ssd.transform_test(frame, short=512, max_size=700)

            # # Run frame through network
            # class_IDs, scores, bounding_boxes = net(rgb_nd)

            displayBuf = frame

            cv2.imshow(windowName, displayBuf)

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

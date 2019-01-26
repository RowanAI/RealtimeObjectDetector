from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import camera
import cv2
import numpy as np
import argparse

model = ResNet50(weights='imagenet')

def parse_cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_device", dest="video_device",
                        help="Video device # of USB webcam (/dev/video?) [-1 for Jetson]",
                        default=-1, type=int)
    arguments = parser.parse_args()
    return arguments


def read_cam(video_capture):
    if video_capture.isOpened():
        windowName = "yolo"
        cv2.namedWindow(windowName, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(windowName, 1280, 720)
        cv2.moveWindow(windowName, 0, 0)
        cv2.setWindowTitle(windowName, "Yolo Object Detection")
        while True:
            # Check to see if the user closed the window
            if cv2.getWindowProperty(windowName, 0) < 0:
                break

            ret_val, frame = video_capture.read()
            frame = cv2.resize(frame,(224,224))
            cv2.imshow(windowName, frame)
            frame = np.expand_dims(frame, axis=0)
            frame = preprocess_input(frame)

            preds = model.predict(frame)
            print(preds)
            print('Predicted:', decode_predictions(preds, top=3)[0])

            
            cv2.waitKey(5)

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

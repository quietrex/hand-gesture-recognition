import argparse
import os
import numpy as np
import cv2
from utils import detector_utils as detector_utils
import tensorflow as tf
from os.path import isfile, join

def normalize(size):
    print(os.listdir("dataset/"))
    poses = os.listdir('dataset/')

    for pose in poses:
        print(">> Working on pose : " + pose)
        subdirs = os.listdir('dataset/' + pose + '/')
        for subdir in subdirs:
            files = os.listdir('dataset/' + pose + '/' + subdir + '/')
            print(">> Working on examples : " + subdir)
            for file in files:
                if(file.endswith(".png")):
                    path = 'dataset/' + pose + '/' + subdir + '/' + file
                    # Read image
                    im = cv2.imread(path)

                    height, width, channels = im.shape
                    if not height == width == size:
                        # Resize image
                        im = cv2.resize(im, (int(size), int(size)), interpolation=cv2.INTER_AREA)
                        # Write image
                        cv2.imwrite(path, im)

def capture_cam(pose, new_path, current_example):
    print("[INFO] Capturing with Video Cam...")
    vs = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(new_path + pose + '.avi', fourcc, 25.0, (640, 480))

    while(vs.isOpened()):
        ret, frame = vs.read()
        if ret:
            # write the frame
            out.write(frame)

            cv2.imshow('frame', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

    # Release everything if job is finished
    vs.release()
    out.release()
    cv2.destroyAllWindows()

    vid = cv2.VideoCapture(new_path + pose + '.avi')

    # Check if the video
    if (not vid.isOpened()):
        print('Error opening video stream or file') 
        return

    print("[INFO] Loading Frozen Video...")
    detection_graph, sess = detector_utils.load_inference_graph()
    sess = tf.Session(graph=detection_graph)
    print("[INFO] Frozen Video is Loaded...")
    
    _iter = 1
    # Read until video is completed
    while(vid.isOpened()):
        # Capture frame-by-frame
        ret, frame = vid.read()
        if ret:
            print('   Processing frame: ' + str(_iter))
            # Resize and convert to RGB for NN to work with
            frame = cv2.resize(frame, (500, 500), interpolation=cv2.INTER_AREA)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect object
            boxes, scores = detector_utils.detect_objects(frame, detection_graph, sess)

            # get region of interest
            res = detector_utils.get_box_image(1, 0.5, scores, boxes, 500, 500, frame)

            # Save cropped image 
            if(res is not None):       
                cv2.imwrite(new_path + current_example + str(_iter) + '.png', cv2.cvtColor(res, cv2.COLOR_RGB2BGR))

            _iter += 1
        # Break the loop
        else:
            break

    print("[INFO] Processed " + str(_iter) + ' frames!')

    vid.release()

    normalize(args["size"])

def main(args):
    pose = args["pose"].lower()
    option = ''
    
    while option != 'y' and option != 'n':
        print('You\'ll now be prompted to record the pose you want to add. Please place your hand-pose beforehand facing the camera.')
        print("Press \'y\' when ready else \'n\' to quit. When finished press \'q\'.")
                
        option = input().lower()
    
    if option == 'y':
        print("[INFO] Generating dataset...")
        currentPath = os.path.join("dataset", pose)
        if not os.path.exists(currentPath):
            current = 1
            new_path = str(os.path.join(currentPath, pose + "_" + str(current))) + "/"
            current_example = pose + "_" + str(current) + "_"
            os.makedirs(new_path)
        else:
            new_dir = os.listdir(currentPath)
            current = len(new_dir) + 1
            new_path = str(os.path.join(currentPath, pose + "_" + str(current))) + "/"
            # Create new example directory
            if not os.path.exists(new_path):
                os.makedirs(new_path)
                current_example = pose + "_" + str(current) + "_"

        capture_cam(pose, new_path, current_example)
    

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--pose", default="garbage", help="Pose file to be created")
    ap.add_argument("-s", "--size", default="128", help="Resize file")
    
    args = vars(ap.parse_args())

    main(args)
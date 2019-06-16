import torch
import cv2
import time
import argparse
import posenet
import time
import os
import glob
from os.path import isfile, join
import pickle
import natsort
from imutils import paths
import numpy as np
import math


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--notxt', action='store_true')
parser.add_argument('--image_dir', type=str, default='./images')
parser.add_argument('--output_dir', type=str, default='./output')
args = parser.parse_args()

threshold = 45


# Compute angle between three points
def getAngle(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang

# Compute angles for each joint    
def computeAngle(a):
    hr1 = (0,0)
    hr2 = (0,10)
      
    r_eye = (a[2][0], a[2][1])
    l_eye = (a[1][0], a[1][1])
    l_ear = (a[3][0], a[3][1])
    r_ear = (a[4][0], a[4][1])
    l_shl = (a[5][0], a[5][1])
    r_shl = (a[6][0], a[6][1])
    l_elb = (a[7][0], a[7][1])
    r_elb = (a[8][0], a[8][1])
    l_wrs = (a[9][0], a[9][1])
    r_wrs = (a[10][0], a[10][1])
    l_hip = (a[11][0], a[11][1])
    r_hip = (a[12][0], a[12][1])
    l_kne = (a[13][0], a[13][1])
    r_kne = (a[14][0], a[14][1])
    l_ank = (a[15][0], a[15][1])
    r_ank = (a[16][0], a[16][1])
    
    v = []
        
    v.append(getAngle(l_wrs,l_elb,l_shl)) # left elbow
    v.append(getAngle(l_elb,l_shl,l_hip)) # left shoulder
    v.append(getAngle(l_shl,l_hip,l_kne)) # left hip
    v.append(getAngle(l_hip,l_kne,l_ank)) # left knee
    
    v.append(getAngle(r_wrs,r_elb,r_shl)) # right elbow
    v.append(getAngle(r_elb,r_shl,r_hip)) # right shoulder
    v.append(getAngle(r_shl,r_hip,r_kne)) # right hip
    v.append(getAngle(r_hip,r_kne,r_ank)) # right knee
    
    return v
    
# Check if an angle is within a range
def angleInRange(m,a,t):
    # print (m,a,t)
    if (m-t <= a <= m+t):
        return True
    else:
        return False
    
# Compute Score
def computeScore(m,a,t):
    dist = []
    for i in range(0, len(a)):
        if (angleInRange(m[i],a[i],t)):
            dist.append(1)
        else:
            dist.append(0)
    
    score = 0
    for i in range(0, len(dist)):
        if dist[i] == 1:
         score += 1
    
    return score/8






def addWatermark(icon, correct, imagePhoto, string, color):
    # load the watermark image, making sure we retain the 4th channel
    # which contains the alpha transparency
    watermark = cv2.imread(icon, cv2.IMREAD_UNCHANGED)
    (wH, wW) = watermark.shape[:2]


    
 

    # split the watermark into its respective Blue, Green, Red, and
    # Alpha channels; then take the bitwise AND between all channels
    # and the Alpha channels to construct the actaul watermark
    # NOTE: I'm not sure why we have to do this, but if we don't,
    # pixels are marked as opaque when they shouldn't be
    if correct > 0:
	    (B, G, R, A) = cv2.split(watermark)
	    B = cv2.bitwise_and(B, B, mask=A)
	    G = cv2.bitwise_and(G, G, mask=A)
	    R = cv2.bitwise_and(R, R, mask=A)
	    watermark = cv2.merge([B, G, R, A])

    image = imagePhoto
    (h, w) = image.shape[:2]
    image = np.dstack([image, np.ones((h, w), dtype="uint8") * 255])

    # construct an overlay that is the same size as the input
    # image, (using an extra dimension for the alpha transparency),
    # then add the watermark to the overlay in the bottom-right
    # corner
    overlay = np.zeros((h, w, 4), dtype="uint8")

    #print(overlay.shape)
    #print(watermark.shape)
    for i in range(wH):
        for j in range(wW):
            overlay[i][j] = watermark[i][j]
        
    #overlay[h - wH - 10:h - 10, w - wW - 10:w - 10] = watermark

    # blend the two images together using transparent overlays
    output = image.copy()
    cv2.addWeighted(overlay, 0.25, output, 1.0, 0, output)

    i

    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(output, string, (32, 20), font, 0.8, color, 2, cv2.LINE_AA)

    return output


def addStats(imagePhoto, color, hits, misses):
    image = imagePhoto
    (h, w) = image.shape[:2]
    

    
    hitString =  "hits: " + str(hits)
    missString = "misses: " + str(misses)
    accuracy= "0.0"
    if(hits + misses != 0):
        accuracy = (hits / (hits + misses)) * 100  
        accuracy = str(round(accuracy, 2))
    accuracyString = "accuracy: " + accuracy + "%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image,hitString, (w - 230, h - 45), font, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(image, missString, (w - 230, h -25), font, 0.8, color, 2, cv2.LINE_AA)
    cv2.putText(image, accuracyString, (w - 230, h -5), font, 0.8, color, 2, cv2.LINE_AA)

    return image


def convertVideoToFrames(videoPath):
    ## This step takes a video and separates it into frames
    # Playing video from file:
    cap = cv2.VideoCapture(videoPath)
    cap.set(cv2.CAP_PROP_FPS, 2)
    

    try:
        if not os.path.exists('data'):
            os.makedirs('data')
    except OSError:
        print ('Error: Creating directory of data')


    FPS_VAR = cap.get(cv2.CAP_PROP_FPS)
    print(FPS_VAR)
    currentFrame = 0
    start = time.time()
    time_to_split_video = 0
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()

        # Saves image of the current frame in jpg file
        name = './data/frame' + str(currentFrame) + '.jpg'
        print ('Creating...' + name)
        if not ret: break
        cv2.imwrite(name, frame)

        # To stop duplicate images
        currentFrame += 1
        

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    time_to_split_video = time.time() - start
    print('time to split video:' , time_to_split_video , 'seconds')
    return FPS_VAR


def processframes():
    vectorList0 = []
    vectorList1 = []
    image_dir = './data'
    output_dir = './output'
    modelglobal = 101
    model = posenet.load_model(modelglobal)
    model = model.cuda()
    output_stride = model.output_stride

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    filenames = [
        f.path for f in os.scandir(image_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]
    filenames = natsort.natsorted(filenames)
    start = time.time()
    hits = 0
    misses = 0
    for f in filenames:


        
       
        input_image, draw_image, output_scale = posenet.read_imgfile(f, scale_factor= 1.0, output_stride=output_stride)
        with torch.no_grad():
            input_image = torch.Tensor(input_image).cuda()

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = model(input_image)

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(0),
                offsets_result.squeeze(0),
                displacement_fwd_result.squeeze(0),
                displacement_bwd_result.squeeze(0),
                output_stride=output_stride,
                max_pose_detections=10,
                min_pose_score=0.25)

        keypoint_coords *= output_scale

        if output_dir:
            draw_image = posenet.draw_skel_and_kp(
                draw_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.25, min_part_score=0.0)

            if  ( pose_scores[0] != 0 and pose_scores[1] != 0):
                print( keypoint_coords[0])
                #processing goes here 
                angle_m = computeAngle(keypoint_coords[0])
                print(angle_m)  
                angle_a = computeAngle(keypoint_coords[1])
                print(angle_a) 
                result = computeScore(angle_m, angle_a, threshold)
                print("Accuracy:" , result)
                if (result > 0.8):
                    hits = hits + 1
                    draw_image = addWatermark("./images/checkmark.png", 1,  draw_image, "Hit", (0, 255, 0))
                else:
                    misses = misses + 1
                    draw_image = addWatermark("./images/crossmark.png", 1,  draw_image, "Miss", (0, 0, 255))
            draw_image = addStats( draw_image, ( 51, 255, 255), hits, misses)

            cv2.imwrite(os.path.join(output_dir, os.path.relpath(f, image_dir)), draw_image)

        if not False:
            print()
            print("Results for image: %s" % f)
            for pi in range(len(pose_scores)):
                if pose_scores[pi] == 0.:
                    break

                print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                print()
              
                   
                
                for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                    #print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))
                    if( pose_scores[0] == 0 or pose_scores[1] == 0):
                        break
                    if pi == 0:
                        vectorList0.append(c)
                        #print(c)
                        
                    if pi == 1:
                        vectorList1.append(c)
                        

    print('Average FPS:', len(filenames) / (time.time() - start))
    time_to_process = time.time() - start
    print(len(vectorList0))
    print(len(vectorList1))
    print('time to process photos:',  time_to_process, "seconds" )
    with open('listfile0.data', 'wb') as filehandle:  
        # store the data as binary data stream
        pickle.dump(vectorList0, filehandle)
    with open('listfile1.data', 'wb') as filehandle:  
        # store the data as binary data stream
        pickle.dump(vectorList1, filehandle)





def convert_frames_to_video(FPS):
    img_array = []
    size = 0
    filenames = [
        f.path for f in os.scandir('./output') if f.is_file() and f.path.endswith(('.png', '.jpg'))]
    filenames = natsort.natsorted(filenames)

    for filename in filenames:
        print(filename)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        img_array.append(img)
 
    
    out = cv2.VideoWriter('project.avi',cv2.VideoWriter_fourcc(*'DIVX'), FPS, size)
 
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()

def main():
    
    #FPS_VAR = convertVideoToFrames('video/modeltargetsame.avi')
    FPS_VAR = convertVideoToFrames('video/modeltargetdif.avi')
    #FPS_VAR = convertVideoToFrames('video/modeltargetstill.avi')
    #FPS_VAR = convertVideoToFrames('video/girlduocover.mp4')
    #FPS_VAR = convertVideoToFrames('video/girlduocover2.mp4')
    #FPS_VAR = convertVideoToFrames('video/tapdance.mp4')
    #FPS_VAR = convertVideoToFrames('video/samplestill.avi')
    processframes()
    convert_frames_to_video(FPS_VAR)
    

if __name__ == "__main__":
    main()
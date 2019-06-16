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


def getFrame(sec, cap, outputFolder, currentFrame):
    cap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    hasFrames,image = cap.read()
    if hasFrames:
        cv2.imwrite( outputFolder + "/frame "+currentFrame+".jpg", image)     # save frame as JPG file
    return hasFrames



def convertVideoToFrames(videoPath, outputFolder):
    ## This step takes a video and separates it into frames
    # Playing video from file:
    cap = cv2.VideoCapture(videoPath)
    
    
    

    try:
        if not os.path.exists(outputFolder):
            os.makedirs(outputFolder)
    except OSError:
        print ('Error: Creating directory of data')

    #cap.set(cv2.CAP_PROP_FPS, 20)
    FPS_VAR = cap.get(cv2.CAP_PROP_FPS)
    print(FPS_VAR)
    currentFrame = 0
    start = time.time()
    time_to_split_video = 0
    #while(True):
        # Capture frame-by-frame
        #ret, frame = cap.read()

        # Saves image of the current frame in jpg file
        #name = outputFolder +'/frame' + str(currentFrame) + '.jpg'
        #print ('Creating...' + name)
        #if not ret: break
        #cv2.imwrite(name, frame)

        # To stop duplicate images
        #currentFrame += 1



    sec = 0
    frameRate = 1/25 #it will capture image in each 0.5 second
    success = getFrame(sec, cap, outputFolder, str(currentFrame))
    while success:
        sec = sec + frameRate
        sec = round(sec, 2)
        success = getFrame(sec, cap, outputFolder, str(currentFrame))
        #Saves image of the current frame in jpg file
        name = outputFolder +'/frame' + str(currentFrame) + '.jpg'
        print ('Creating...' + name)
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
    p0 = []
    p1 = []
    image_dir_model = './solomodelframes'
    image_dir_target = './solotargetframes'
    output_dir_model = './outputmodelframes'
    output_dir_target = './outputtargetframes'
    modelglobal = 101
    model = posenet.load_model(modelglobal)
    model = model.cuda()
    output_stride = model.output_stride

    if output_dir_model:
        if not os.path.exists(output_dir_model):
            os.makedirs(output_dir_model)
    if output_dir_target:
        if not os.path.exists(output_dir_target):
            os.makedirs(output_dir_target)
    filenames_model = [
        f.path for f in os.scandir(image_dir_model) if f.is_file() and f.path.endswith(('.png', '.jpg'))]
    filenames_model = natsort.natsorted(filenames_model)
    filenames_target = [
        f.path for f in os.scandir(image_dir_target) if f.is_file() and f.path.endswith(('.png', '.jpg'))]
    filenames_target = natsort.natsorted(filenames_target)
    start = time.time()
   
    for f in filenames_model:
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

        if output_dir_model:
            draw_image = posenet.draw_skel_and_kp(
                draw_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.25, min_part_score=0.0)
            cv2.imwrite(os.path.join(output_dir_model, os.path.relpath(f, image_dir_model)), draw_image)

        if not False:
            print()
            print("Results for image: %s" % f)
            for pi in range(len(pose_scores)):
                if pi == 1:
                    break

                print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                print()
                print(keypoint_coords[0])
                vectorList0.append(keypoint_coords[0])
                p0.append(pose_scores[0])

              
                   
                
                
                        
                        
                    
                        

    print('Average FPS:', len(filenames_model) / (time.time() - start))
    time_to_process = time.time() - start
    print(len(vectorList0))
    
    print('time to process photos:',  time_to_process, "seconds" )
    with open('listfile0.data', 'wb') as filehandle:  
        # store the data as binary data stream
        pickle.dump(vectorList0, filehandle)
    with open('listfile0pscores.data', 'wb') as filehandle:  
        # store the data as binary data stream
        pickle.dump(p0, filehandle)
    
    
    start = time.time()
    for f in filenames_target:
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

        if output_dir_target:
            draw_image = posenet.draw_skel_and_kp(
                draw_image, pose_scores, keypoint_scores, keypoint_coords,
                min_pose_score=0.25, min_part_score=0.0)
            cv2.imwrite(os.path.join(output_dir_target, os.path.relpath(f, image_dir_target)), draw_image)

        if not False:
            print()
            print("Results for image: %s" % f)
            for pi in range(len(pose_scores)):
                if pi == 1:
                    break
                
                print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                print()
                print(keypoint_coords[0])
                vectorList1.append(keypoint_coords[0])
                p1.append(pose_scores[0])
                   
                
               
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    print('Average FPS:', len(filenames_model) / (time.time() - start))
    print(len(vectorList0))
    print(len(vectorList1))
    with open('listfile1.data', 'wb') as filehandle:  
        # store the data as binary data stream
        pickle.dump(vectorList1, filehandle)
    with open('listfile1pscores.data', 'wb') as filehandle:  
        # store the data as binary data stream
        pickle.dump(p1, filehandle)    
    return [vectorList0, vectorList1, p0 , p1]





def convert_frames_to_video(FPS, startIndex, endIndex, v0, v1, p0, p1):
    img_array_model = []
    img_array_target = []
    img_array = []
    labeledImages = []
    final = []
    size = 0
    filenames_model = [
        f.path for f in os.scandir('./outputmodelframes') if f.is_file() and f.path.endswith(('.png', '.jpg'))]
    filenames_model = natsort.natsorted(filenames_model)
    print(len(filenames_model))

    filenames_target = [
        f.path for f in os.scandir('./outputtargetframes') if f.is_file() and f.path.endswith(('.png', '.jpg'))]
    filenames_target = natsort.natsorted(filenames_target)
    print(len(filenames_target))

    if (len(filenames_target) > len(filenames_model)):
        filenames_target = filenames_target[startIndex:endIndex]
       


    for filename_model in filenames_model:
        print(filename_model)
        img = cv2.imread(filename_model)
        height, width, layers = img.shape
        size = (width,height)
        img_array_model.append(img)
        
    
    for filename_target in filenames_target:
        print(filename_target)
        img = cv2.imread(filename_target)
        height, width, layers = img.shape
        size = (width,height)
        img_array_target.append(img)

    for i in range(len(img_array_model)):
        img1 = img_array_model[i]
        img2 = img_array_target[i]
        h1, w1, layers = img1.shape
        h2, w2, layers = img2.shape
        

        #create empty matrix
        vis = np.zeros((max(h1, h2), w1+w2,3), np.uint8)

        #combine 2 images
        vis[:h1, :w1] = img1
        vis[:h2, w1:w1+w2,:3] = img2
        img_array.append(vis)


    output_dir = './timeframes'

    if output_dir:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    hits = 0
    misses = 0
    for i in range(len(v0)):
        
        if(p0[i] != 0 and p1[i+ startIndex] != 0):
            #processing goes here 
            angle_m = computeAngle(v0[i])
            print(angle_m)  
            angle_a = computeAngle(v1[i + startIndex])
            print(angle_a) 
            result = computeScore(angle_m, angle_a, 40)
            print("Accuracy:" , result)
            if (result > 0.8):
                hits = hits + 1
                labeledImages.append( addWatermark("./images/checkmark.png", 1,  img_array[i], "Hit", (0, 255, 0)) )
                print(labeledImages[i].shape)
            else:
                misses = misses + 1
                labeledImages.append( addWatermark("./images/crossmark.png", 1,  img_array[i], "Miss", (0, 0, 255)) )
                print(labeledImages[i].shape)
                
        else:
            labeledImages.append(img_array[i])
            print(labeledImages[i].shape)
        labeledImages[i] = addStats( labeledImages[i], ( 51, 255, 255), hits, misses)

        
       
        #cv2.imwrite(os.path.join(output_dir, draw_image)
        cv2.imwrite( output_dir + "/frame "+str(i)+".jpg", labeledImages[i])

    filenames = [
        f.path for f in os.scandir(output_dir) if f.is_file() and f.path.endswith(('.png', '.jpg'))]
    filenames = natsort.natsorted(filenames)

    for filename in filenames:
        print(filename)
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width,height)
        final.append(img)
 
    
    out = cv2.VideoWriter('project_time.avi',cv2.VideoWriter_fourcc(*'DIVX'), FPS, size)

   
   
    for i in range(len(final)):
        out.write(final[i])
    out.release()

def findClosestOverlap(v0,v1):
    if( len(v0) == len(v1) ):
        return  [0 , len(v0)]
    if( len(v0) > len(v1)):
        longVector = v0
        shortVector = v1
    else:
        shortVector = v0
        longVector = v1
    max = [0 , 0]
    for i in range(len(longVector) - len(shortVector)):
        hits = 0
        for j in range(len(shortVector)):
            angle_s = computeAngle(shortVector[j])
            angle_l = computeAngle(longVector[i+j])
            result = computeScore(angle_s, angle_l, 40)
            if(result > 0.8):
                hits = hits + 1
        if (hits > max[0]):
            max = [hits, i] 
    print(max[1])
    return [max[1], max[1] + len(shortVector) ]




    
        
    
    

def main():
    
    
    FPS_VAR1 = convertVideoToFrames('solovideo/target3.mp4', "solomodelframes")
    FPS_VAR2 = convertVideoToFrames('solovideo/target3pass.mp4', "solotargetframes")
    
    v0 , v1 , p0, p1= processframes()

    with open('listfile0.data', 'rb') as filehandle:  
        # read the data as binary data stream
        v0 = pickle.load(filehandle)
    
    with open('listfile1.data', 'rb') as filehandle: 
        # read the data as binary data stream
        v1 = pickle.load(filehandle)
    with open('listfile0pscores.data', 'rb') as filehandle:  
        # read the data as binary data stream
        p0 = pickle.load(filehandle)
    
    with open('listfile1pscores.data', 'rb') as filehandle:  
        # read the data as binary data stream
        p1 = pickle.load(filehandle)

    print(len(v0))
    print(len(v1))
    print(len(p0))
    print(len(p1))

    #print(v1)

    startIndex , endIndex = findClosestOverlap(v0, v1)
    convert_frames_to_video(25, startIndex, endIndex, v0, v1 , p0 , p1)
    

if __name__ == "__main__":
    main()
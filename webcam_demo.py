import torch
import cv2
import time
import argparse
import posenet
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--model', type=int, default=101)
parser.add_argument('--cam_id', type=int, default=0)
parser.add_argument('--cam_width', type=int, default=1280)
parser.add_argument('--cam_height', type=int, default=720)
parser.add_argument('--scale_factor', type=float, default=0.7125)
args = parser.parse_args()
filename = 'live.avi'

def main():
    model = posenet.load_model(args.model)
    model = model.cuda()
    output_stride = model.output_stride

    cap = cv2.VideoCapture(args.cam_id)
    cap.set(3, args.cam_width)
    cap.set(4, args.cam_height)
    
    
    
    # Define the codec and create VideoWriter object
    vtype = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(filename, vtype, 20, (args.cam_width,args.cam_height ))
    
    start = time.time()
    frame_count = 0
    time_elapsed = 0
    while  time_elapsed < 15:
        time_elapsed = time.time() - start
        input_image, display_image, output_scale = posenet.read_cap(
            cap, scale_factor=args.scale_factor, output_stride=output_stride)

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
                min_pose_score=0.15)

        keypoint_coords *= output_scale

        # TODO this isn't particularly fast, use GL for drawing and display someday...
        overlay_image = posenet.draw_skel_and_kp(
            display_image, pose_scores, keypoint_scores, keypoint_coords,
            min_pose_score=0.15, min_part_score=0.1)
        

        cv2.imshow('posenet', overlay_image)
        
        ret, frame = cap.read()
        
        if ret==True:
            out.write(overlay_image)
        
        frame_count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        f = 'not described image'
        if not False:
            print()
            print("Results for image: %s" % f)
            for pi in range(len(pose_scores)):
                if pose_scores[pi] == 0.:
                    break
                print('Pose #%d, score = %f' % (pi, pose_scores[pi]))
                for ki, (s, c) in enumerate(zip(keypoint_scores[pi, :], keypoint_coords[pi, :, :])):
                    print('Keypoint %s, score = %f, coord = %s' % (posenet.PART_NAMES[ki], s, c))

        print('Average FPS: ', frame_count / (time.time() - start))
   

    # Release everything if job is finished
    cap.release()
    out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
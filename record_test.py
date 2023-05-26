from roboflowoak import RoboflowOak
import cv2
import time
import numpy as np
import depthai


if __name__ == '__main__':

    # instantiating an object (rf) with the RoboflowOak module
    rf = RoboflowOak(model="sandman", confidence=0.05, overlap=0.5,
    version="3", api_key="35csIGZ7d2GUtQvEXMKD", rgb=True,
    depth=True, device=None, blocking=True)
    
    res_w, res_h = rf.getCameraSize(depthai.CameraBoardSocket.RGB)
    # setting parameters for video output
    filename = "testoutputvideo.mp4"
    fps = 20.0
    frame_width = res_w
    frame_height = res_h

    # initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (frame_width, frame_height))

    # Running our model and displaying the video output with detections
    while True:
        t0 = time.time()
        # The rf.detect() function runs the model inference
        result, frame, raw_frame, depth = rf.detect()
        #result, frame, raw_frame, depth = rf.detect(visualize=True)
        predictions = result["predictions"]
        #{
        #    predictions:
        #    [ {
        #        x: (middle),
        #        y:(middle),
        #        width:
        #        height:
        #        depth: ###->
        #        confidence:
        #        class:
        #        mask: {
        #    ]
        #}
        #frame - frame after preprocs, with predictions
        #raw_frame - original frame from your OAK
        #depth - depth map for raw_frame, center-rectified to the center camera

        # timing: for benchmarking purposes
        t = time.time()-t0
        print("INFERENCE TIME IN MS ", 1/t)
        print("PREDICTIONS ", [p.json() for p in predictions])

        # setting parameters for depth calculation
        max_depth = np.amax(depth)
        cv2.imshow("depth", depth/max_depth)
        # displaying the video feed as successive frames
        cv2.imshow("frame", frame)

        # save frame to video output
        out.write(frame)

        # how to close the OAK inference window / stop inference: CTRL+q or CTRL+c
        if cv2.waitKey(1) == ord('q'):
            break
    
    # release video writer and close windows
    out.release()
    cv2.destroyAllWindows()
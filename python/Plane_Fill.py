import numpy as np
import cv2 as cv
from zed import ZED
from time import time

newVal = (1,1,1)

loDiff, upDiff = (0.01,0.01,0.01), (0.01,0.01,0.01)
loDiff2, upDiff2 = (0.1,0.1,0.1), (0.1,0.1,0.1)

zed = ZED()
# zed.open(svo_file='/home/mint/Desktop/test_zed/data/230116_M327/auto_v.svo', svo_realtime=True, min_depth=0.2)
zed.open(svo_file='/home/mint/Desktop/test_zed/data/220902_Gym/short.svo', svo_realtime=True, min_depth=0.2)
# zed.open(min_depth=0.2)

zed_info = zed.camera.get_camera_information()
width = zed_info.camera_resolution.width
height = zed_info.camera_resolution.height
seed_x_n=8
seed_y_n=4

grad_size = 10

xs = np.linspace(0, width,  seed_x_n+2, dtype=np.int32)[1:-1]
ys = np.linspace(0, height, seed_y_n+2, dtype=np.int32)[1:-1]
xx, yy = np.meshgrid(xs, ys)
seed_pts = np.dstack((xx, yy)).reshape(-1, 2)

while zed.is_open():
    if zed.grab():
        d_v = zed.get_depth()*500
        np.nan_to_num(d_v, copy=False, nan=0, posinf=0, neginf=0)
        color,_,_ = zed.get_images()
        norm_dv = d_v-d_v.min()
        norm_dv = norm_dv/d_v.max()
        
        #Increase step -> Decrease noise, Make contour
        grad_x = (d_v[:,grad_size:]-d_v[:,:-grad_size])/(2*grad_size)
        grad_y = (d_v[grad_size:]-d_v[:-grad_size])/(2*grad_size)
        
        direction = np.dstack((-grad_x[grad_size//2:-grad_size//2], -grad_y[:,grad_size//2:-grad_size//2], norm_dv[grad_size//2:-grad_size//2,grad_size//2:-grad_size//2]))
        
        magnitude = np.sqrt((direction**2).sum(axis=2))
        magnitude = np.dstack((magnitude, magnitude, magnitude))
        
        normal = (direction/magnitude+1)/2          #Make -1~1 scale to 0~1 scale,   num<0.5 : minus
        
        rows, cols = normal.shape[:2]
        
        label_arr = np.zeros_like(normal[:,:,0])
        label_n = 1
        for pts in seed_pts:
            if label_arr[pts[1], pts[0]] == 0:
                mask1 = np.zeros((rows+2, cols+2), np.uint8)
                mask2 = np.zeros((rows+2, cols+2), np.uint8)
                retval1 = cv.floodFill(normal.copy(), mask1, pts, newVal, upDiff, loDiff, flags=8)
                retval2 = cv.floodFill(normal.copy(), mask2, pts, newVal, upDiff2, loDiff2, flags=cv.FLOODFILL_FIXED_RANGE)
                    
                if retval1[0] > 3000 and retval2[0] > 3000:
                    mask = mask1*mask2
                    label_arr[(mask==1)[1:-1,1:-1]] = label_n
                    label_n += 1
        
        print(zed.camera.get_current_fps())
        
        cv.imshow('normal', normal)
        cv.imshow('color', color)
        cv.imshow('labeling', label_arr/label_arr.max())
        
        
        key = cv.waitKey(1)
        if key == ord(' '):
            key = cv.waitKey(0)
        if key == 27:
            break;
    
cv.destroyAllWindows()
zed.close()

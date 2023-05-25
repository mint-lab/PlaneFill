import numpy as np
import cv2 as cv
import pyzed.sl as sl
import sys

class ZED():
    '''StereoLabs ZED image grabber'''

    def __init__(self):
        '''Instantiate the camera and others'''
        self.camera = sl.Camera()
        self.is_opened = False
        self.runtime_params = sl.RuntimeParameters()
        self.left_image = sl.Mat()
        self.right_image = sl.Mat()
        self.depth_image = sl.Mat()
        self.depth_float = sl.Mat()
        self.xyz = sl.Mat()
        self.sensor_data = sl.SensorsData()

        self.MAP_RESOLUTION = {'2k': sl.RESOLUTION.HD2K, '1080p': sl.RESOLUTION.HD1080, '720p': sl.RESOLUTION.HD720, 'vga': sl.RESOLUTION.VGA}
        self.MAP_DEPTH_MODE = {'neural': sl.DEPTH_MODE.NEURAL, 'ultra': sl.DEPTH_MODE.ULTRA, 'quality': sl.DEPTH_MODE.QUALITY, 'performance': sl.DEPTH_MODE.PERFORMANCE}
        self.MAP_COORD_UNIT = {'mm': sl.UNIT.MILLIMETER, 'cm': sl.UNIT.CENTIMETER, 'm': sl.UNIT.METER}

    def open(self, resolution='720p', fps=30, depth_mode='neural', coord_unit='m', min_depth = 0.4, svo_file='', svo_realtime=False, config=sl.InitParameters()):
        '''Start the camera'''
        if svo_file:
            config.set_from_svo_file(svo_file)
            config.svo_real_time_mode = svo_realtime
        if resolution is not None:
            config.camera_resolution = self.MAP_RESOLUTION[resolution.lower()]
        if fps is not None:
            config.camera_fps = fps
        if depth_mode is not None:
            config.depth_mode = self.MAP_DEPTH_MODE[depth_mode.lower()]
        if coord_unit is not None:
            config.coordinate_units = self.MAP_COORD_UNIT[coord_unit.lower()]
        if min_depth is not None:
            config.depth_minimum_distance = min_depth
        config.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
        status = self.camera.open(config)
        self.is_opened = (status == sl.ERROR_CODE.SUCCESS)
        return self.is_opened

    def is_open(self):
        '''Check whether the camera is openend or not'''
        return self.is_opened

    def close(self):
        '''Stop the camera'''
        if self.is_opened:
            self.is_opened = False
            return self.camera.close()
        return False

    def grab(self):
        '''Grab a set of sensor data'''
        status = self.camera.grab(self.runtime_params)
        self.camera.get_sensors_data(self.sensor_data, sl.TIME_REFERENCE.IMAGE)
        return status == sl.ERROR_CODE.SUCCESS

    def get_timestamp(self):
        '''Retrieve the timestamp when images and sensor data are acquired'''
        timestamp = self.camera.get_timestamp(sl.TIME_REFERENCE.IMAGE)
        return timestamp.get_seconds()

    def get_images(self):
        '''Retrieve all images'''
        self.camera.retrieve_image(self.left_image, sl.VIEW.LEFT)
        self.camera.retrieve_image(self.right_image, sl.VIEW.RIGHT)
        self.camera.retrieve_image(self.depth_image, sl.VIEW.DEPTH)
        return self.left_image.get_data()[:,:,:3], self.right_image.get_data()[:,:,:3], self.depth_image.get_data()[:,:,:3]

    def get_depth(self):
        '''Retrieve the depth data'''
        self.camera.retrieve_measure(self.depth_float, sl.MEASURE.DEPTH)
        return self.depth_float.get_data()

    def get_tracking_pose(self):
        '''Retrieve the pose (orientation and position) from positional tracking'''
        pose = sl.Pose()
        status = self.camera.get_position(pose, sl.REFERENCE_FRAME.WORLD)
        return status == sl.POSITIONAL_TRACKING_STATE.OK, pose.get_orientation().get(), pose.get_translation().get()

    def get_imu_orientation(self):
        '''Retrieve the orientation from IMU'''
        pose = self.sensor_data.get_imu_data().get_pose()
        return pose.get_orientation().get()

    def get_imu_linear_acc(self):
        '''Retrieve the Linear Acceleration from IMU'''
        return self.sensor_data.get_imu_data().get_linear_acceleration()

    def get_barometer(self):
        '''Retrieve the barometer value (unit: [hPa])'''
        return self.sensor_data.get_barometer_data().pressure

    def get_magnetic_field(self):
        return self.sensor_data.get_magnetometer_data().get_magnetic_field_calibrated()

    def get_xyz(self):
        '''Retrieve the depth data as point cloud'''
        self.camera.retrieve_measure(self.xyz, sl.MEASURE.XYZ)
        return self.xyz.get_data()[:,:,:3]

    def start_tracking(self):
        '''Start positional tracking'''
        tracking_param = sl.PositionalTrackingParameters()
        status = self.camera.enable_positional_tracking(tracking_param)
        return status == sl.ERROR_CODE.SUCCESS

    def start_recording(self, file):
        '''Start SVO recording'''
        recording_param = sl.RecordingParameters(file, sl.SVO_COMPRESSION_MODE.H264)
        status = self.camera.enable_recording(recording_param)
        return status == sl.ERROR_CODE.SUCCESS

def print_zed_info(zed:ZED):
    '''Print camera information of the ZED camera'''
    cam_info = zed.camera.get_camera_information()

    print('* Device information')
    print(f'  * Model     : {cam_info.camera_model}')
    print(f'  * Serial    : {cam_info.serial_number}')
    print(f'  * Firmware  : {cam_info.camera_firmware_version}')
    print(f'  * Connection: {cam_info.input_type}')
    print(f'  * Resolution: {cam_info.camera_resolution.width}x{cam_info.camera_resolution.height} (FPS: {cam_info.camera_fps})')
    print('')
    print('* Intrinsic parameters')
    param = cam_info.calibration_parameters.left_cam
    print(f'  * Left : fx={param.fx}, fy={param.fy}, cx={param.cx}, cy={param.cy}, dist={param.disto}')
    param = cam_info.calibration_parameters.right_cam
    print(f'  * Right: fx={param.fx}, fy={param.fy}, cx={param.cx}, cy={param.cy}, dist={param.disto}')
    print('')
    print('* Extrinsic parameters')
    print(f'  * R: {cam_info.calibration_parameters.R}')
    print(f'  * T: {cam_info.calibration_parameters.T}')
    print('')

def test_zed(resolution='720p', fps=30, depth_mode='neural', zoom=0.5):
    '''Show live images of the ZED camera'''

    zed = ZED()
    zed.open(resolution, fps, depth_mode)
    print_zed_info(zed)
    sys.stdout.flush()

    cv.namedWindow('ZED Live: Color and Depth')
    while zed.is_open():
        if zed.grab():
            # Get all images
            color, _, depth = zed.get_images()

            # Show the images
            depth_color = cv.applyColorMap(depth, cv.COLORMAP_JET)
            merge = cv.resize(np.vstack((color, depth_color)), (0, 0), fx=zoom, fy=zoom)
            cv.imshow('ZED Live: Color and Depth', merge)

        key = cv.waitKey(1)
        if key == ord(' '): # Space
            key = cv.waitKey(0)
        if key == 27:       # ESC
            break

    cv.destroyAllWindows()
    zed.close()



if __name__ == '__main__':
    test_zed()
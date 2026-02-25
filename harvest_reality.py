# harvest_reality.py (Runs on the Pi 5)
import numpy as np
import cv2
import h5py
import time

class RealityHarvester:
    def __init__(self, output_file="prometheus_incubation.h5"):
        self.gs_cam = cv2.VideoCapture(0)
        self.camera_width = 640
        self.camera_height = 480
        self.camera_fov = 90.0
        
        # Open an HDF5 file to store the massive dataset
        print(f"💾 Opening dataset file: {output_file}")
        self.h5_file = h5py.File(output_file, 'a')
        
        # Create a resizable dataset for 4-Channel images
        if 'rgbd_frames' not in self.h5_file:
            self.dataset = self.h5_file.create_dataset(
                'rgbd_frames', 
                shape=(0, 4, self.camera_height, self.camera_width), # PyTorch expects Channels first
                maxshape=(None, 4, self.camera_height, self.camera_width), 
                dtype='uint8',
                compression="gzip"
            )
        else:
            self.dataset = self.h5_file['rgbd_frames']

    def mock_lidar(self):
        # Placeholder for your actual serial LiDAR read
        return np.random.randint(0, 255, 360)

    def capture_and_save(self):
        ret, frame = self.gs_cam.read()
        if not ret: return
        
        raw_lidar = self.mock_lidar()
        depth_channel = np.zeros((self.camera_height, self.camera_width), dtype=np.uint8)
        
        # The LiDAR Projection Math
        for angle in range(360):
            rel_angle = angle if angle <= 180 else angle - 360
            if -45 <= rel_angle <= 45:
                norm_x = (rel_angle + (self.camera_fov / 2)) / self.camera_fov
                pixel_x = int(norm_x * self.camera_width)
                pixel_x = max(0, min(self.camera_width - 1, pixel_x))
                
                distance_val = raw_lidar[angle] 
                cv2.line(depth_channel, (pixel_x, 0), (pixel_x, self.camera_height), 
                         color=int(distance_val), thickness=2)

        # 1. Fuse to 4 Channels (Height, Width, 4)
        rgbd_frame = np.dstack((frame, depth_channel))
        
        # 2. Reshape to PyTorch format (4, Height, Width)
        rgbd_pytorch = np.transpose(rgbd_frame, (2, 0, 1))
        
        # 3. Append to the HDF5 file
        current_size = self.dataset.shape[0]
        self.dataset.resize(current_size + 1, axis=0)
        self.dataset[current_size] = rgbd_pytorch
        
        if current_size % 100 == 0:
            print(f"📸 Harvested {current_size} frames of reality...")

# --- THE RECORDING LOOP ---
harvester = RealityHarvester()

print("👁️ Prometheus is opening its eyes. Press Ctrl+C to stop harvesting.")
try:
    while True:
        harvester.capture_and_save()
        
        # We only need to capture about 5 frames per second. 
        # Any faster is just saving identical images and wasting SD card space.
        time.sleep(0.2) 
        
except KeyboardInterrupt:
    print("\n💤 Closing eyes. Dataset saved.")
    harvester.h5_file.close()
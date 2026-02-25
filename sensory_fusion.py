import numpy as np
import cv2

class SensoryHardware:
    def __init__(self):
        self.gs_cam = cv2.VideoCapture(0)
        self.camera_width = 640
        self.camera_height = 480
        self.camera_fov = 90.0 # Degrees
        
        # Initialize Hailo NPU to accept a 4-Channel input (RGB + Depth)
        # self.hailo = HailoModel("rgbd_encoder.hef")

    def get_fused_visual_cortex(self, raw_lidar_360):
        # 1. Capture the visual reality (RGB)
        ret, frame = self.gs_cam.read()
        
        # 2. Create a blank "Depth" channel (same size as the image)
        depth_channel = np.zeros((self.camera_height, self.camera_width), dtype=np.float32)
        
        # 3. Snipping the LiDAR (Assuming LiDAR 0 is dead center)
        # We only care about the angles between -45 and +45 degrees
        for angle in range(360):
            # Convert 0-360 to -180 to +180
            relative_angle = angle if angle <= 180 else angle - 360
            
            if -45 <= relative_angle <= 45:
                # 4. Map the physical angle to the camera's X-pixel coordinate
                # (Math: Map -45/45 to 0/640)
                normalized_x = (relative_angle + (self.camera_fov / 2)) / self.camera_fov
                pixel_x = int(normalized_x * self.camera_width)
                
                # Ensure it fits on the screen
                pixel_x = max(0, min(self.camera_width - 1, pixel_x))
                
                # We project the LiDAR distance as a vertical line onto the depth channel.
                # (A true 3D projection matrix would use the camera's intrinsic calibration, 
                # but this 2D vertical slice method works perfectly for a flat LiDAR plane).
                distance = raw_lidar_360[angle]
                
                # Draw the depth reading down the center of the image
                cv2.line(depth_channel, (pixel_x, self.camera_height//2 - 20), 
                                        (pixel_x, self.camera_height//2 + 20), 
                                        color=distance, thickness=2)

        # 5. Fuse them together into a 4-Channel Matrix! (R, G, B, Depth)
        rgbd_frame = np.dstack((frame, depth_channel))
        
        # 6. Compress the fused reality through the NPU
        # The NPU now inherently associates the texture of a wall with its physical distance
        # latent_vector = self.hailo.infer(rgbd_frame)
        latent_vector = np.random.rand(512).astype(np.float32) # Mocked
        
        return latent_vector
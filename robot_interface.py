# robot_interface.py
import json
import socket
import threading
import numpy as np
from typing import Dict, Any

class RobotInterface:
    def __init__(self, port=65432):
        self.host = '0.0.0.0' # Listen on all interfaces
        self.port = port
        self.latest_state = {
            "lidar_front": 0.0,
            "ultrasonic": 0.0,
            "visual_objects": [],
            "battery": 100.0
        }
        self.running = True
        self.conn = None
        
        # Start listening in background
        self.thread = threading.Thread(target=self._server_loop, daemon=True)
        self.thread.start()

    def _server_loop(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((self.host, self.port))
            s.listen()
            print(f"🤖 Robot Interface listening on {self.port}...")
            
            while self.running:
                self.conn, addr = s.accept()
                with self.conn:
                    print(f"🔗 Connected to Robot: {addr}")
                    while True:
                        data = self.conn.recv(4096)
                        if not data: break
                        try:
                            # Parse incoming JSON from Pi
                            sensor_data = json.loads(data.decode('utf-8'))
                            self._update_state(sensor_data)
                        except:
                            pass

    def _update_state(self, raw_data):
        """
        Translates Raw Sensors -> Semantic Concepts
        This addresses the GROUNDING problem.
        """
        # 1. Lidar/Ultrasonic Grounding
        dist = raw_data.get("ultrasonic", 100)
        self.latest_state["ultrasonic"] = dist
        
        # Create high-level concepts for the Agent's Working Memory
        if dist < 15:
            self.latest_state["spatial_concept"] = "OBSTACLE_CLOSE"
        elif dist < 50:
            self.latest_state["spatial_concept"] = "OBJECT_NEAR"
        else:
            self.latest_state["spatial_concept"] = "CLEAR_PATH"

        # 2. Visual Grounding (Simple Object Detection labels from Pi)
        # The Pi should run a tiny YOLO/MobileNet and just send labels like "Chair", "Person"
        self.latest_state["visual_objects"] = raw_data.get("objects", [])

    def get_context(self) -> Dict[str, Any]:
        """Returns the current physical reality for the Executive"""
        return self.latest_state

    def send_command(self, action_type: str, payload: str):
        """Sends motor commands back to Pi"""
        if self.conn:
            msg = json.dumps({"action": action_type, "payload": payload})
            self.conn.sendall(msg.encode('utf-8'))
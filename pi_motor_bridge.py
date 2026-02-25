# pi_motor_bridge.py (Runs on Raspberry Pi 5)
import zmq
import serial
import struct

def ignite_motor_bridge(server_ip="192.168.1.100"):
    print("🔌 Starting Pi -> Arduino Motor Bridge...")
    
    # 1. Connect to the Arduino Giga over USB
    # (Check /dev/ttyACM0 or /dev/ttyUSB0 depending on the Pi's mapping)
    try:
        arduino = serial.Serial('/dev/ttyACM0', baudrate=115200, timeout=1)
        print("✅ Arduino Giga connected.")
    except Exception as e:
        print(f"⛔ Could not find Arduino: {e}")
        return

    # 2. Connect to the Server's Motor Spine (Port 5556)
    context = zmq.Context()
    motor_nerve = context.socket(zmq.SUB)
    motor_nerve.connect(f"tcp://{server_ip}:5556")
    motor_nerve.setsockopt_string(zmq.SUBSCRIBE, "") # Listen to everything
    
    print("⚡ Listening for motor intents from Prometheus...")

    # 3. The Passthrough Loop
    while True:
        # Catch the 32 bytes (8 Float32s) from the Julia server
        msg_bytes = motor_nerve.recv()
        
        if len(msg_bytes) == 32:
            # Blast the exact same raw bytes straight into the Arduino's serial buffer.
            # No string conversion, no JSON parsing. Just pure byte throughput.
            arduino.write(msg_bytes)

if __name__ == "__main__":
    ignite_motor_bridge()
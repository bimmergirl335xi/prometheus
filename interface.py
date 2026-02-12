# interface.py
import time
import os
import sys

# Configuration
AGENT_OUTPUT_FILE = "communication.txt"
USER_INPUT_FILE = "user_input.txt"

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def print_agent_messages():
    """Reads new lines from the agent's output file."""
    if not os.path.exists(AGENT_OUTPUT_FILE):
        return

    with open(AGENT_OUTPUT_FILE, "r", encoding='utf-8') as f:
        # Move to the end of file to ignore old history on startup?
        # For now, let's just read everything so you see context.
        content = f.read()
        if content:
            print(content)

def main():
    clear_screen()
    print("--- PROMETHEUS UPLINK ESTABLISHED ---")
    print("Type your message and press ENTER. Press Ctrl+C to exit.\n")

    # Create input file if it doesn't exist
    if not os.path.exists(USER_INPUT_FILE):
        with open(USER_INPUT_FILE, "w") as f: pass

    # Track file position to only show *new* agent messages
    last_pos = 0
    if os.path.exists(AGENT_OUTPUT_FILE):
        last_pos = os.path.getsize(AGENT_OUTPUT_FILE)

    import threading
    import queue
    
    input_queue = queue.Queue()

    def listen_for_input():
        while True:
            try:
                user_text = input() # Blocking input
                input_queue.put(user_text)
            except EOFError:
                break

    # Start input thread so we can print agent messages while waiting for user
    input_thread = threading.Thread(target=listen_for_input, daemon=True)
    input_thread.start()

    while True:
        # 1. Check for new Agent Messages
        if os.path.exists(AGENT_OUTPUT_FILE):
            current_size = os.path.getsize(AGENT_OUTPUT_FILE)
            if current_size > last_pos:
                with open(AGENT_OUTPUT_FILE, "r", encoding='utf-8') as f:
                    f.seek(last_pos)
                    new_text = f.read()
                    if new_text:
                        print(new_text, end="")
                last_pos = current_size

        # 2. Check for new User Input
        while not input_queue.empty():
            text = input_queue.get()
            if text.strip():
                # Append to the input file for the Agent to read
                with open(USER_INPUT_FILE, "a", encoding='utf-8') as f:
                    f.write(text + "\n")
                print(f"\033[32mYou > {text}\033[0m") # Echo back in Green

        time.sleep(0.1)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nUplink closed.")
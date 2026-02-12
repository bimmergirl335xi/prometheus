# mind_reader.py
import time
import os
import sys

# ASCII Colors for visualization
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    
    # Specific Cognitive Colors
    ABSTRACT = '\033[35m' # Purple
    REASON = '\033[36m'   # Cyan
    LANG = '\033[32m'     # Green
    MEM = '\033[90m'      # Dark Grey

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def process_line(line):
    """Parses a log line and formats it based on the cognitive module."""
    line = line.strip()
    if not line: return

    # Timestamp removal for cleaner look (optional)
    # plain_text = line.split(']', 1)[1] if ']' in line else line

    if "ABSTRACT" in line:
        print(f"{Colors.ABSTRACT}{Colors.BOLD}⚡ ABSTRACTION:{Colors.ENDC} {line}")
    elif "REASON" in line or "PLAN" in line:
        print(f"{Colors.REASON}🤔 REASONING:{Colors.ENDC}   {line}")
    elif "LANG" in line or "TALK" in line:
        print(f"{Colors.LANG}🗣️  LANGUAGE:{Colors.ENDC}    {line}")
    elif "MEM" in line:
        print(f"{Colors.MEM}💾 MEMORY:{Colors.ENDC}      {line}")
    elif "META" in line:
        print(f"{Colors.YELLOW}👑 META:{Colors.ENDC}        {line}")
    # Filter out generic system noise to keep the feed focused
    # else:
    #    print(line) 

def main():
    log_file = "thought_steam.log"
    
    # Create file if it doesn't exist
    if not os.path.exists(log_file):
        with open(log_file, 'w') as f:
            f.write("--- NEURAL LINK ESTABLISHED ---\n")

    clear_screen()
    print(f"{Colors.HEADER}{'='*60}")
    print(f"   🧠  COGNITIVE MONITOR - LISTENING TO SUBCONSCIOUS")
    print(f"{'='*60}{Colors.ENDC}\n")

    try:
        # Open the file and go to the end
        with open(log_file, 'r') as f:
            f.seek(0, 2) # Seek to end
            
            while True:
                line = f.readline()
                if not line:
                    time.sleep(0.1) # Sleep briefly to avoid CPU spike
                    continue
                
                process_line(line)

    except KeyboardInterrupt:
        print(f"\n{Colors.RED}Neural Link Severed.{Colors.ENDC}")

if __name__ == "__main__":
    main()
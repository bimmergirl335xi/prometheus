# utils.py
from datetime import datetime
import os
import time
import sys
from config import *

class Term:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    ORANGE = '\033[33m'
    RED = '\033[91m'
    PURPLE = '\033[35m'
    YELLOW = '\033[93m'
    WHITE = '\033[97m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    CLEAR = '\033[2K\r'

STREAM_FILE = "thought_stream.log"

def log(source: str, message: str, color: str = Term.WHITE):
    """
    Logs to console AND broadcasts to the thought stream file.
    Maintains a rolling buffer of the last 100 lines in the file.
    """
    timestamp = time.strftime("%H:%M:%S")
    
    # 1. Print to Main Console (Standard output)
    # We use a try/except here in case color codes fail on some terminals
    try:
        print(f"[{timestamp}] [{color}{source.center(8)}{Term.ENDC}] {message}")
    except:
        print(f"[{timestamp}] [{source.center(8)}] {message}")
    
    # 2. Broadcast to Mind Reader (File output)
    interesting_channels = ["ABSTRACT", "REASON", "PLAN", "LANG", "META", "MEM"]
    
    if any(ch in source for ch in interesting_channels) or "Error" in message:
        try:
            lines = []
            # Read existing lines if file exists
            if os.path.exists(STREAM_FILE):
                with open(STREAM_FILE, "r", encoding='utf-8') as f:
                    lines = f.readlines()
            
            # Append new line
            lines.append(f"[{source}] {message}\n")
            
            # Prune: Keep only the last 100 lines
            if len(lines) > 100:
                lines = lines[-100:]
            
            # Overwrite the file with the pruned list
            with open(STREAM_FILE, "w", encoding='utf-8') as f:
                f.writelines(lines)
                
        except Exception:
            pass # Fail silently to avoid crashing the agent logic

def log_engine_event(engine: str, method: str, input_data: any, output_data: any):
    """
    Logs structured I/O for cognitive engines.
    Maintains a rolling buffer of the last 50 complex entries.
    """
    timestamp = time.strftime("%H:%M:%S")
    
    # Format the entry clearly
    entry = f"════════════════════════════════════════════════════════════\n"
    entry += f"[{timestamp}] {engine}::{method}\n"
    entry += f"► INPUT : {str(input_data)}\n"
    entry += f"► OUTPUT: {str(output_data)}\n"
    entry += "════════════════════════════════════════════════════════════\n"

    try:
        content = ""
        # Read existing content
        if os.path.exists(TRACE_FILE):
            with open(TRACE_FILE, "r", encoding='utf-8') as f:
                content = f.read()
        
        # Split by distinct separator to manage "Entries" not just lines
        separator = "════════════════════════════════════════════════════════════\n"
        entries = content.split(separator)
        
        # Remove empty strings from split
        entries = [e for e in entries if e.strip()]
        
        # Append new entry (cleaned of the separator which split removes)
        # We reconstruct the separator inside the entry variable above, 
        # but for the list logic, we just treat the block as text.
        
        # Actually, simpler logic: Just Append, then Readlines, Keep Last N lines
        # (Text block logic can be brittle if separators change)
        
        with open(TRACE_FILE, "a", encoding='utf-8') as f:
            f.write(entry)

        # Prune if too big (Simpler Rolling Line Buffer)
        with open(TRACE_FILE, "r", encoding='utf-8') as f:
            lines = f.readlines()
            
        if len(lines) > 500: # Approx 50 entries
            with open(TRACE_FILE, "w", encoding='utf-8') as f:
                f.writelines(lines[-500:])
                
    except Exception as e:
        print(f"Trace Log Error: {e}")

def speak(text: str):
    """
    Writes to the external communication file instantly (Unbuffered).
    """
    # Clean ANSI codes
    import re
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    clean_text = ansi_escape.sub('', text)

    # 'a' = Append mode. 
    # encoding='utf-8' ensures emojis don't crash it.
    with open(COMMS_FILE, "a", encoding='utf-8') as f:
        f.write(f"\n🤖 > {clean_text}\n")
        f.flush()            # Flush internal buffer
        os.fsync(f.fileno()) # Force OS to write to disk immediately
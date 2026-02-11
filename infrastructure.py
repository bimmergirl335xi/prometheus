# infrastructure.py
import asyncio
import time
from config import *
from utils import *
from typing import Any, Dict, Optional
from utils import log, Term

class CognitiveNode:
    """
    An autonomous cognitive process (Neuron/Module).
    - Listen: Watches the Global Workspace.
    - Activation: Fires if the signal matches its 'Receptive Field'.
    - Output: Posts result back to Workspace.
    """
    def __init__(self, name: str, workspace: Any, sensitivity: float = 0.5):
        self.name = name
        self.workspace = workspace
        self.sensitivity = sensitivity  # 0.0 to 1.0 (Threshold to fire)
        self.running = False
        self.last_fire = 0.0
        self.refractory_period = 0.1 # Seconds to wait before firing again

    async def run_forever(self):
        """The main heartbeat of the node."""
        self.running = True
        log("SYS", f"⚡ Node Online: {self.name}", Term.GREEN)
        
        while self.running:
            # 1. Sense (Non-destructive peek)
            packet = self.workspace.peek()
            
            if packet and (time.time() - self.last_fire > self.refractory_period):
                # 2. Evaluate (Do I care?)
                relevance = self._calculate_relevance(packet)
                
                # 3. Fire (Process & Respond)
                if relevance > self.sensitivity:
                    # log("NODE", f"🔥 {self.name} activated (Rel: {relevance:.2f})", Term.DIM)
                    
                    try:
                        result_packet = await self.process(packet)
                        
                        if result_packet:
                            self.workspace.post(
                                source=self.name,
                                content=result_packet['content'],
                                salience=relevance * 1.1, # Boost signal slightly
                                type=result_packet.get('type', 'THOUGHT')
                            )
                            self.last_fire = time.time()
                            
                    except Exception as e:
                        log("ERR", f"{self.name} crashed: {e}", Term.RED)

            # Idle wait (Async yield)
            await asyncio.sleep(0.01)

    def _calculate_relevance(self, packet: Dict) -> float:
        """Override this: Return 0.0 to 1.0 based on packet content."""
        return 0.0

    async def process(self, packet: Dict) -> Optional[Dict]:
        """Override this: The actual work logic."""
        return None

class NeuralBus:
    """Message bus for inter-system communication"""
    def __init__(self):
        self.channels: Dict[str, asyncio.Queue] = {
            "network": asyncio.Queue(maxsize=10)
        }

    async def publish(self, channel: str, message: Any) -> bool:
        """Publish message to channel"""
        if channel not in self.channels:
            log("BUS", f"Unknown channel: {channel}", Term.RED)
            return False
        
        try:
            await asyncio.wait_for(
                self.channels[channel].put(message),
                timeout=1.0
            )
            return True
        except asyncio.TimeoutError:
            log("BUS", f"Channel {channel} full, message dropped", Term.YELLOW)
            return False
        except Exception as e:
            log("BUS", f"Publish error: {e}", Term.RED)
            return False

# NOTE: GlobalWorkspace is referenced but was missing from the source.
# Ensure it is imported or defined before running.
class NetworkInterface:
    """Enhanced network interface with caching"""
    def __init__(self, bus: NeuralBus, workspace: 'ActiveGlobalWorkspace'):
        self.bus = bus
        self.workspace = workspace
        self.request_cache: Dict[str, Tuple[str, float]] = {}
        self.cache_ttl = 300

    def fetch(self, query: str) -> str:
        """Fetch data from Wikipedia with caching"""
        if query in self.request_cache:
            result, timestamp = self.request_cache[query]
            if time.time() - timestamp < self.cache_ttl:
                log("NET", f"📦 Cache hit: {query[:30]}...", Term.DIM)
                return result
        
        try:
            url = f"https://en.wikipedia.org/w/api.php?action=opensearch&search={query}&limit=1&format=json"
            response = requests.get(url, timeout=3)
            data = response.json()
            
            if not data[1]:
                return "NO_DATA"
            
            page_url = data[3][0]
            page_response = requests.get(page_url, timeout=3)
            soup = BeautifulSoup(page_response.text, 'html.parser')
            
            paragraphs = soup.select('p')
            result = paragraphs[0].text[:400] if paragraphs else "NO_DATA"
            
            self.request_cache[query] = (result, time.time())
            
            return result
        
        except requests.Timeout:
            log("NET", "⏱️  Request timeout", Term.RED)
            return "NET_ERROR"
        except Exception as e:
            log("NET", f"❌ Network error: {e}", Term.RED)
            return "NET_ERROR"

    async def run(self) -> None:
        """Background network service"""
        loop = asyncio.get_running_loop()
        
        while True:
            try:
                if not self.bus.channels["network"].empty():
                    query = await self.bus.channels["network"].get()
                    result = await loop.run_in_executor(None, self.fetch, query)
                    self.workspace.post(8, "NETWORK", result)
                
                await asyncio.sleep(0.1)
            
            except Exception as e:
                log("NET", f"Service error: {e}", Term.RED)
                await asyncio.sleep(1)

class InputSystem:
    """Enhanced input system with persistent prompt and file commands"""
    def __init__(self, workspace: 'ActiveGlobalWorkspace'):
        self.workspace = workspace
        self.loop = None

        self.input_file = "user_input.txt"
        self.last_pos = 0

    async def run(self) -> None:
        """Background file watcher"""
        log("INPUT", "Watching user_input.txt for commands...", Term.CYAN)
        
        while True:
            try:
                if os.path.exists(self.input_file):
                    current_size = os.path.getsize(self.input_file)
                    
                    if current_size > self.last_pos:
                        with open(self.input_file, "r", encoding='utf-8') as f:
                            f.seek(self.last_pos)
                            new_lines = f.readlines()
                        
                        self.last_pos = current_size
                        
                        for line in new_lines:
                            text = line.strip()
                            if not text: continue
                            
                            log("INPUT", f"Received: {text}", Term.GREEN)
                            
                            # HIGHEST PRIORITY (0) - Drop everything and listen
                            # The lower the number, the higher the priority in the heap
                            
                            if text.upper().startswith("READ "):
                                filename = text[5:].strip()
                                self.workspace.post(0, "USER", {"type": "file", "path": filename})
                            elif text.startswith("SYS_MOD"):
                                self.workspace.post(0, "USER", text)
                            else:
                                self.workspace.post(0, "USER", text)
                                
                    elif current_size < self.last_pos:
                        # File was reset/truncated
                        self.last_pos = 0
                
                await asyncio.sleep(0.2)
                
            except Exception as e:
                log("INPUT", f"Error: {e}", Term.RED)
                await asyncio.sleep(1)
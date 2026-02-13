# infrastructure.py
import asyncio
import time
import os
import requests
from config import *
from utils import *
from typing import Any, Dict, Optional, List, Tuple, TYPE_CHECKING
from utils import log, Term
from dataclasses import dataclass
from bs4 import BeautifulSoup

if TYPE_CHECKING:
    from core_systems import ActiveGlobalWorkspace

# Define Packet type for clarity
Packet = Dict[str, Any]

class CognitiveNode:
    """
    Base class for distributed cognitive 'engines'.
    Each node decides if it cares via _calculate_relevance(packet),
    then may emit a new packet (or None).
    """
    def __init__(self, name: str, workspace: Any, sensitivity: float = 0.5):
        self.name = name
        self.workspace = workspace
        self.sensitivity = float(sensitivity)

    def _calculate_relevance(self, packet: Optional[Packet]) -> float:
        return 0.0

    async def process(self, packet: Optional[Packet]) -> Optional[Packet]:
        return None

    async def maybe_fire(self, packet: Optional[Packet]) -> Optional[Packet]:
        rel = self._calculate_relevance(packet)
        if rel < self.sensitivity:
            return None
        try:
            return await self.process(packet)
        except Exception as e:
            log("NODE", f"{self.name} error: {e}", Term.RED)
            return None


class CognitiveMesh:
    """
    Distributed scheduler:
    - Pulls one salient workspace item (or None)
    - Lets nodes fire based on relevance
    - Posts node outputs back into workspace
    """
    def __init__(self, workspace: Any, genome: Any):
        self.workspace = workspace
        self.genome = genome
        self.nodes: List[CognitiveNode] = []
        self.last_tick = time.time()

        # recursion budget (prevents infinite loops)
        self.max_thought_depth = 5
        self.max_fires_per_tick = 6

    def add_node(self, node: CognitiveNode) -> None:
        self.nodes.append(node)

    def _workspace_to_packet(self, item: Optional[Tuple[float, str, Any]]) -> Optional[Packet]:
        if item is None:
            return None
        priority, source, content = item
        return {"type": "WORKSPACE", "source": source, "priority": priority, "content": content}

    def post(self, packet: Packet, priority: float = 5.0) -> None:
        # Use the packet's explicit 'source' if available, otherwise fallback to 'type'
        source = packet.get("source", packet.get("type", "PKT"))
        self.workspace.post(priority, source, packet)
        
    async def tick(self) -> None:
        item = self.workspace.get_most_salient()
        packet = self._workspace_to_packet(item)

        # --- FIX: Initialize variables to prevent UnboundLocalError ---
        bypass_filters = True  # Default to bypass if no packet (or silence)
        allowed_targets = []

        if packet:
            # 1. Identify the source of the signal
            # Use 'source' if available, otherwise fallback to 'type'
            source_module = packet.get("source", packet.get("type", "unknown")).lower()
            
            # 2. Get allowed connections from Genome
            allowed_targets = self.genome.architecture["connections"].get(source_module, [])
            
            # 3. Bypass filters for System/User inputs or if the source is unknown
            # (Prevents the agent from going mute if a module name doesn't match the genome exactly)
            bypass_filters = source_module in ["system", "user", "input", "executor", "network"]

        fired = 0
        for node in self.nodes:
            if fired >= self.max_fires_per_tick:
                break

            # Ask the node: "Is this relevant to you?"
            out = await node.maybe_fire(packet)
            
            if out is None:
                continue

            # Enforce recursion budget
            if out.get("type") == "THOUGHT":
                depth = int(out.get("depth", 0))
                if depth > self.max_thought_depth:
                    continue
            
            # --- LOGIC FIX: Self-Identification ---
            # Ensure the output packet knows who sent it (for the next cycle's topology check)
            if "source" not in out:
                out["source"] = node.name

            self.post(out, priority=float(out.get("priority", 5.0)))
            fired += 1

    async def run(self, cycle_dt: float = 0.02) -> None:
        while True:
            await self.tick()
            await asyncio.sleep(cycle_dt)

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
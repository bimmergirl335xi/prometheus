# main.py
import asyncio
import os
import time
import traceback
from datetime import datetime
import numpy as np

# --- Configuration & Utilities ---
from config import *
from utils import *
from definitions import *

# --- Core Systems ---
from genome import Genome
from health import CognitiveImmuneSystem
from memory import LongTermMemory, SelfModel
from core_systems import ( 
    PrincipleSystem, 
    SkillLibrary, 
    ActiveGlobalWorkspace, 
    PredictiveWorldModel
)

# --- Cognitive Engines ---
from abstraction import AbstractionEngine
from reasoning import ReasoningEngine
from language import LanguageEngine
from cognitive_bridge import CognitiveBridge

# --- Infrastructure ---
from infrastructure import CognitiveMesh, NeuralBus, NetworkInterface, InputSystem
from robot_interface import RobotInterface

# --- Distributed Nodes ---
from nodes import (
    ReasoningNode, 
    AbstractionNode, 
    MotorNode, 
    AttentionNode, 
    CuriosityNode, 
    TheoryOfMindNode,
    RecursiveThoughtNode
)

# --- Agents ---
from agents import (
     MetaCognition, 
     PerceptionAgent, 
     PlanningAgent, 
     ActionExecutor
 )

async def main():
    """
    PROMETHEUS v12.2 Entry Point
    Initializes the Cognitive Mesh and starts the primary event loop.
    """
    
    # 0. System Prep
    os.system('cls' if os.name == 'nt' else 'clear')
    with open(COMMS_FILE, "w") as f:
        f.write("--- PROMETHEUS UPLINK ESTABLISHED ---\n")

    print(f"{Term.BOLD}{Term.HEADER}PROMETHEUS v12.2 - Unified Mesh Core{Term.ENDC}")
    print(f"{Term.DIM}Initializing cognitive architecture...{Term.ENDC}\n")

    # 1. Infrastructure & Core Data Structures
    bus = NeuralBus()
    genome = Genome()
    # The unified workspace (Heap Queue + Broadcast)
    workspace = ActiveGlobalWorkspace(genome) 
    
    # 2. Knowledge Systems
    world_model = PredictiveWorldModel()
    self_model = SelfModel()
    principle_system = PrincipleSystem()
    immune_system = CognitiveImmuneSystem(genome)
    memory = LongTermMemory(genome)
    skill_library = SkillLibrary()
    
    # 3. Engines (Instantiated here to be shared across Nodes and Agents)
    abstraction_engine = AbstractionEngine(genome, memory)
    # Reasoning Engine is shared between Planner and ReasoningNode
    reasoning_engine = ReasoningEngine(memory, abstraction_engine) 
    
    # 4. Meta & Body
    meta = MetaCognition(genome, self_model, principle_system, immune_system)
    robot = RobotInterface(port=65432)
    bridge = CognitiveBridge(memory, robot, immune_system, meta)

    # 5. Agents (Functional Modules)
    
    # Perception: Monitors workspace and stimulates memory
    perception = PerceptionAgent(workspace, memory, abstraction_engine)
    
    # Planner: Needs World Model for simulation and Reasoning Engine for logic
    planner = PlanningAgent(
        genome, 
        memory, 
        self_model, 
        skill_library, 
        abstraction_engine,
        world_model 
    )
    planner.reasoner = reasoning_engine # Inject shared engine
    
    # Executor: Performs actions and posts results back to Workspace
    executor = ActionExecutor(
        genome=genome, 
        memory=memory, 
        bus=bus, 
        meta=meta, 
        skill_library=skill_library, 
        abstraction_engine=abstraction_engine,
        language_engine=perception.language_engine, 
        reasoning_engine=reasoning_engine,          
        immune_system=immune_system,
        bridge=bridge,
        workspace=workspace
    )

    # Register Engines to Bridge (for recursive callbacks)
    bridge.register_engines(
        abstract=abstraction_engine,
        reason=reasoning_engine,
        lang=perception.language_engine
    )

    # 6. Cognitive Mesh (The Scheduler)
    mesh = CognitiveMesh(workspace)
    
    # Add Distributed Processing Nodes
    # These nodes compete to process items in the Workspace
    mesh.add_node(AttentionNode(workspace))
    mesh.add_node(CuriosityNode(workspace))
    mesh.add_node(TheoryOfMindNode(workspace))
    mesh.add_node(ReasoningNode(workspace, reasoning_engine))
    mesh.add_node(AbstractionNode(workspace, abstraction_engine))
    mesh.add_node(MotorNode(workspace, bridge))
    mesh.add_node(RecursiveThoughtNode(workspace, reasoning_engine))

    # 7. IO Systems (Background Tasks)
    net = NetworkInterface(bus, workspace)
    inp = InputSystem(workspace)

    # Kickstart the Mind
    workspace.post(10.0, "SYSTEM", "System initialized. Waiting for input.", type="BOOT")

    # Start Background Services
    tasks = [
        asyncio.create_task(net.run(), name="network"),
        asyncio.create_task(inp.run(), name="input"),
        asyncio.create_task(workspace.decay_loop(), name="decay") # Active thought decay
    ]

    log("SYS", "🚀 Prometheus Online (Mesh + ActiveWorkspace)", Term.GREEN)
    log("SYS", f"🧠 Memory: {memory.semantic.next_idx} concepts | 📚 Skills: {len(skill_library.skills)}", Term.CYAN)

    step_count = 0
    last_status = time.time()
    last_save = time.time()

    try:
        while True:
            # --- PHASE A: SENSE & PERCEIVE ---
            # 1. Sync physical sensors (Robot -> Bridge -> Memory)
            bridge.synchronize()
            
            # 2. Perception Agent: Check Active Memory associations
            # (Does not consume workspace items, just adds context)
            perception.perceive() 

            # --- PHASE B: THINK (Distributed Mesh) ---
            # 3. The Mesh ticks: Nodes compete to process the highest priority item
            await mesh.tick()

            # --- PHASE C: DECIDE & ACT (Executive Function) ---
            # 4. Check if the most salient item implies an Executive Action
            # (We peek first, then pop if we handle it)
            top_item_tuple = workspace.get_most_salient()
            
            if top_item_tuple:
                p, source, content = top_item_tuple
                
                # Case 1: It's an explicit Action object (e.g., from Planner)
                if isinstance(content, Action):
                     goal = Goal(priority=abs(p), action=content)
                     await executor.execute(goal)
                
                # Case 2: It's a Dictionary instruction marked as ACTION (e.g., from Node)
                elif isinstance(content, dict) and content.get("type") == "ACTION":
                     # Convert to executable Action
                     cmd_content = content.get("content")
                     if cmd_content:
                        act = Action(ActionType.PROCESS_INPUT, cmd_content)
                        await executor.execute(Goal(priority=abs(p), action=act))
                
                # Case 3: Re-queue if not actionable by Executor (let Nodes handle it next tick)
                # Note: workspace.get_most_salient() POPs the item. 
                # If we didn't act on it, and it wasn't processed by mesh (mesh ticks before this),
                # it might be lost. 
                # However, Mesh.tick() *also* pops items. 
                # In this architecture, Mesh handles "Thinking" items, Executor handles "Doing" items.

            # --- PHASE D: MAINTENANCE ---
            await asyncio.sleep(genome.traits["cycle_speed"])
            step_count += 1
            workspace.prune_stale()

            # Periodic Status Report (Every 30s)
            if time.time() - last_status > 30:
                # Calculate simple health metric
                health_score = 1.0 - (sum(1 for m in immune_system.health_metrics.values() if m.critical) * 0.2)
                log("STATUS", 
                    f"Cycles: {step_count} | "
                    f"Focus: {workspace.current_broadcast.get('source', 'None') if workspace.current_broadcast else 'Drifting'} | "
                    f"Health: {health_score:.2f}", 
                    Term.CYAN)
                last_status = time.time()

            # Periodic Auto-Save (Every 60s)
            if time.time() - last_save > 60:
                memory.save()
                genome.save()
                skill_library.save()
                last_save = time.time()

    except KeyboardInterrupt:
        print(f"\n{Term.YELLOW}Shutdown initiated...{Term.ENDC}")
        
    except Exception as e:
        log("CRITICAL", f"System Crash: {e}", Term.RED)
        traceback.print_exc()

    finally:
        # Graceful Shutdown
        log("SYS", "💾 Emergency Save...", Term.YELLOW)
        for task in tasks: task.cancel()
        
        genome.save()
        memory.save()
        skill_library.save()
        immune_system.save()
        
        log("SYS", "✅ System Halted.", Term.RED)

if __name__ == "__main__":
    asyncio.run(main())
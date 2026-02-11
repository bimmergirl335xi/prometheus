# main.py
from config import *
from utils import *
from definitions import *

from health import CognitiveImmuneSystem
from memory import LongTermMemory, SelfModel
from executive import ExecutivePlanner

from abstraction import AbstractionEngine
from reasoning import ReasoningEngine
from language import LanguageEngine, UniversalGrammar
from cognitive_bridge import CognitiveBridge

from robot_interface import RobotInterface

from core_systems import (
    Genome,
    PrincipleSystem,
    SkillLibrary,
)

from infrastructure import (
    NeuralBus,
    NetworkInterface,
    InputSystem
)

from agents import (
     GlobalWorkspace, 
     MetaCognition, 
     PerceptionAgent, 
     PlanningAgent, 
     ActionExecutor
 )

async def main():
    """Main initialization and event loop for Proto-AGI"""

    # Clear screen
    os.system('cls' if os.name == 'nt' else 'clear')

    # [Add this near the top of main()]
    with open(COMMS_FILE, "w") as f:
        f.write("--- PROMETHEUS UPLINK ESTABLISHED ---\n")

    # Banner
    print(f"{Term.BOLD}{Term.HEADER}{'='*80}{Term.ENDC}")
    print(f"{Term.BOLD}{Term.HEADER}PROMETHEUS v12.0 - Proto-AGI Core{Term.ENDC}")
    print(f"{Term.BOLD}{Term.HEADER}{'='*80}{Term.ENDC}\n")

    print(f"{Term.GREEN}Philosophy:{Term.ENDC}")
    print(f"  • A mind without knowledge - pure cognitive architecture")
    print(f"  • Behavior emerges from principles, not hardcoded rules")
    print(f"  • Self-modifying architecture with safety bounds")
    print(f"  • Unbounded capability growth through skill acquisition\n")

    print(f"{Term.YELLOW}Core Systems:{Term.ENDC}")
    print(f"  • Principle-based cognition (intrinsic motivation)")
    print(f"  • Dynamic skill library (learn & compose capabilities)")
    print(f"  • Abstraction engine (form own concepts)")
    print(f"  • Cognitive immune system (self-repair)")
    print(f"  • Fluid genome (architectural plasticity)")
    print(f"  • GPU acceleration: {Term.CYAN}{GPU_AVAILABLE}{Term.ENDC}\n")

    print(f"{Term.ORANGE}New Capabilities:{Term.ENDC}")
    print(f"  • {Term.PURPLE}BUILD_SKILL <name>{Term.ENDC} → create new capability")
    print(f"  • {Term.RED}SYS_MOD: param = value{Term.ENDC} → modify traits")
    print(f"  • {Term.CYAN}CHECKPOINT <name>{Term.ENDC} → save state for rollback")
    print(f"  • Watch it develop skills and abstractions autonomously\n")

    print(f"{Term.DIM}Press Ctrl+C to save and exit{Term.ENDC}\n")

    print(f"{Term.CYAN}Communication Channel: {os.path.abspath(COMMS_FILE)}{Term.ENDC}\n")

    # Initialize core systems
    bus = NeuralBus()
    genome = Genome()
    workspace = GlobalWorkspace(genome) # MISSING CLASS
    self_model = SelfModel()
    principle_system = PrincipleSystem()
    immune_system = CognitiveImmuneSystem(genome)
    memory = LongTermMemory(genome)
    skill_library = SkillLibrary()
    abstraction_engine = AbstractionEngine(genome, memory)

    # Initialize meta-cognition
    meta = MetaCognition(genome, self_model, principle_system, immune_system) # MISSING CLASS

    # 1. Init Robot Interface (The Body)
    robot = RobotInterface(port=65432)

    # 2. Init The Bridge (Must happen before agents)
    bridge = CognitiveBridge(memory, robot, immune_system, meta)

    # Initialize agents
    perception = PerceptionAgent(workspace, memory, abstraction_engine) # MISSING CLASS
    
    planner = PlanningAgent(genome, memory, self_model, skill_library, abstraction_engine) # MISSING CLASS
    
    executor = ActionExecutor(
        genome=genome, 
        memory=memory, 
        bus=bus, 
        meta=meta, 
        skill_library=skill_library, 
        abstraction_engine=abstraction_engine,
        language_engine=perception.language_engine, 
        reasoning_engine=planner.reasoner,          
        immune_system=immune_system,
        bridge=bridge
    )

    bridge.register_engines(
        abstract=abstraction_engine,
        reason=planner.reasoner,
        lang=perception.language_engine
    )

    executive = ExecutivePlanner(
        perception=perception,
        planner=planner,
        executor=executor,
        genome=genome,
        meta=meta,
        self_model=self_model,
        principle_system=principle_system,
        immune_system=immune_system,
        workspace=workspace,
        abstraction_engine=abstraction_engine 
    )

    # Initialize infrastructure
    net = NetworkInterface(bus, workspace)
    inp = InputSystem(workspace)

    # Start background services
    tasks = [
        asyncio.create_task(net.run(), name="network"),
        asyncio.create_task(inp.run(), name="input")
    ]

    try:
        log("SYS", "🚀 Prometheus v12.0 Proto-AGI Online", Term.GREEN)
        log("SYS", f"💾 Memory: {memory.semantic.next_idx} concepts loaded", Term.CYAN)
        log("SYS", f"📚 Skills: {len(skill_library.skills)} loaded", Term.CYAN)
        log("SYS", f"💡 Abstractions: {len(abstraction_engine.abstractions)} loaded", Term.CYAN)
        log("SYS", f"🏥 Health: Immune system active", Term.CYAN)
        log("SYS", f"🧬 Genome: {len(genome.checkpoints)} checkpoints", Term.CYAN)
        
        # Initialize tracking
        executive.concepts_at_start = memory.semantic.next_idx
        executive.add_goal(Action(ActionType.PROCESS_INPUT, "System initialized. Hello World."), priority=5.0)

        step_count = 0
        last_status = time.time()
        last_save = time.time()
        
        while True:
             bridge.synchronize()
            # Main cognitive cycle
             await executive.step()
            
            # Periodic workspace maintenance
             if step_count % 100 == 0:
                 workspace.prune_stale(max_age=30.0)
            
            # Periodic status report (every 30 seconds)
                 current_time = time.time()
             if current_time - last_status > 30 and step_count > 0:
                    status = meta.get_status_report()
                    log("STATUS", 
                        f"Steps: {step_count} | " +
                        f"SR: {status['success_rate']:.1%} | " +
                        f"IR: {status['intrinsic_reward']:.2f} | " +
                        f"Queue: {len(executive.goal_queue)} | " +
                        f"Concepts: {memory.semantic.next_idx} | " +
                        f"Skills: {len(skill_library.skills)} | " +
                        f"Abs: {len(abstraction_engine.abstractions)}",
                        Term.CYAN)
                    
                    # Health status
                    critical = sum(1 for s in immune_system.health_metrics.values() if s.critical)
                    if critical > 0:
                        log("STATUS", f"⚠️  Health: {critical} critical metrics", Term.YELLOW)
                    
                    last_status = current_time
                
                    step_count += 1
             
             current_time = time.time()
             if current_time - last_save > 60:  # Save every 60 seconds
                log("SYS", "💾 Auto-saving memory...", Term.DIM)
                memory.save()
                genome.save()
                skill_library.save()
                immune_system.save()
                last_save = current_time
            
            # Adaptive cycle speed
             await asyncio.sleep(genome.traits["cycle_speed"])

    except KeyboardInterrupt:
        print(f"\n{Term.YELLOW}Shutdown initiated...{Term.ENDC}\n")
        
        # Cancel background tasks
        for task in tasks:
            task.cancel()
        
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Save state
        log("SYS", "💾 Saving state...", Term.YELLOW)
        genome.save()
        memory.save()
        skill_library.save()
        immune_system.save()
        
        # Final report
        # status = meta.get_status_report()
        introspection = self_model.introspect()
        
        print(f"\n{Term.CYAN}{'='*80}{Term.ENDC}")
        print(f"{Term.BOLD}Final Status Report{Term.ENDC}")
        print(f"{Term.CYAN}{'='*80}{Term.ENDC}\n")
        
        print(f"{Term.GREEN}Performance:{Term.ENDC}")
        # print(f"  Success Rate: {status['success_rate']:.1%}")
        # print(f"  Confidence: {meta.confidence:.2f}")
        # print(f"  Intrinsic Reward: {status['intrinsic_reward']:.2f}")
        print(f"  Total Actions: {introspection['total_actions']}")
        
        print(f"\n{Term.YELLOW}Growth:{Term.ENDC}")
        print(f"  Concepts Learned: {memory.semantic.next_idx}")
        print(f"  Skills Acquired: {len(skill_library.skills)}")
        print(f"  Abstractions Formed: {len(abstraction_engine.abstractions)}")
        print(f"  Episodes Stored: {len(memory.episodic.episodes)}")
        
        print(f"\n{Term.PURPLE}Adaptations:{Term.ENDC}")
        print(f"  Total Adaptations: {status['adaptations']}")
        print(f"  Interventions: {status['interventions']}")
        print(f"  Architecture Mods: {len([m for m in genome.modification_history if m.get('type') == 'architecture'])}")
        
        print(f"\n{Term.CYAN}Health:{Term.ENDC}")
        critical_count = sum(1 for s in immune_system.health_metrics.values() if s.critical)
        print(f"  Critical Metrics: {critical_count}")
        print(f"  Repairs Applied: {len(immune_system.repair_history)}")
        
        print(f"\n{Term.RED}Principle Satisfaction:{Term.ENDC}")
        for principle_name, satisfaction in status.get('principle_satisfaction', {}).items():
            bar_length = int(satisfaction * 20)
            bar = '█' * bar_length + '░' * (20 - bar_length)
            print(f"  {principle_name:20s} {bar} {satisfaction:.2f}")
        
        print(f"\n{Term.GREEN}Checkpoints: {len(genome.checkpoints)}{Term.ENDC}")
        for checkpoint in genome.checkpoints[-3:]:
            ts = datetime.fromtimestamp(checkpoint['timestamp']).strftime("%H:%M:%S")
            print(f"  • {checkpoint['id']} ({ts})")
        
        log("SYS", "👋 Shutdown complete. Proto-AGI sleeping.", Term.GREEN)

    except Exception as e:
        log("SYS", f"CRITICAL CRASH: {e}", Term.RED)
        traceback.print_exc()

    finally:
        # This block runs NO MATTER WHAT (Crash, Ctrl+C, Error)
        print(f"\n{Term.CYAN}{'='*80}{Term.ENDC}")
        log("SYS", "💾 SAVING STATE (DO NOT CLOSE)...", Term.YELLOW)
        
        # Cancel background tasks
        for task in tasks:
            task.cancel()
        
        # Force Save
        genome.save()
        memory.save()
        skill_library.save()
        immune_system.save()
        
        log("SYS", "✅ Data Saved. Goodnight.", Term.GREEN)
        print(f"{Term.CYAN}{'='*80}{Term.ENDC}\n")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
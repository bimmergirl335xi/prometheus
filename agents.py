# agents.py
import asyncio
import time
import heapq
import random
import ast
import os
import re
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import deque, defaultdict
import numpy as np

# Import Engines
from language import LanguageEngine, UniversalGrammar
from abstraction import AbstractionEngine
from reasoning import ReasoningEngine
from cognitive_bridge import CognitiveBridge

from config import *
from utils import speak, log, Term
from definitions import *
from genome import Genome
from core_systems import PrincipleSystem, SkillLibrary, ActiveGlobalWorkspace
from memory import LongTermMemory, SelfModel, Episode, WorkingMemory
from health import CognitiveImmuneSystem
from infrastructure import NeuralBus

class MetaCognition:
    """
    The 'Observer' module. Monitors the agent's internal state,
    detects stalls/loops, and manages confidence/frustration.
    """
    def __init__(self, 
                 genome: Genome, 
                 self_model: SelfModel, 
                 principle_system: PrincipleSystem,
                 immune_system: CognitiveImmuneSystem):
        self.genome = genome
        self.self_model = self_model
        self.principle_system = principle_system
        self.immune_system = immune_system
        
        self.confidence = 0.5
        self.frustration = 0.0
        self.success_history = deque(maxlen=50)
        self.events: List[str] = []

    def observe(self, event_type: str, goal: Optional['Goal'] = None) -> None:
        """Register a significant cognitive event"""
        self.events.append(event_type)
        
        if event_type == "SUCCESS":
            self.success_history.append(1.0)
            self.confidence = min(1.0, self.confidence + 0.05)
            self.frustration = max(0.0, self.frustration - 0.2)
        
        elif event_type == "FAILURE":
            self.success_history.append(0.0)
            # Confidence takes a bigger hit from failure than gain from success (conservative)
            self.confidence = max(0.1, self.confidence - 0.1)
            self.frustration = min(1.0, self.frustration + 0.15)
        
        elif event_type == "STALL":
            self.frustration = min(1.0, self.frustration + 0.05)
            log("META", "⚠️  Cognitive Stall Detected", Term.YELLOW)
        
        elif event_type == "OVERLOAD":
            self.frustration = min(1.0, self.frustration + 0.1)
        
        elif event_type == "BREAKTHROUGH":
            self.confidence = min(1.0, self.confidence + 0.2)
            self.frustration = 0.0
            log("META", "🌟 BREAKTHROUGH REGISTERED", Term.PURPLE)

    def get_success_rate(self) -> float:
        if not self.success_history:
            return 0.5
        return sum(self.success_history) / len(self.success_history)

    def get_status_report(self) -> Dict[str, Any]:
        """Generate a high-level status report"""
        return {
            "confidence": self.confidence,
            "frustration": self.frustration,
            "success_rate": self.get_success_rate(),
            "intrinsic_reward": self.principle_system.get_intrinsic_reward(),
            "adaptations": len(self.genome.modification_history),
            "interventions": len(self.immune_system.repair_history),
            "principle_satisfaction": {p.name: obj.satisfaction for p, obj in self.principle_system.principles.items()}
        }

# Define MetaEvent constants for consistency
class MetaEvent:
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    STALL = "STALL"
    OVERLOAD = "OVERLOAD"
    BREAKTHROUGH = "BREAKTHROUGH"

class PerceptionAgent:
    """
    Processes raw input from the Workspace and converts it into actionable signals.
    """
    def __init__(self, 
                 workspace: ActiveGlobalWorkspace, 
                 memory: LongTermMemory,
                 abstraction_engine: AbstractionEngine):
        self.workspace = workspace
        self.memory = memory
        self.abstraction_engine = abstraction_engine
        self.language_engine = LanguageEngine(memory)
        self.universal_grammar = UniversalGrammar(memory)

    def perceive(self) -> Optional[Tuple[float, str, Any]]:
        """Check workspace for new salient information"""
        
        # Run the stream of consciousness
        self.associate_stream()
        
        return self.workspace.get_most_salient()

    def associate_stream(self):
        """
        The 'Stream of Consciousness'. 
        Uses items in Working Memory to pull related LTM concepts back into WM.
        """
        # Get current focus items
        active_items = self.workspace.working_memory.retrieve()
        
        for item in active_items:
            if isinstance(item, str):
                # 1. Check direct associations
                related = self.memory.semantic.get_associated(item, top_k=2)
                
                # 2. Check semantic similarity (The vector improvement)
                # We interpret similar vectors as "reminders"
                
                for concept, weight in related:
                    # Chance to bubble up into working memory
                    if weight > 0.5 and random.random() < 0.3:
                        # "Reminded" of something
                        log("MEM", f"💭 Association: {item} -> {concept}", Term.DIM)
                        self.workspace.working_memory.hold(concept, "associative")

    def interpret(self, signal: Tuple[float, str, Any]) -> Optional[Action]:
        """Convert a raw signal into a cognitive action"""
        priority, source, content = signal
        
        if source == "USER":
            text = str(content)
            
            # LEARN GRAMMAR: Every user input is a lesson in English
            self.language_engine.learn_grammar(text)

        # Stimulate memory with keywords from content
        if isinstance(content, str):
            words = content.split()
            for word in words:
                if len(word) > 3:
                    self.memory.semantic.stimulate(word, strength=0.2)

        # 1. User Commands
        if source == "USER":
            text = str(content)

            if isinstance(content, dict) and content.get("type") == "file":
                return Action(ActionType.READ_FILE, content["path"], metadata={"source": "user"})
            
            text = str(content)

        if source == "USER":
            text = str(content)
            
            # Instead of just splitting strings, we understand the EVENT.
            structural_parse = self.universal_grammar.parse_universal(text)
            
            if structural_parse.get("action"):
                log("LANG", f"Parsed Event: {structural_parse}", Term.GREEN)

            # System Modification
            if text.startswith("SYS_MOD"):
                try:
                    # Parse "SYS_MOD: trait = value"
                    _, assignment = text.split(":", 1)
                    trait, value = assignment.split("=")
                    trait = trait.strip()
                    value = float(value.strip())
                    return Action(ActionType.SELF_MODIFY, f"{trait}={value}", metadata={"source": "user"})
                except ValueError:
                    log("PERCEPT", "Invalid syntax for SYS_MOD", Term.RED)
                    return None
                pass

            # Checkpoint
            elif text.startswith("CHECKPOINT"):
                name = text.split(" ", 1)[1] if " " in text else None
                return Action(ActionType.CHECKPOINT, name)

            # Build Skill
            elif text.startswith("BUILD_SKILL"):
                skill_name = text.split(" ", 1)[1] if " " in text else "new_skill"
                return Action(ActionType.BUILD_SKILL, skill_name)

            # General Inquiry / Task
            else:
                return Action(ActionType.PROCESS_INPUT, text, metadata={"source": "user"})

        # 2. Network Results
        elif source == "NETWORK":
            return Action(ActionType.LEARN, content, metadata={"source": "network"})

        return None
    
    def extract_skill_from_text(self, text: str) -> Optional[Action]:
        """
        Social Learning: Scans text for instructional patterns 
        (e.g., "To do X: Step 1, Step 2...") and compiles a Skill.
        """
        # Regex to find "To [SkillName]: [Steps]"
        pattern = r"To\s+(\w+):(.+)"
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            skill_name = match.group(1).strip()
            steps_text = match.group(2).strip()
            
            # Simple parser: split by commas or numbers
            steps = [s.strip() for s in re.split(r'[,;]|\d+\.', steps_text) if len(s) > 3]
            
            if len(steps) > 1:
                # We construct a new skill programmatically
                code_block = "def execute(context):\n"
                for step in steps:
                    code_block += f"    print('Executing step: {step}')\n"
                    # In a real scenario, we would map 'step' words to internal functions
                
                log("SOCIAL", f"🧠 Learned skill '{skill_name}' from text!", Term.GREEN)
                
                # Return an action to formally build this skill
                return Action(
                    ActionType.BUILD_SKILL, 
                    payload=skill_name
                )
        return None

class PlanningAgent:
    """
    Responsible for breaking down high-level goals into executable steps.
    Uses the causal model to predict outcomes.
    """
    def __init__(self, 
                 genome: Genome, 
                 memory: LongTermMemory, 
                 self_model: SelfModel, 
                 skill_library: SkillLibrary,
                 abstraction_engine: AbstractionEngine,
                 world_model: 'PredictiveWorldModel'):
        self.genome = genome
        self.memory = memory
        self.self_model = self_model
        self.skill_library = skill_library
        self.abstraction_engine = abstraction_engine
        self.world_model = world_model

        self.reasoner = ReasoningEngine(memory, abstraction_engine)

    async def generate_plan_tree(self, goal: Goal, depth: int = 3, breadth: int = 3) -> List[Action]:
        """
        Tree of Thoughts (ToT) Search.
        Explores future states to find the optimal path.
        """
        log("PLAN", f"🌳 Growing thought tree for: {goal.action.action_type.name}", Term.CYAN)
        
        # State = (Accumulated Actions, Current Context Score)
        # Queue = List of paths
        queue = [([], 0.0)] 
        
        best_path = []
        best_score = -float('inf')

        current_context_vec = self.executor.executive._get_state_vector() # Access executive state

        for d in range(depth):
            candidates = []
            
            # 1. Expand (Breadth)
            while queue:
                path, score = queue.pop(0)
                
                # Generate 'breadth' number of next logical steps
                possible_next_actions = self._propose_next_actions(path, breadth)
                
                for action in possible_next_actions:
                    # 2. Evaluate (Simulate)
                    # Use World Model to imagine the result of this specific action sequence
                    predicted_state_vec = self.executor.executive.world_model.imagine_sequence(
                        current_context_vec, 
                        path + [action]
                    )
                    
                    # Score the state against the Goal (Similarity check)
                    # (Assuming goal has a target vector representation)
                    state_val = self._evaluate_state(predicted_state_vec, goal)
                    
                    candidates.append((path + [action], state_val))

            # 3. Prune
            # Keep only the top 'breadth' paths for the next layer
            candidates.sort(key=lambda x: x[1], reverse=True)
            queue = candidates[:breadth]
            
            if queue and queue[0][1] > best_score:
                best_score = queue[0][1]
                best_path = queue[0][0]

        log("PLAN", f"✅ ToT Complete. Best path length: {len(best_path)} Score: {best_score:.2f}", Term.GREEN)
        return best_path

    def _propose_next_actions(self, current_path: List[Action], n: int) -> List[Action]:
        """Uses Skill Library + Randomness to propose valid next moves."""
        # Simple heuristic: What skills fit the current context?
        valid_skills = self.skill_library.find_applicable_skills(self.memory.get_context()) # Simplified
        
        actions = []
        for _ in range(n):
            if valid_skills:
                skill = random.choice(valid_skills)
                actions.append(Action(ActionType.EXECUTE_SKILL, skill.name))
            else:
                # Fallback to generic actions
                actions.append(Action(ActionType.RESEARCH, "fallback_strategy"))
        return actions

    def predict_success(self, goal: 'Goal') -> float:
        """Predict the likelihood of a goal succeeding based on history"""
        # Base prediction from causal model
        causal_pred = self.memory.causal.predict_success(
            goal.action.action_type, 
            goal.context
        )
        
        # Adjust based on self-confidence in that capability
        self_conf = self.self_model.get_confidence(goal.action.action_type)
        
        # Weighted average
        return (causal_pred * 0.4) + (self_conf * 0.6)

    # [Add to PlanningAgent in agents.py]
    def recall_solution(self, goal_action_type: ActionType, context_keys: List[str]) -> Optional[str]:
        """
        Find a past episode with the same action and similar context that succeeded.
        Returns the specific payload used previously.
        """
        # Retrieve successful episodes of this type
        successes = self.memory.episodic.get_by_action(goal_action_type)
        successes = [e for e in successes if e.outcome]
        
        best_match = None
        best_score = 0.0
        
        for ep in successes:
            # Simple context overlap score
            match_count = sum(1 for k in context_keys if k in ep.context)
            if match_count > best_score:
                best_score = match_count
                best_match = ep
        
        if best_match and best_score > 0:
            log("PLAN", f"💡 Recalled solution from {best_match.age():.1f}s ago", Term.CYAN)
            return best_match.action.payload
            
        return None

    def should_decompose(self, goal: 'Goal') -> bool:
        """Decide if a goal is too complex and needs breakdown"""
        if goal.action.action_type in [ActionType.PROCESS_INPUT, ActionType.RESEARCH]:
            # Complex tasks often need decomposition
            return self.predict_success(goal) < self.genome.traits["confidence_threshold"]
        return False

    def decompose_goal(self, goal: 'Goal') -> List['Goal']:
        """Break a goal into subgoals"""
        subgoals = []
        parent_prio = goal.priority
        
        if goal.action.action_type == ActionType.PROCESS_INPUT:
            # Standard problem solving strategy
            # 1. Research/Understand
            # 2. Hypothesize/Plan
            # 3. Execute/Respond
            
            subgoals.append(Goal(
                priority=parent_prio + 0.1, 
                action=Action(ActionType.RESEARCH, goal.action.payload)
            ))
            
            # If we have relevant skills, use them
            best_skill = self.skill_library.get_best_skill(goal.context)
            if best_skill:
                # We interpret "using a skill" as executing its code/steps
                # For now, we wrap it in a generic execution action
                subgoals.append(Goal(
                    priority=parent_prio,
                    action=Action(ActionType.EXECUTION_ERROR, best_skill.name) # Placeholder logic
                ))
                pass

            if not subgoals and not best_skill:
                log("PLAN", "🤔 No standard skill found. Attempting reasoning...", Term.PURPLE)
            
            # A. Try Analogy (Solve based on similar past experience)
            analogous_action = self.reasoner.solve_by_analogy(goal.context)
            if analogous_action:
                log("PLAN", f"💡 Found analogy! Adapting strategy.", Term.CYAN)
                subgoals.append(Goal(
                    priority=parent_prio,
                    action=analogous_action,
                    context=goal.context
                ))
            
            # B. Try Deduction (If the goal is to find a relation)
            elif "relation" in goal.context:
                target = goal.context.get("target")
                source = goal.context.get("source")
                if target and source:
                    score, path = self.reasoner.deduce_relation(source, target)
                    if score > 0.1:
                        log("PLAN", f"🔗 Deduced path: {' -> '.join(path)}", Term.CYAN)
                        # Create a plan to verify this deduction
                        subgoals.append(Goal(
                            priority=parent_prio,
                            action=Action(ActionType.RESEARCH, f"Verify connection: {path}"),
                            context=goal.context
                        ))
            
        return subgoals

    def generate_autonomous_goals(self) -> List['Goal']:
        """
        Generate goals based on intrinsic drives.
        Now includes MOTOR BABBLING: Randomly trying tools to see what they do.
        """
        goals = []
        
        # 1. Motor Babbling (The Infant Phase)
        # If curiosity is high, try a random action with a random concept
        if self.genome.values["curiosity"] > 0.3:
            # Pick a random available action type
            # We filter out internal-only actions to focus on external effects
            options = [
                ActionType.SPEAK, 
                ActionType.RESEARCH, 
                ActionType.READ_FILE, 
                ActionType.EXPLORE
            ]
            action_type = random.choice(options)
            
            # Pick a random concept to use as the "Payload"
            payload = "init"
            if self.memory.semantic.idx_to_concept:
                idx = random.choice(list(self.memory.semantic.idx_to_concept.keys()))
                payload = self.memory.semantic.idx_to_concept[idx]
            
            # Create a "Just try it" goal
            # We give it slightly lower priority than survival, but high enough to happen often
            log("PLAN", f"👶 Motor Babbling: Trying {action_type.name} with '{payload}'", Term.DIM)
            goals.append(Goal(
                priority=2.0, 
                action=Action(action_type, payload, emergent=True)
            ))

        # 2. Memory Consolidation (Maintenance)
        if self.memory.semantic.next_idx > 100 and random.random() < 0.1:
             goals.append(Goal(priority=1.5, action=Action(ActionType.CONSOLIDATE, None, emergent=True)))

        # 3. Reflection (Self-Analysis)
        if random.random() < 0.2:
            goals.append(Goal(priority=1.0, action=Action(ActionType.REFLECT, "idle_reflection", emergent=True)))

        return goals

    def simulate(self, action: Action) -> Tuple[bool, str]:
        """
        Run a mental simulation of an action to check for safety/outcome.
        Returns (success, message).
        """
        # Safety check for self-modification
        if action.action_type == ActionType.SELF_MODIFY:
            # Check if modification is within bounds (logic mirrored from Genome)
            if "=" in str(action.payload):
                try:
                    trait, val = str(action.payload).split("=")
                    trait = trait.strip()
                    val = float(val)
                    bounds = self.genome._get_safety_bounds(trait)
                    if not (bounds[0] <= val <= bounds[1]):
                        return False, f"Simulation: Value {val} for {trait} out of bounds {bounds}"
                except:
                    pass
        
        return True, "Simulation safe"

    def recall_solution(self, goal_action_type: ActionType, context_keys: List[str]) -> Optional[str]:
        """
        Find a past episode with the same action and similar context that succeeded.
        Returns the specific payload used previously.
        """
        # Retrieve successful episodes of this type
        successes = self.memory.episodic.get_by_action(goal_action_type)
        successes = [e for e in successes if e.outcome]
        
        best_match = None
        best_score = 0.0
        
        for ep in successes:
            # simple context overlap score
            match_count = sum(1 for k in context_keys if k in ep.context)
            if match_count > best_score:
                best_score = match_count
                best_match = ep
        
        if best_match and best_score > 0:
            log("PLAN", f"💡 Recalled solution from {best_match.age():.1f}s ago", Term.CYAN)
            return best_match.action.payload
            
        return None

    def find_transferable_skill(self, context: Dict[str, Any]) -> Optional[str]:
        """
        Generalization: If no exact skill exists, find a skill 
        used in a semantically similar context.
        """
        # Get vector for current context keys
        current_concepts = list(context.keys())
        
        best_skill = None
        best_similarity = 0.0
        
        # Iterate through all known skills
        for name, skill in self.skill_library.skills.items():
            # Check skill's 'preconditions' (concepts it usually works with)
            for pre in skill.preconditions:
                # Check similarity against current context
                for curr in current_concepts:
                    sim = self.memory.semantic.get_similarity(pre, curr)
                    
                    if sim > 0.7 and sim > best_similarity:
                        best_similarity = sim
                        best_skill = name
                        
        if best_skill and best_similarity > 0.7:
            log("PLAN", f"💡 Generalization: Transferring '{best_skill}' to new domain (Sim: {best_similarity:.2f})", Term.CYAN)
            return best_skill
            
        return None

    def select_best_action(self, goal: 'Goal', candidates: List[Action]) -> Action:
        """
        Model-Based Reinforcement Learning (MBRL).
        Simulates candidates and picks the one leading to the best 'Principle Satisfaction'.
        """
        current_state = self.executor.executive._get_state_vector() # Access state
        best_action = None
        best_score = -float('inf')

        for action in candidates:
            # Vectorize the action for the neural net
            action_vec = self.executor.executive._vectorize_action(action)
            
            # 1. IMAGINE: Rollout the future 3 steps deep
            trajectory = self.executor.executive.world_model.imagine(
                current_state, action_vec, steps=3
            )
            
            # 2. EVALUATE: How good is this future?
            # We project the imagined state vector back to Principle Scores
            # (Requires mapping vector -> satisfaction, strictly speaking)
            
            predicted_value = 0.0
            cumulative_uncertainty = 0.0
            
            for step in trajectory:
                # Intrinsic Motivation: We like low uncertainty (Knowledge)
                # UNLESS we are bored (High curiosity drive)
                curiosity_drive = self.genome.values["curiosity"]
                
                if curiosity_drive > 0.7:
                    # Reward high uncertainty (Exploration)
                    predicted_value += step["uncertainty"] * 2.0
                else:
                    # Punish high uncertainty (Safety)
                    predicted_value -= step["uncertainty"]
                
                cumulative_uncertainty += step["uncertainty"]

            # Heuristic: If we are too unsure, don't do it (unless exploring)
            if cumulative_uncertainty > 2.0 and curiosity_drive < 0.5:
                continue

            if predicted_value > best_score:
                best_score = predicted_value
                best_action = action
                
        return best_action or candidates[0]

class ActionExecutor:
    """
    Executes the primitive actions. 
    Interacts with the world, the bus, and internal state.
    """
    def __init__(self, 
        genome: Genome, 
        memory: LongTermMemory, 
        bus: NeuralBus, 
        meta: MetaCognition, 
        skill_library: SkillLibrary,
        abstraction_engine: AbstractionEngine,
        language_engine: LanguageEngine,
        reasoning_engine: ReasoningEngine,
        bridge: CognitiveBridge,
        immune_system: CognitiveImmuneSystem,
        workspace: ActiveGlobalWorkspace): # Added workspace
        self.genome = genome
        self.memory = memory
        self.bus = bus
        self.meta = meta
        self.skill_library = skill_library
        self.abstraction_engine = abstraction_engine
        self.language_engine = language_engine
        self.reasoning_engine = reasoning_engine
        self.immune_system = immune_system
        self.bridge = bridge
        self.workspace = workspace # Store it

        # Simple drive state
        self.drives = {
            "CURIOSITY": 1.0,
            "ENERGY": 1.0
        }

    async def execute(self, goal: Goal) -> Tuple[bool, Optional[FailureType]]:
        """
        Execute the action contained in the goal.
        Returns (success, failure_reason).
        """
        action = goal.action
        start_time = time.time()
        
        # DEBUG: Confirm we received the action
        log("DEBUG", f"Attempting to execute: {action.action_type.name}", Term.DIM)
        
        try:
            # --- ACTION HANDLERS ---
            
            if action.action_type == ActionType.PROCESS_INPUT:
                response = f"Processed: {action.payload}"
                log("ACT", response, Term.GREEN)
                return True, None

            elif action.action_type == ActionType.RECURSIVE_THINK:
                # Hand off the logic to the Bridge
                context = action.payload if isinstance(action.payload, dict) else {}
                depth = context.get("depth", 0)
                
                success, next_action = self.bridge.recursive_thought_loop(context, depth)
                
                if success and next_action:
                    # REPLACED: self.executive.add_goal(...)
                    # WITH:
                    self.workspace.post(
                        priority=goal.priority + 0.1, 
                        source="EXECUTOR", 
                        content=next_action
                    )
                    return True, None
                
            elif action.action_type == ActionType.MOTOR_COMMAND:
                # Parse "MOVE_FORWARD 10"
                try:
                    cmd, val = str(action.payload).split(" ", 1)
                    self.bridge.dispatch_motor_command(cmd, val)
                    return True, None
                except:
                    return False, FailureType.INVALID_INPUT

            elif action.action_type == ActionType.RESEARCH:
                query = str(action.payload)
                log("ACT", f"Researching: {query}", Term.BLUE)
                await self.bus.publish("network", query)
                return True, None      

            elif action.action_type == ActionType.READ_FILE:
                filepath = str(action.payload)
                log("ACT", f"📖 Opening file: {filepath}", Term.BLUE)
                
                # Check for "Deep Mode" flag (set by Executive or User)
                is_deep_mode = action.metadata.get("mode") == "DEEP"
                
                if not os.path.exists(filepath):
                    return False, FailureType.INVALID_INPUT
                
                content_text = ""
                try:
                    # --- LOAD CONTENT ---
                    # (Keep your existing PDF/Text loading logic here)
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content_text = f.read()

                    # --- PROCESSING ---
                    sentences = re.split(r'(?<=[.!?])\s+', content_text)
                    total_sentences = len(sentences)
                    
                    log("ACT", f"Processing {total_sentences} sentences (Mode: {'DEEP' if is_deep_mode else 'SKIM'})...", Term.CYAN)

                    for i, sentence in enumerate(sentences):
                        words = [w.strip() for w in sentence.split() if len(w) > 2]
                        if not words: continue

                        # 1. Metrics: How hard is this sentence?
                        complexity = self.language_engine.calculate_complexity(sentence)
                        
                        # 2. Dynamic Mode Switching
                        # If complexity spikes, auto-trigger Deep Mode for this sentence
                        if complexity > 0.8 and not is_deep_mode:
                            log("META", f"⚠️  Spike in complexity ({complexity:.2f}). slowing down.", Term.YELLOW)
                            is_deep_mode = True
                        elif complexity < 0.3 and is_deep_mode:
                            is_deep_mode = False

                        # 3. Learning (Standard)
                        # (Keep your existing stimulate/learn_sequence/learn_context calls here)
                        for word in words:
                             self.memory.semantic.stimulate(word, strength=0.1)
                        if hasattr(self.memory.semantic, 'learn_sequence'):
                             self.memory.semantic.learn_sequence(words)

                        # 4. DEEP PROCESSING LOOP
                        if is_deep_mode:
                            # A. Logic Check
                            plausible, msg = self.reasoning_engine.validate_proposition(sentence)
                            if not plausible:
                                log("REASON", f"🤔 Doubt: {msg}", Term.ORANGE)
                                # Trigger a strong memory update to flag this conflict
                                self.memory.semantic.stimulate("CONFLICT", strength=0.5)

                            # B. Abstraction Check
                            # Does this sentence match a known Rule?
                            # We create a pseudo-context from the sentence words
                            # (In a real system, we'd extract key-values. Here we treat words as keys)
                            sent_context = {w: 1 for w in words}
                            for name in self.abstraction_engine.abstractions:
                                if self.abstraction_engine.apply_abstraction(name, sent_context):
                                    log("ABSTRACT", f"💡 Matched pattern: {name}", Term.PURPLE)
                                    self.memory.episodic.record(Episode(
                                        time.time(), action, sent_context, True, 0, abstraction_tags=[name]
                                    ))

                            # C. Simulate Thinking Time
                            # Deep reading takes time!
                            await asyncio.sleep(0.1) 
                        
                        # Yield every few sentences to keep the agent responsive
                        if i % 10 == 0:
                            await asyncio.sleep(0.001) 
                            print(f"\rReading... {i}/{total_sentences} (C:{complexity:.1f})", end="")

                    print(f"\rReading Complete!             ")
                    return True, None

                except Exception as e:
                    log("ACT", f"Read error: {e}", Term.RED)
                    return False, FailureType.EXECUTION_ERROR
                
            elif action.action_type == ActionType.SPEAK:
                payload = action.payload
                
                # If payload is a list of concepts (Raw Thought), convert to Sentence
                if isinstance(payload, list):
                    # We need access to the language engine. 
                    # Ideally, ActionExecutor should share the instance or have its own.
                    # For now, we can instantiate a lightweight one or pass it down.
                    # HACK: We'll do a simple join for now, but in the future, 
                    # pass 'language_engine' to ActionExecutor just like PerceptionAgent.
                    message = " ".join(str(p) for p in payload)

                elif action.action_type == ActionType.SPEAK:
                    payload = action.payload
                
                # If payload is a list (raw concepts), turn it into a sentence
                if isinstance(payload, list):
                    # Use the Perception Agent's language engine (or the shared one)
                    # Assuming you passed 'perception' to Executor or gave Executor its own engine
                    message = self.perception.language_engine.construct_sentence(payload)
   
                else:
                    message = str(payload)
                
                log("ACT", f"🗣️  Speaking: {message}", Term.CYAN)
                speak(message)
                return True, None
                
            elif action.action_type == ActionType.SELF_MODIFY:
                try:
                    payload = str(action.payload)
                    trait, value = payload.split("=")
                    success = self.genome.modify(trait.strip(), float(value))
                    return success, (None if success else FailureType.INVALID_INPUT)
                except Exception:
                    return False, FailureType.INVALID_INPUT

            elif action.action_type == ActionType.CHECKPOINT:
                self.genome.create_checkpoint(action.payload)
                return True, None

            elif action.action_type == ActionType.BUILD_SKILL:
                name = str(action.payload)
                new_skill = Skill(
                    name=name,
                    description="Auto-generated skill",
                    preconditions=[],
                    effects=[],
                    code="print('Executed auto skill')"
                )
                if self.skill_library.add_skill(new_skill):
                    return True, None
                return False, FailureType.CONFLICT

            elif action.action_type == ActionType.HEAL:
                repairs = self.immune_system.prescribe_repair()
                for repair_action in repairs:
                    log("ACT", f"Executing repair: {repair_action.action_type.name}", Term.CYAN)
                return True, None

            elif action.action_type == ActionType.EXPLORE:
                if self.memory.semantic.idx_to_concept:
                    idx = random.choice(list(self.memory.semantic.idx_to_concept.keys()))
                    concept = self.memory.semantic.idx_to_concept[idx]
                    log("ACT", f"Exploring concept: {concept}", Term.PURPLE)
                    self.memory.semantic.stimulate(concept, strength=0.5)
                return True, None
            
            elif action.action_type == ActionType.REFLECT:
                log("ACT", "Reflecting on recent events...", Term.BLUE)
                self.memory.consolidate()
                return True, None

            # Default fallback
            log("ACT", f"Executed {action.action_type.name}", Term.DIM)
            return True, None

        except Exception as e:
            log("ACT", f"Execution Error: {e}", Term.RED)
            return False, FailureType.EXECUTION_ERROR
        finally:
            # Record episode
            duration = time.time() - start_time
            episode = Episode(
                timestamp=start_time,
                action=action,
                context=goal.context,
                outcome=True,
                latency=duration
            )
            self.memory.episodic.record(episode)

class GeneticProgrammer:
    """
    Evolves new skills by mutating the Abstract Syntax Tree (AST) 
    of existing skills.
    """
    def evolve_skill(self, parent_code: str) -> str:
        """
        Takes a working Python function and mutates it 
        (changes numbers, swaps operators) to try and optimize it.
        """
        try:
            tree = ast.parse(parent_code)
            
            class MutationVisitor(ast.NodeTransformer):
                def visit_BinOp(self, node):
                    # Randomly swap operators (e.g., + becomes *)
                    if random.random() < 0.2:
                        ops = [ast.Add(), ast.Sub(), ast.Mult(), ast.Div()]
                        node.op = random.choice(ops)
                    return node

                def visit_Constant(self, node):
                    # Randomly tweak numbers
                    if isinstance(node.value, (int, float)) and random.random() < 0.2:
                        node.value *= random.uniform(0.5, 1.5)
                    return node

            # Apply mutation
            mutator = MutationVisitor()
            new_tree = mutator.visit(tree)
            return ast.unparse(new_tree)
            
        except Exception:
            return parent_code # Fallback if syntax error

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
from engines.abstraction import AbstractionEngine
from engines.reasoning import ReasoningEngine
from engines.language import LanguageEngine

from Prometheus.genome import Genome


from config import *
from utils import speak, log, Term
from definitions import *
from genome import Genome
from core_systems import PrincipleSystem, SkillLibrary, ActiveGlobalWorkspace
from Prometheus.memory import LongTermMemory, SelfModel, Episode, WorkingMemory
from Prometheus.health import CognitiveImmuneSystem
from infrastructure import NeuralBus


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
        
        # NOTE: Track real outcome so episodic memory isn't poisoned.
        outcome = False
        failure_reason: Optional[FailureType] = None


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
                    outcome = True
                    failure_reason = None
                    return outcome, failure_reason
                                    
            elif action.action_type == ActionType.MOTOR_COMMAND:
                # Parse "MOVE_FORWARD 10"
                try:
                    cmd, val = str(action.payload).split(" ", 1)
                    self.bridge.dispatch_motor_command(cmd, val)
                    outcome = True
                    failure_reason = None
                    return outcome, failure_reason
                             
                except:
                    return False, FailureType.INVALID_INPUT

            elif action.action_type == ActionType.RESEARCH:
                query = str(action.payload)
                log("ACT", f"Researching: {query}", Term.BLUE)
                await self.bus.publish("network", query)
                outcome = True
                failure_reason = None
                return outcome, failure_reason
                             
            elif action.action_type == ActionType.READ_FILE:
                filepath = str(action.payload)
                log("ACT", f"📖 Opening file: {filepath}", Term.BLUE)
                
                # Check for "Deep Mode" flag (set by Executive or User)
                is_deep_mode = action.metadata.get("mode") == "DEEP"
                
                if not os.path.exists(filepath):
                    outcome = False
                    failure_reason = FailureType.INVALID_INPUT
                    return outcome, failure_reason

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
                    outcome = True
                    failure_reason = None
                    return outcome, failure_reason
                             
                except Exception as e:
                    log("ACT", f"Read error: {e}", Term.RED)
                    return False, FailureType.EXECUTION_ERROR
                
            elif action.action_type == ActionType.SPEAK:
                payload = action.payload
                
                # If payload is a list (Raw Thought/Concepts), convert to Sentence
                if isinstance(payload, list):
                    # Use the internal language engine directly
                    try:
                        message = self.language_engine.construct_sentence(payload)
                    except AttributeError:
                        # Fallback if construct_sentence isn't available or fails
                        message = " ".join(str(p) for p in payload)
                else:
                    message = str(payload)
                
                log("ACT", f"🗣️  Speaking: {message}", Term.CYAN)
                speak(message)
                outcome = True
                failure_reason = None
                return outcome, failure_reason
                             
            elif action.action_type == ActionType.SELF_MODIFY:
                try:
                    payload = str(action.payload)
                    trait, value = payload.split("=")
                    success = self.genome.modify(trait.strip(), float(value))
                    return success, (None if success else FailureType.INVALID_INPUT)
                except Exception:
                    outcome = False
                    failure_reason = FailureType.INVALID_INPUT
                    return outcome, failure_reason

            elif action.action_type == ActionType.CHECKPOINT:
                self.genome.create_checkpoint(action.payload)
                outcome = True
                failure_reason = None
                return outcome, failure_reason
                             
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
                    outcome = True
                    failure_reason = None
                    return outcome, failure_reason
                             
                return False, FailureType.CONFLICT
            
            elif action.action_type == ActionType.MODIFY_ARCHITECTURE:
                # Payload example: {"new_connection": ["perception", "execution"]}
                # or {"module_weights": {"planning": 0.2}}
                try:
                    modification = action.payload
                    if isinstance(modification, str):
                        # Try to parse stringified dict if necessary
                        import ast
                        modification = ast.literal_eval(modification)
                    
                    if isinstance(modification, dict):
                        success = self.genome.modify_architecture(modification, source="EXECUTOR")
                        return success, (None if success else FailureType.INVALID_INPUT)
                    else:
                        return False, FailureType.INVALID_INPUT
                except Exception as e:
                    log("ACT", f"Arch Mod Error: {e}", Term.RED)
                    return False, FailureType.EXECUTION_ERROR

            elif action.action_type == ActionType.HEAL:
                repairs = self.immune_system.prescribe_repair()
                for repair_action in repairs:
                    log("ACT", f"Executing repair: {repair_action.action_type.name}", Term.CYAN)
                outcome = True
                failure_reason = None
                return outcome, failure_reason
                             
            elif action.action_type == ActionType.EXPLORE:
                if self.memory.semantic.idx_to_concept:
                    idx = random.choice(list(self.memory.semantic.idx_to_concept.keys()))
                    concept = self.memory.semantic.idx_to_concept[idx]
                    log("ACT", f"Exploring concept: {concept}", Term.PURPLE)
                    self.memory.semantic.stimulate(concept, strength=0.5)
                outcome = True
                failure_reason = None
                return outcome, failure_reason
                             
            elif action.action_type == ActionType.REFLECT:
                log("ACT", "Reflecting on recent events...", Term.BLUE)
                self.memory.consolidate()
                outcome = True
                failure_reason = None
                return outcome, failure_reason
                             
            # Default fallback
            log("ACT", f"Executed {action.action_type.name}", Term.DIM)
            outcome = True
            failure_reason = None
            return outcome, failure_reason
                             
        except Exception as e:
            log("ACT", f"Execution Error: {e}", Term.RED)
            return False, FailureType.EXECUTION_ERROR
        finally:
            duration = time.time() - start_time

            # NOTE: outcome/failure_reason now reflect what actually happened.
            episode = Episode(
                action=action,
                outcome=outcome,
                failure_type=failure_reason,
                context=goal.context,
                timestamp=start_time,
                latency=duration
            )
            self.memory.episodic.record(episode)

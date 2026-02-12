# core_systems.py
import heapq
import time
import copy
import json
import os
import asyncio
from typing import List, Dict, Any, Optional, Tuple, Set, Deque
from collections import deque, defaultdict

# Safe import for GPU acceleration
try:
    import cupy as cp
    xp = cp
except ImportError:
    import numpy as np
    xp = np

from config import *
from utils import log, Term
from definitions import *
from neural import SimpleNeuralNet
from genome import Genome
from memory import WorkingMemory

class PrincipleSystem:
    """
    Intrinsic motivation system based on principles, not hardcoded drives.
    The agent develops behavior from first principles.
    """
    def __init__(self):
        self.principles = {
            p: Principle(p) for p in CognitivePrinciple
        }
        self.principle_history = deque(maxlen=1000)

    def evaluate_state(self, state: Dict[str, Any]) -> Dict[CognitivePrinciple, float]:
        """Evaluate how well each principle is satisfied"""
        evaluations = {}
        
        # HOMEOSTASIS - balanced internal state
        if 'frustration' in state and 'confidence' in state:
            balance = 1.0 - abs(state['frustration'] - state['confidence'])
            evaluations[CognitivePrinciple.HOMEOSTASIS] = balance
        
        # PREDICTION - success rate indicates good world model
        if 'success_rate' in state:
            evaluations[CognitivePrinciple.PREDICTION] = state['success_rate']
        
        # INFORMATION_GAIN - learning rate
        if 'concepts_learned_rate' in state:
            evaluations[CognitivePrinciple.INFORMATION_GAIN] = min(1.0, state['concepts_learned_rate'])
        
        # COMPLEXITY_BOUND - avoid cognitive overload
        if 'goal_queue_size' in state and 'max_queue_size' in state:
            load = state['goal_queue_size'] / state['max_queue_size']
            evaluations[CognitivePrinciple.COMPLEXITY_BOUND] = 1.0 - load
        
        # Update satisfaction
        for principle, value in evaluations.items():
            self.principles[principle].update_satisfaction(value)
            self.principle_history.append((time.time(), principle, value))
        
        return evaluations

    def get_intrinsic_reward(self) -> float:
        """
        Compute intrinsic reward from principle satisfaction.
        This drives behavior without explicit goals.
        """
        weighted_sum = sum(
            p.weight * p.satisfaction 
            for p in self.principles.values()
        )
        total_weight = sum(p.weight for p in self.principles.values())
        return weighted_sum / total_weight if total_weight > 0 else 0.5

    def adapt_weights(self):
        """Meta-learn: adjust principle weights based on outcomes"""
        # Increase weight of frequently violated principles
        for principle in self.principles.values():
            if principle.satisfaction < 0.3:
                principle.violations += 1
                if principle.violations > 5:
                    principle.weight = min(2.0, principle.weight * 1.1)
                    log("PRINCIPLE", f"↑ {principle.type.name} weight: {principle.weight:.2f}", 
                        Term.PURPLE)
            else:
                principle.violations = max(0, principle.violations - 1)

class SkillLibrary:
    """
    Dynamic library of learned skills.
    Agent can create, compose, and modify skills.
    """
    def __init__(self):
        self.skills: Dict[str, Skill] = {}
        self.skill_graph: Dict[str, Set[str]] = defaultdict(set)  # Dependencies
        self.load()

    def add_skill(self, skill: Skill) -> bool:
        """Add a new skill to the library"""
        if skill.name in self.skills:
            log("SKILLS", f"Skill {skill.name} already exists", Term.YELLOW)
            return False
        
        self.skills[skill.name] = skill
        
        # Update dependency graph
        for parent in skill.parent_skills:
            self.skill_graph[parent].add(skill.name)
        
        log("SKILLS", f"📚 Learned new skill: {skill.name}", Term.GREEN)
        self.save()
        return True

    def compose_skills(self, skill_names: List[str], new_name: str) -> Optional[Skill]:
        """
        Compose multiple skills into a new compound skill.
        This is how the agent builds complex behaviors from simple ones.
        """
        if not all(name in self.skills for name in skill_names):
            return None
        
        skills = [self.skills[name] for name in skill_names]
        
        # Combine preconditions (union)
        combined_preconditions = list(set(
            pre for skill in skills for pre in skill.preconditions
        ))
        
        # Combine effects (last skill's effects + intermediate effects)
        combined_effects = list(set(
            eff for skill in skills for eff in skill.effects
        ))
        
        # Compose code (sequential execution)
        combined_code = "\n".join(skill.code for skill in skills)
        
        # Description
        description = f"Composite of: {', '.join(skill_names)}"
        
        new_skill = Skill(
            name=new_name,
            description=description,
            preconditions=combined_preconditions,
            effects=combined_effects,
            code=combined_code,
            parent_skills=skill_names
        )
        
        return new_skill

    def find_applicable_skills(self, context: Dict[str, Any]) -> List[Skill]:
        """Find skills that can be executed in current context"""
        return [
            skill for skill in self.skills.values()
            if skill.can_execute(context)
        ]

    def get_best_skill(self, context: Dict[str, Any], objective: str = None) -> Optional[Skill]:
        """Get best skill for current context"""
        applicable = self.find_applicable_skills(context)
        if not applicable:
            return None
        
        # Simple heuristic: highest success rate
        return max(applicable, key=lambda s: s.success_rate)

    def save(self):
        """Persist skills to disk"""
        try:
            data = {
                name: {
                    'description': skill.description,
                    'preconditions': skill.preconditions,
                    'effects': skill.effects,
                    'code': skill.code,
                    'success_rate': skill.success_rate,
                    'uses': skill.uses,
                    'created_at': skill.created_at,
                    'parent_skills': skill.parent_skills
                }
                for name, skill in self.skills.items()
            }
            with open(SKILLS_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            log("SKILLS", f"Save failed: {e}", Term.RED)

    def load(self):
        """Load skills from disk"""
        if os.path.exists(SKILLS_FILE):
            try:
                with open(SKILLS_FILE, 'r') as f:
                    data = json.load(f)
                
                for name, skill_data in data.items():
                    skill = Skill(
                        name=name,
                        description=skill_data['description'],
                        preconditions=skill_data['preconditions'],
                        effects=skill_data['effects'],
                        code=skill_data['code'],
                        success_rate=skill_data['success_rate'],
                        uses=skill_data['uses'],
                        created_at=skill_data['created_at'],
                        parent_skills=skill_data.get('parent_skills', [])
                    )
                    self.skills[name] = skill
                
                log("SKILLS", f"Loaded {len(self.skills)} skills", Term.GREEN)
            except Exception as e:
                log("SKILLS", f"Load failed: {e}", Term.RED)

class ActiveGlobalWorkspace:
    """
    The unified 'Theater of Consciousness'.
    Combines priority-queue messaging (for the Cognitive Mesh) 
    with a 'Broadcast' mechanism (for global awareness).
    """
    def __init__(self, genome: Genome):
        self.genome = genome
        
        # 1. Mesh Support (Heap Queue)
        # Priority queue: (priority, timestamp, source, content)
        # We use negative priority for max-heap behavior via heapq
        self.message_queue = [] 
        self.history = deque(maxlen=100)
        self.msg_counter = 0 
        
        # 2. Broadcasting (Current "Conscious" Thought)
        self.current_broadcast: Optional[Dict] = None
        
        # 3. Working Memory (Short-term storage)
        self.working_memory = WorkingMemory()

    def post(self, priority: float, source: str, content: Any, type: str = "GENERIC") -> None:
        """
        Post a message to the workspace.
        This handles both queuing it for the Mesh and updating the global broadcast.
        """
        timestamp = time.time()
        self.msg_counter += 1
        
        # A. Queue for Mesh Processing
        # (Using negative priority because heapq is a min-heap)
        heapq.heappush(self.message_queue, (-priority, timestamp, source, content))
        
        # B. Update Global Broadcast (Competition)
        # If this new signal is stronger than the current thought, it takes over.
        current_strength = self.current_broadcast['salience'] if self.current_broadcast else 0.0
        
        if priority > current_strength:
            self.current_broadcast = {
                "source": source,
                "content": content,
                "salience": priority,
                "type": type,
                "timestamp": timestamp
            }
            # Log significant shifts
            if priority > 6.0:
                preview = str(content)[:60].replace('\n', ' ')
                log("GWS", f"📢 FOCUS: [{source}] -> {preview}...", Term.BOLD)

        # History log
        self.history.append({
            "time": timestamp, 
            "source": source, 
            "content": content, 
            "p": priority
        })

    def get_most_salient(self) -> Optional[Tuple[float, str, Any]]:
        """
        Retrieve the most important message for the Cognitive Mesh to process.
        Returns: (priority, source, content)
        """
        if not self.message_queue:
            return None
        
        # Pop highest priority (smallest negative number)
        p, ts, source, content = heapq.heappop(self.message_queue)
        
        # Return as positive priority
        return (-p, source, content)

    def peek(self) -> Optional[Dict]:
        """Nodes read this to see what is currently 'conscious' without consuming it."""
        return self.current_broadcast

    def prune_stale(self, max_age: float = 30.0) -> None:
        """Remove old messages that weren't processed."""
        current_time = time.time()
        new_queue = []
        for p, ts, source, content in self.message_queue:
            if current_time - ts < max_age:
                new_queue.append((p, ts, source, content))
        
        if len(new_queue) < len(self.message_queue):
            heapq.heapify(new_queue)
            self.message_queue = new_queue

    async def decay_loop(self):
        """
        Background process to decay the 'Current Broadcast'.
        Prevents the mind from getting stuck on one thought.
        """
        while True:
            if self.current_broadcast:
                # Exponential decay
                self.current_broadcast['salience'] *= 0.90
                
                # If too weak, mind goes blank
                if self.current_broadcast['salience'] < 0.05:
                    self.current_broadcast = None 
            
            await asyncio.sleep(0.1)

class PredictiveWorldModel:
    """
    Upgraded: Probabilistic Recurrent World Model.
    Predicts next state (Mean) and Uncertainty (Variance).
    """
    def __init__(self, state_dim=256, action_dim=64):
        self.state_dim = state_dim
        # Hidden state for temporal continuity (LSTM-like memory)
        self.hidden_state = xp.zeros((1, 128)) 
        
        # We use two heads: one for prediction, one for uncertainty (aleatoric)
        self.net = SimpleNeuralNet(
            input_size=state_dim + action_dim + 128, # State + Action + Memory
            hidden_size=256,
            output_size=state_dim + 128 # NextState + NewMemory
        )
        
        # Uncertainty estimator (simple variance network)
        self.uncertainty_net = SimpleNeuralNet(
            input_size=state_dim + action_dim,
            hidden_size=64,
            output_size=1
        )

    def imagine(self, current_state, action_vector, steps=1) -> List[Dict]:
        """
        Mental Time Travel: Recursively predict future states.
        Returns a trajectory of imagined states and accumulated uncertainty.
        """
        trajectory = []
        sim_state = current_state
        sim_hidden = self.hidden_state.copy()
        
        for _ in range(steps):
            inputs = xp.concatenate([sim_state, action_vector, sim_hidden], axis=None)
            
            # 1. Predict Next Reality
            output = self.net.forward(inputs).flatten()
            next_state = output[:self.state_dim]
            next_hidden = output[self.state_dim:]
            
            # 2. Estimate Confidence (Am I hallucinating?)
            uncertainty_inputs = xp.concatenate([sim_state, action_vector], axis=None)
            uncertainty = float(self.uncertainty_net.forward(uncertainty_inputs))
            
            trajectory.append({
                "state": next_state,
                "uncertainty": uncertainty,
                "timestamp": time.time() + (_ * 0.1)
            })
            
            # Update for next step
            sim_state = next_state
            sim_hidden = next_hidden
            
        return trajectory

    def learn(self, state_t, action_t, state_t1, modulation: float = 1.0):
        """
        Active Inference Update: Minimizing 'Free Energy' (Surprise).
        Added: modulation parameter for variable plasticity.
        """
        # Ensure inputs are flat arrays
        s_t = xp.array(state_t).flatten()
        a_t = xp.array(action_t).flatten()
        s_t1 = xp.array(state_t1).flatten()
        
        # 1. Prepare Inputs
        current_hidden = self.hidden_state.flatten()
        inputs = xp.concatenate([s_t, a_t, current_hidden])

        # 2. Prepare Targets
        targets = xp.concatenate([s_t1, current_hidden])

        # 3. Train Prediction Network (Pass modulation down)
        surprise = self.net.train(inputs, targets, modulation=modulation)

        # 4. Train Uncertainty Network
        uncertainty_inputs = xp.concatenate([s_t, a_t])
        target_uncertainty = xp.array([surprise ** 2])
        
        # Uncertainty net also learns faster if we are in a high-plasticity state
        self.uncertainty_net.train(uncertainty_inputs, target_uncertainty, modulation=modulation)

        return surprise
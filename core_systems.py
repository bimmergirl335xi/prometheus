# core_systems.py
from config import *
from utils import *
from definitions import *
from neural import SimpleNeuralNet
import numpy as np

class Genome:
    """
    Not just parameters - this represents the entire cognitive architecture.
    The genome can modify itself, not just traits.
    """
    def __init__(self):
        # Core traits
        self.traits = {
            "cycle_speed": 0.2,
            "recursion_limit": 3,
            "curiosity_decay": 0.005,
            "learning_rate": 0.1,
            "confidence_threshold": 0.7,
            "plasticity": 0.5,
            "risk_tolerance": 0.3,
            "attention_decay": 0.999,
            "simulation_depth": 2,
            "memory_consolidation_rate": 0.1,
            "memory_retention": 0.9995,
            "initiative_level": 0.5,
            "exploration_bonus": 0.2,
            "planning_horizon": 5,
            
            # New meta-traits
            "architectural_plasticity": 0.3,  # How much structure can change
            "meta_learning_rate": 0.05,       # How fast to learn to learn
            "abstraction_threshold": 0.6,     # When to form abstractions
            "pruning_threshold": 0.05,         # When to remove unused structures
            "composition_bias": 0.4,          # Preference for combining vs creating
        }
        
        self.values = {
            "coherence": 0.8,
            "curiosity": 0.6,
            "self_preservation": 0.4,
            "efficiency": 0.5,
            "autonomy": 0.6,
            "growth": 0.7,  # New: drive to expand capabilities
        }
        
        # Architectural components
        self.architecture = {
            "modules": ["perception", "planning", "execution", "meta"],
            "connections": {
                "perception": ["planning", "meta"],
                "planning": ["execution", "meta"],
                "execution": ["meta"],
                "meta": ["perception", "planning", "execution"]
            },
            "module_weights": {
                "perception": 1.0,
                "planning": 1.0,
                "execution": 1.0,
                "meta": 1.0
            }
        }
        
        # Modification history with rollback capability
        self.modification_history: List[Dict[str, Any]] = []
        self.checkpoints: List[Dict[str, Any]] = []
        
        self.load()

    def modify(self, trait: str, value: float, source: str = "self") -> bool:
        """Modify trait with safety and logging"""
        if trait not in self.traits:
            log("GENOME", f"Unknown trait: {trait}", Term.RED)
            return False
        
        old = self.traits[trait]
        delta = (value - old) * self.traits["plasticity"]
        new_value = old + delta
        
        # Safety bounds
        bounds = self._get_safety_bounds(trait)
        new_value = max(bounds[0], min(bounds[1], new_value))
        
        if abs(new_value - old) < 1e-6:
            return False
        
        self.traits[trait] = new_value
        
        # Record for rollback
        self.modification_history.append({
            "type": "trait",
            "trait": trait,
            "old": old,
            "new": new_value,
            "timestamp": time.time(),
            "source": source
        })
        
        log("GENOME", f"🧬 {trait} {old:.3f} → {new_value:.3f} (by {source})", Term.RED)
        self.save()
        return True

    def modify_architecture(self, modification: dict[str, any], source: str = "self") -> bool:
        """
        Modify the cognitive architecture itself.
        This is meta-self-modification - changing how you change.
        """
        if self.values["self_preservation"] > 0.5:
            # Safety check: don't allow critical structural changes without careful review
            if "modules" in modification and len(modification["modules"]) < 2:
                log("GENOME", "❌ Architectural modification rejected: too few modules", Term.RED)
                return False
        
        plasticity = self.traits["architectural_plasticity"]
        
        try:
            old_architecture = copy.deepcopy(self.architecture)
            
            # Apply modification with plasticity-based weighting
            if "module_weights" in modification:
                for module, new_weight in modification["module_weights"].items():
                    if module in self.architecture["module_weights"]:
                        old_weight = self.architecture["module_weights"][module]
                        delta = (new_weight - old_weight) * plasticity
                        self.architecture["module_weights"][module] = max(0.1, old_weight + delta)
            
            # Record for rollback
            self.modification_history.append({
                "type": "architecture",
                "old": old_architecture,
                "new": copy.deepcopy(self.architecture),
                "timestamp": time.time(),
                "source": source
            })
            
            log("GENOME", f"🏗️  Architecture modified by {source}", Term.RED)
            self.save()
            return True
        
        except Exception as e:
            log("GENOME", f"Architecture modification failed: {e}", Term.RED)
            return False

    def create_checkpoint(self, name: str = None) -> str:
        """Create a rollback point"""
        checkpoint_id = name or f"checkpoint_{int(time.time())}"
        
        checkpoint = {
            "id": checkpoint_id,
            "timestamp": time.time(),
            "traits": copy.deepcopy(self.traits),
            "values": copy.deepcopy(self.values),
            "architecture": copy.deepcopy(self.architecture)
        }
        
        self.checkpoints.append(checkpoint)
        log("GENOME", f"💾 Checkpoint created: {checkpoint_id}", Term.CYAN)
        return checkpoint_id

    def rollback(self, checkpoint_id: str = None) -> bool:
        """
        Rollback to a previous state.
        Critical for safe self-modification.
        """
        if not self.checkpoints:
            log("GENOME", "No checkpoints available", Term.RED)
            return False
        
        if checkpoint_id:
            checkpoint = next((c for c in self.checkpoints if c["id"] == checkpoint_id), None)
        else:
            checkpoint = self.checkpoints[-1]  # Most recent
        
        if not checkpoint:
            log("GENOME", f"Checkpoint {checkpoint_id} not found", Term.RED)
            return False
        
        try:
            self.traits = copy.deepcopy(checkpoint["traits"])
            self.values = copy.deepcopy(checkpoint["values"])
            self.architecture = copy.deepcopy(checkpoint["architecture"])
            
            log("GENOME", f"⏮️  Rolled back to: {checkpoint['id']}", Term.YELLOW)
            self.save()
            return True
        
        except Exception as e:
            log("GENOME", f"Rollback failed: {e}", Term.RED)
            return False

    def _get_safety_bounds(self, trait: str) -> Tuple[float, float]:
        """Dynamic safety bounds based on architectural plasticity"""
        base_bounds = {
            "cycle_speed": (0.01, 2.0),
            "recursion_limit": (1, 20),
            "confidence_threshold": (0.1, 0.95),
            "attention_decay": (0.1, 0.99),
            "plasticity": (0.01, 1.0),
            "learning_rate": (0.001, 1.0),
            "curiosity_decay": (0.0001, 0.1),
            "risk_tolerance": (0.0, 1.0),
            "simulation_depth": (1, 10),
            "memory_consolidation_rate": (0.01, 1.0),
            "memory_retention": (0.5, 0.99),
            "initiative_level": (0.0, 1.0),
            "exploration_bonus": (0.0, 1.0),
            "planning_horizon": (1, 20),
            "architectural_plasticity": (0.0, 1.0),
            "meta_learning_rate": (0.001, 0.5),
            "abstraction_threshold": (0.1, 0.95),
            "pruning_threshold": (0.01, 0.5),
            "composition_bias": (0.0, 1.0),
        }
        
        # With high architectural plasticity, bounds can be more flexible
        bounds = base_bounds.get(trait, (0.0, 1.0))
        plasticity_factor = self.traits.get("architectural_plasticity", 0.3)
        
        # Expand bounds slightly with higher plasticity
        range_expansion = (bounds[1] - bounds[0]) * plasticity_factor * 0.2
        expanded_bounds = (
            max(0.0, bounds[0] - range_expansion),
            bounds[1] + range_expansion
        )
        
        return expanded_bounds

    def save(self):
        """Persist genome and architecture"""
        try:
            data = {
                "traits": self.traits,
                "values": self.values,
                "architecture": self.architecture,
                "modification_history": self.modification_history[-100:],
                "checkpoints": self.checkpoints[-10:]
            }
            with open(SAVE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            log("GENOME", f"Save failed: {e}", Term.RED)

    def load(self):
        """Load genome and architecture"""
        if os.path.exists(SAVE_FILE):
            try:
                with open(SAVE_FILE, "r") as f:
                    data = json.load(f)
                    self.traits.update(data.get("traits", {}))
                    self.values.update(data.get("values", {}))
                    self.architecture.update(data.get("architecture", {}))
                    self.modification_history = data.get("modification_history", [])
                    self.checkpoints = data.get("checkpoints", [])
                    log("GENOME", "Loaded adaptive genome", Term.RED)
            except Exception as e:
                log("GENOME", f"Load failed: {e}", Term.RED)

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
    A competitive "Blackboard" system.
    Thoughts fight for dominance. Only the strongest signal is 'Conscious'.
    """
    def __init__(self):
        self.current_broadcast: Optional[Dict] = None
        self.history = []
        
    def peek(self) -> Optional[Dict]:
        """Nodes read this to see what is currently 'conscious'."""
        return self.current_broadcast

    def post(self, source: str, content: Any, salience: float, type: str = "GENERIC"):
        """
        Post a new signal. 
        Interrupts current thought ONLY if salience is higher.
        """
        current_strength = self.current_broadcast['salience'] if self.current_broadcast else 0.0
        
        # COMPETITION LOGIC
        if salience > current_strength:
            # Winner takes all
            self.current_broadcast = {
                "source": source,
                "content": content,
                "salience": salience,
                "type": type,
                "timestamp": time.time()
            }
            # Log significant shifts
            if salience > 0.5:
                preview = str(content)[:60].replace('\n', ' ')
                log("GWS", f"📢 FOCUS: [{source}] ({type}) -> {preview}...", Term.BOLD)
            
            self.history.append(self.current_broadcast)
            if len(self.history) > 50: self.history.pop(0)

    async def decay_loop(self):
        """
        The 'Forgetting' Cycle.
        Reduces salience of the current thought so the mind doesn't get stuck.
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
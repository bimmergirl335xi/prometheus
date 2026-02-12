import time
import json
import os
import copy
from typing import Dict, Any, List, Tuple
from config import SAVE_FILE
from utils import log, Term

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

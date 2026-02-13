from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import random
from utils import log, Term
from definitions import Action, ActionType
from genome import Genome

class CuriosityEngine:
    """
    Drives intrinsic motivation by monitoring information entropy and novelty.
    Uses Genome traits to balance between Exploration (trying new things) 
    and Exploitation (refining known things).
    """
    def __init__(self, genome: Genome, memory: Any):
        self.genome = genome
        self.memory = memory
        
        # Internal State
        self.boredom_level = 0.0
        self.current_obsession: Optional[str] = None
        self.known_keys_cache = set()
        
        # Wundt Curve parameters (optimal novelty)
        self.optimal_novelty = 0.6  # Can be modulated by genome
        
    def _get_trait(self, key: str, default: float = 0.5) -> float:
        return self.genome.traits.get(key, default)
    
    def calculate_novelty(self, context: Dict[str, Any]) -> float:
        """
        Quantifies how 'new' the current situation is.
        0.0 = Totally mundane/repetitive.
        1.0 = Completely alien/unpredictable.
        """
        if not context:
            return 0.0
            
        novelty_score = 0.0
        keys = list(context.keys())
        
        # 1. Structural Novelty (New keys we haven't seen)
        unknown_keys = [k for k in keys if k not in self.known_keys_cache]
        if unknown_keys:
            novelty_score += 0.3 * len(unknown_keys)
            self.known_keys_cache.update(unknown_keys)

        # 2. Semantic Novelty (Vector distance)
        # We sample a few values and check if they exist in Semantic Memory
        semantic = self.memory.semantic
        hits = 0
        total_checks = 0
        
        for val in context.values():
            if isinstance(val, str) and len(val) < 50:
                total_checks += 1
                if val in semantic.concept_to_idx:
                    hits += 1
        
        if total_checks > 0:
            familiarity = hits / total_checks
            novelty_score += (1.0 - familiarity) * 0.5
            
        return min(1.0, novelty_score)
    
    def process_stimuli(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        The main tick. 
        Returns a 'Curiosity Goal' if intrinsic drive triggers, else None.
        """
        # 1. Decay/Grow Boredom
        dt_decay = self._get_trait("curiosity_decay", 0.005)
        novelty = self.calculate_novelty(context)
        
        # If novelty is low, boredom grows. If high, boredom drops.
        if novelty < 0.2:
            self.boredom_level += (dt_decay * 5)
        else:
            self.boredom_level -= (novelty * 2.0)
            
        self.boredom_level = max(0.0, min(1.0, self.boredom_level))

        # 2. Check Thresholds (modulated by Genome)
        initiative = self._get_trait("initiative_level", 0.5)
        
        # THRESHOLD 1: Exploration (Too Bored)
        # High initiative makes the agent get bored faster
        if self.boredom_level > (1.0 - initiative):
            log("CURIOSITY", f"😒 Boredom critical ({self.boredom_level:.2f}). Seeking novelty.", Term.CYAN)
            self.boredom_level = 0.0 # Reset
            return self._generate_exploration_goal()

        # THRESHOLD 2: Investigation (Too Surprised)
        # If something is VERY novel, we must understand it
        if novelty > self._get_trait("risk_tolerance", 0.3) + 0.4:
            log("CURIOSITY", f"😲 High Surprise ({novelty:.2f}). Investigating.", Term.MAGENTA)
            return self._generate_investigation_goal(context)

        return None
    
    def _generate_exploration_goal(self) -> Dict[str, Any]:
        """Generates a goal to break out of a loop or try something random."""
        # Check gaps in memory (Concepts with few connections)
        target = "random_concept"
        if hasattr(self.memory.semantic, "get_sparse_concepts"):
             sparse = self.memory.semantic.get_sparse_concepts()
             if sparse: target = random.choice(sparse)

        return {
            "type": "GOAL",
            "source": "CuriosityEngine",
            "content": f"Increase knowledge about '{target}'",
            "priority": 4.0 + (self._get_trait("exploration_bonus", 0.2) * 10),
            "metadata": {"reason": "boredom_alleviation"}
        }

    def _generate_investigation_goal(self, context: Dict) -> Dict[str, Any]:
        """Generates a goal to analyze a specific anomaly."""
        # Identify the most complex/unknown part of the context
        focus_point = "Unknown"
        longest_val = sorted([str(v) for v in context.values()], key=len, reverse=True)
        if longest_val:
            focus_point = longest_val[0][:20]

        return {
            "type": "GOAL",
            "source": "CuriosityEngine",
            "content": f"Analyze structure of: {focus_point}",
            "priority": 8.0, # High priority because it's reacting to surprise
            "metadata": {"reason": "surprise_resolution"}
        }
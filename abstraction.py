# abstraction.py
from typing import List, Dict, Set, Any, Optional, Tuple, Union, TYPE_CHECKING
from collections import defaultdict
import random
import uuid

# Safe import for GPU acceleration
try:
    import cupy as cp
    xp = cp
except ImportError:
    import numpy as np
    xp = np

if TYPE_CHECKING:
    from reasoning import ReasoningEngine

import numpy as np # Keep numpy for CPU fallback
from config import *
from utils import log, Term, log_engine_event
from definitions import Abstraction, ActionType, Episode
from genome import Genome

class AbstractionEngine:
    """
    The 'Scientist' of the agent.
    1. Reviews Episodic Memory to find causal rules (Procedural Abstraction).
    2. Reviews Semantic Memory to group concepts into categories (Conceptual Abstraction).
    """
    def __init__(self, genome: Genome, memory: Any, reasoning_engine: Any = None):
        self.genome = genome
        self.memory = memory
        self.reasoning_engine = reasoning_engine
        self.abstractions: Dict[str, Abstraction] = {}
        self.similarity_threshold = 0.85
        self.min_samples = 3

    def process_new_episodes(self, episodes: List[Episode]) -> List[Abstraction]:
        if len(episodes) < self.min_samples:
            return []
        
        new_rules = []
        by_action = defaultdict(list)
        for ep in episodes:
            if hasattr(ep, 'action'):
                by_action[ep.action.action_type].append(ep)
        
        for action_type, action_episodes in by_action.items():
            successes = [e for e in action_episodes if e.outcome]
            if len(successes) >= self.min_samples:
                rule = self._derive_procedural_rule(action_type, successes)
                if rule:
                    new_rules.append(rule)
        
        valid_rules = []
        for rule in new_rules:
            if rule.name in self.abstractions:
                continue

            if self.reasoning_engine:
                ok, msg = self.reasoning_engine.validate_proposition(
                    f"Rule {rule.name} implies {rule.pattern}"
                )
                if not ok:
                    log("ABSTRACT", f"❌ Rule {rule.name} rejected by Logic: {msg}", Term.RED)
                    continue

            self.abstractions[rule.name] = rule
            log("ABSTRACT", f"💡 New Heuristic: {rule.name}", Term.CYAN)
            valid_rules.append(rule)

        return valid_rules

    def _derive_procedural_rule(self, action_type: ActionType, episodes: List[Episode]) -> Optional[Abstraction]:
        contexts = [e.context for e in episodes]
        if not contexts: return None

        input_snapshot = f"Action: {action_type.name} | Samples: {len(episodes)}"
        common_constraints = {}
        keys = set().union(*(d.keys() for d in contexts))
        
        for key in keys:
            values = [ctx.get(key) for ctx in contexts if key in ctx]
            if len(values) < len(contexts): continue 

            if all(isinstance(v, (int, float)) for v in values):
                arr = np.array(values)
                mean = np.mean(arr)
                std = np.std(arr)
                if std < (mean * 0.2):
                    common_constraints[key] = {"type": "range", "min": mean - std, "max": mean + std}
            
            elif all(isinstance(v, str) for v in values):
                if len(set(values)) == 1:
                    common_constraints[key] = {"type": "exact", "value": values[0]}

        if common_constraints:
            rule_name = f"Rule_{action_type.name}_{len(self.abstractions)}"
            abstraction = Abstraction(
                name=rule_name,
                pattern=common_constraints,
                instances=episodes,
                generality=len(episodes) / (len(episodes) + 1.0)
            )
            return abstraction

        return None

    def cluster_and_consolidate(self):
        semantic = self.memory.semantic
        if not hasattr(semantic, 'idx_to_concept') or len(semantic.idx_to_concept) < 5:
            return

        indices = list(semantic.idx_to_concept.keys())
        vectors = []
        valid_indices = []

        for idx in indices:
            vec = semantic.context_vectors.get(idx) if hasattr(semantic.context_vectors, 'get') else semantic.context_vectors[idx]
            if vec is not None:
                if hasattr(vec, 'get'): vec = vec.get() # Ensure numpy
                vectors.append(vec)
                valid_indices.append(idx)
        
        if not vectors: return
        vectors = np.array(vectors)
        
        visited = set()
        clusters = []

        for i in range(len(vectors)):
            if i in visited: continue
            current_cluster = [valid_indices[i]]
            visited.add(i)
            
            for j in range(i + 1, len(vectors)):
                if j in visited: continue
                sim = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]) + 1e-9)
                if sim > self.similarity_threshold:
                    current_cluster.append(valid_indices[j])
                    visited.add(j)
            
            if len(current_cluster) > 1:
                clusters.append(current_cluster)

        for cluster_indices in clusters:
            self._form_categorical_concept(cluster_indices)

    def _form_categorical_concept(self, child_indices: List[int]):
        semantic = self.memory.semantic
        children_names = [semantic.idx_to_concept[i] for i in child_indices]
        
        child_vectors = [semantic.context_vectors[i] for i in child_indices]
        if hasattr(child_vectors[0], 'get'):
             import cupy as cp
             centroid = cp.mean(cp.array(child_vectors), axis=0)
        else:
             centroid = np.mean(child_vectors, axis=0)

        name = f"CATEGORY_{children_names[0]}_{len(children_names)}"
        
        # Check if already exists to avoid dupes?
        # For now, just add
        new_idx = semantic.add_concept(name, vector=centroid)
        
        if new_idx:
            log("ABSTRACT", f"🔗 Formed Category '{name}' from {children_names}", Term.PURPLE)
            for child_name in children_names:
                semantic.add_fact(child_name, "IS_A", name)
    
    # [Keep form_hierarchies and apply_abstraction as is...]
    def apply_abstraction(self, abstraction_name: str, current_context: Dict) -> bool:
        if abstraction_name not in self.abstractions: return False
        rule = self.abstractions[abstraction_name]
        constraints = rule.pattern
        
        for key, criteria in constraints.items():
            val = current_context.get(key)
            if val is None: return False
            
            if criteria["type"] == "range":
                if not (criteria["min"] <= val <= criteria["max"]): return False
            elif criteria["type"] == "exact":
                if val != criteria["value"]: return False
        return True
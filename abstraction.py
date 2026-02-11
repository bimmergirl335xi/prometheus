# abstraction.py
from typing import List, Dict, Set, Any, Optional, Tuple, Union
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

from config import *
from utils import log, Term
from utils import log, log_engine_event, Term
from definitions import Abstraction, ActionType, Episode
from core_systems import Genome
from memory import SemanticMemory, EpisodicMemory

class AbstractionEngine:
    """
    The 'Scientist' of the agent.
    1. Reviews Episodic Memory to find causal rules (Procedural Abstraction).
    2. Reviews Semantic Memory to group concepts into categories (Conceptual Abstraction).
    """
    def __init__(self, genome: Genome, memory: Any, reasoning_engine: Any = None): # memory is the composite MemorySystem
        self.genome = genome
        self.memory = memory
        self.reasoning_engine = reasoning_engine
        self.abstractions: Dict[str, Abstraction] = {}
        
        # Hyperparameters
        self.similarity_threshold = 0.85  # For conceptual merging
        self.min_samples = 3              # Minimum episodes to form a rule

    def process_new_episodes(self, episodes: List[Episode]) -> List[Abstraction]:
        """
        Main entry point for episodic consolidation.
        Discover patterns in recent episodes and form new heuristics.
        """
        if len(episodes) < self.min_samples:
            return []
        
        new_rules = []
        
        # 1. Group by Action Type (e.g., all "JUMP" actions)
        by_action = defaultdict(list)
        for ep in episodes:
            if hasattr(ep, 'action'):
                by_action[ep.action.action_type].append(ep)
        
        # 2. Analyze each action group
        for action_type, action_episodes in by_action.items():
            # We only learn from successes (positive reinforcement) or 
            # consistent failures (negative constraints)
            successes = [e for e in action_episodes if e.outcome]
            
            if len(successes) >= self.min_samples:
                rule = self._derive_procedural_rule(action_type, successes)
                if rule:
                    new_rules.append(rule)
        
        # 3. Save valid rules
        for rule in new_rules:
            # Check if this rule is novel (not already known)
            if rule.name not in self.abstractions:
                self.abstractions[rule.name] = rule
                log("ABSTRACT", f"💡 New Heuristic: {rule.name}", Term.CYAN)
                log("ABSTRACT", f"   └─ Condition: {rule.pattern}", Term.DIM)

        if rule.name not in self.abstractions:
            self.abstractions[rule.name] = rule
        
        # COUPLING TRIGGER: Abstraction created, verify with Reason
        if self.reasoning_engine:
            # Check if this new rule contradicts existing logical facts
            # This is "Sanity Checking" your own generalizations
            is_valid, msg = self.reasoning_engine.validate_proposition(f"Rule {rule.name} implies {rule.pattern}")
            if not is_valid:
                log("ABSTRACT", f"❌ Rule {rule.name} rejected by Logic: {msg}", Term.RED)
                del self.abstractions[rule.name] # Kill the bad rule
                return []

        return new_rules

    def _derive_procedural_rule(self, action_type: ActionType, episodes: List[Episode]) -> Optional[Abstraction]:
        """
        Statistical analysis of contexts to find what makes an action succeed.
        Returns an Abstraction if strong correlations are found.
        """
        # Flatten contexts: [{"speed": 10}, {"speed": 12}]
        contexts = [e.context for e in episodes]
        
        if not contexts:
            return None

        input_snapshot = f"Action: {action_type.name} | Samples: {len(episodes)} | Context Keys: {list(contexts[0].keys()) if contexts else 'None'}"

        common_constraints = {}
        keys = set().union(*(d.keys() for d in contexts))
        
        for key in keys:
            values = [ctx.get(key) for ctx in contexts if key in ctx]
            
            if len(values) < len(contexts):
                continue # Key not present in all successful cases, ignore it.

            # Handle Numerical Features (Range based)
            if all(isinstance(v, (int, float)) for v in values):
                arr = np.array(values)
                mean = np.mean(arr)
                std = np.std(arr)
                
                # If variance is low, it's a strict requirement (e.g., "Must be speed 10-12")
                # If variance is high, it likely doesn't matter (e.g., "Time of day")
                if std < (mean * 0.2): # 20% tolerance
                    common_constraints[key] = {"type": "range", "min": mean - std, "max": mean + std}
            
            # Handle Categorical Features (Exact match)
            elif all(isinstance(v, str) for v in values):
                # Check if all values are the same
                if len(set(values)) == 1:
                    common_constraints[key] = {"type": "exact", "value": values[0]}

        if common_constraints:
            return Abstraction(
                name=f"Rule_{action_type.name}_{len(self.abstractions)}",
                pattern=common_constraints,
                instances=episodes, # Link back to source memory
                generality=len(episodes) / (len(episodes) + 1.0) # Simple confidence score
            )
        
        if common_constraints:
            rule_name = f"Rule_{action_type.name}_{len(self.abstractions)}"
            
            # Create the object
            abstraction = Abstraction(
                name=rule_name,
                pattern=common_constraints,
                instances=episodes,
                generality=len(episodes) / (len(episodes) + 1.0)
            )
            
            # Log success
            log_engine_event("Abstraction", "DeriveRule", input_snapshot, f"CREATED: {rule_name} => {common_constraints}")
            return abstraction

        return None

    def cluster_and_consolidate(self):
        """
        Unsupervised Conceptual Learning.
        Scans Semantic Memory for concepts that are vector-similar and groups them.
        E.g., "Chair", "Stool", "Bench" -> Form "Seat_Category"
        """
        semantic = self.memory.semantic
        if not hasattr(semantic, 'idx_to_concept') or len(semantic.idx_to_concept) < 5:
            return

        # 1. Retrieve all vectors
        # NOTE: Assumes memory exposes these maps.
        indices = list(semantic.idx_to_concept.keys())
        vectors = []
        valid_indices = []

        for idx in indices:
            vec = semantic.context_vectors.get(idx)
            if vec is not None:
                # Normalize to Numpy for clustering logic
                if hasattr(vec, 'get'): vec = vec.get()
                vectors.append(vec)
                valid_indices.append(idx)
        
        if not vectors: return
        
        vectors = np.array(vectors)
        
        # 2. Naive Clustering (Cosine Similarity)
        # For a production system, use sklearn.cluster.DBSCAN or KMeans
        # Here we do a greedy pass for simplicity.
        
        visited = set()
        clusters = []

        for i in range(len(vectors)):
            if i in visited: continue
            
            current_cluster = [valid_indices[i]]
            visited.add(i)
            
            for j in range(i + 1, len(vectors)):
                if j in visited: continue
                
                # Cosine Similarity
                sim = np.dot(vectors[i], vectors[j]) / (np.linalg.norm(vectors[i]) * np.linalg.norm(vectors[j]) + 1e-9)
                
                if sim > self.similarity_threshold:
                    current_cluster.append(valid_indices[j])
                    visited.add(j)
            
            if len(current_cluster) > 1:
                clusters.append(current_cluster)

        # 3. Create Higher-Order Concepts
        for cluster_indices in clusters:
            self._form_categorical_concept(cluster_indices)

    def _form_categorical_concept(self, child_indices: List[int]):
        """
        Creates a parent concept (Centroid) for a list of children.
        """
        semantic = self.memory.semantic
        children_names = [semantic.idx_to_concept[i] for i in child_indices]
        
        # Check if they already share a parent to avoid duplicates
        # (Simplified check)
        
        # --- LOGGING INPUT ---
        input_data = f"Clustering Children: {children_names}"

        # Calculate Centroid
        child_vectors = [semantic.context_vectors[i] for i in child_indices]
        if hasattr(child_vectors[0], 'get'):
             # Convert list of cupy arrays to numpy for averaging if needed, or stick to cupy
             import cupy as cp
             centroid = cp.mean(cp.array(child_vectors), axis=0)
        else:
             centroid = np.mean(child_vectors, axis=0)

        name = f"CATEGORY_{children_names[0]}_{len(children_names)}"
        new_idx = semantic.add_concept(name, vector=centroid)
        
        if new_idx:
            # Link children to parent
            for child_name in children_names:
                semantic.add_fact(child_name, "IS_A", name)
                
            # --- LOGGING OUTPUT ---
            log_engine_event("Abstraction", "FormCategory", input_data, f"NEW CONCEPT: {name} (ID: {new_idx})")

        # Create Name
        name = f"CATEGORY_{children_names[0]}_{len(children_names)}"
        
        # Add to Memory
        new_idx = semantic.add_concept(name, vector=centroid)
        
        if new_idx:
            log("ABSTRACT", f"🔗 Formed Category '{name}' from {children_names}", Term.PURPLE)
            
            # Link children to parent
            for child_name in children_names:
                semantic.add_fact(child_name, "IS_A", name)

    def form_hierarchies(self):
        """
        Deep Abstraction: Iteratively clusters concepts to form a tree.
        Runs periodically (e.g., during 'sleep' cycles).
        """
        semantic = self.memory.semantic
        if semantic.next_idx < 50: return

        # 1. Get all concept vectors
        # (Assume retrieval logic here)
        vectors = semantic.get_all_vectors()
        
        # 2. Hierarchical Clustering (Simplified Agglomerative)
        # We look for tight clusters of concepts that don't yet have a parent
        
        clusters = self._find_clusters(vectors, threshold=0.85)
        
        for cluster_indices in clusters:
            # Check if this cluster already has a unified "Hypernym" (Parent)
            children = [semantic.idx_to_concept[i] for i in cluster_indices]
            
            # Verify via Semantic Triples if they share a common "IS_A"
            parents = set()
            for child in children:
                p = semantic.query_fact(child, "IS_A")
                if p: parents.add(p)
            
            if len(parents) == 1:
                continue # Already abstracted
            
            # 3. Create the Abstract Concept (The Centroid)
            # e.g., "ABSTRACT_CLUSTER_42" -> renamed to "FRUIT" later by language engine
            new_name = f"CONCEPT_{uuid.uuid4().hex[:4].upper()}"
            
            # Compute Centroid
            child_vectors = [vectors[i] for i in cluster_indices]
            centroid = np.mean(child_vectors, axis=0)
            
            # Add to memory as ABSTRACT type (concept_type=1)
            semantic.add_concept(new_name, vector=centroid, concept_type=1)
            
            # Link children to new Parent
            for child in children:
                semantic.add_fact(child, "IS_A", new_name)
                # Strengthen association
                semantic.associate(child, new_name, weight=0.9)
                
            log("ABSTRACT", f"🧩 Formed Higher-Order Concept: {new_name} covering {children}", Term.PURPLE)

    def apply_abstraction(self, abstraction_name: str, current_context: Dict) -> bool:
        """
        Checks if the current context matches a learned rule.
        Used by the Planner to see if a rule applies.
        """
        if abstraction_name not in self.abstractions:
            return False
        
        rule = self.abstractions[abstraction_name]
        constraints = rule.pattern # e.g. {"speed": {"min": 5, "max": 10}}
        
        matches = True
        for key, criteria in constraints.items():
            val = current_context.get(key)
            if val is None:
                return False # Missing required key
            
            if criteria["type"] == "range":
                if not (criteria["min"] <= val <= criteria["max"]):
                    matches = False
                    break
            elif criteria["type"] == "exact":
                if val != criteria["value"]:
                    matches = False
                    break
                    
        return matches
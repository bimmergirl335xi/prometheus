import heapq
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, TYPE_CHECKING
from config import *
from definitions import *
from memory import LongTermMemory
from utils import log, log_engine_event

if TYPE_CHECKING:
    from abstraction import AbstractionEngine

class ReasoningEngine:
    """
    The logic core. Uses graph traversal and pattern matching to 
    derive new information without an LLM.
    """
    def __init__(self, memory: LongTermMemory, abstraction_engine: Any):
        self.memory = memory
        self.abstraction_engine = abstraction_engine

    def deduce_relation(self, concept_a: str, concept_b: str, max_depth: int = 2) -> Tuple[float, List[str]]:
        """
        Transitive Inference: Finds a logical path between two concepts.
        If A is related to X, and X is related to B, then A is related to B.
        Returns (confidence_score, path_trace).
        """
        # Bidirectional BFS (Breadth-First Search) for pathfinding
        # We search the Semantic Memory association matrix
        
        start_node = self.memory.semantic._get_or_create_idx(concept_a)
        end_node = self.memory.semantic._get_or_create_idx(concept_b)
            
        if start_node == end_node:
            return 1.0, [concept_a]

        # Queue: (current_node, path_list, current_confidence)
        queue = [(start_node, [concept_a], 1.0)]
        visited = set()
        best_path = []
        best_score = 0.0

        while queue:
            current, path, score = queue.pop(0)
            
            if len(path) > max_depth + 1:
                continue
            
            if current == end_node:
                if score > best_score:
                    best_score = score
                    best_path = path
                continue

            if current in visited:
                continue
            visited.add(current)

            # Get neighbors from semantic memory (strongest associations only)
            # We access the raw matrix for speed
            row = self.memory.semantic.association_matrix[current]
            if GPU_AVAILABLE:
                row = xp.asnumpy(row)
            
            # Filter for meaningful connections (> 0.2)
            neighbors = np.where(row > 0.2)[0]
            
            for neighbor in neighbors:
                strength = float(row[neighbor])
                neighbor_name = self.memory.semantic.idx_to_concept[neighbor]
                
                # Decay score with each step (transitive property is weaker than direct)
                new_score = score * strength * 0.8
                queue.append((neighbor, path + [neighbor_name], new_score))
        
            if best_score > 0.0:
                log_engine_event(
                "Reasoning", 
                "DeduceRelation", 
                f"{concept_a} -> {concept_b}", 
                f"Path: {best_path} (Conf: {best_score:.2f})"
            )

        return best_score, best_path
    
    def validate_proposition(self, sentence: str) -> Tuple[bool, str]:
        """
        'Sanity Check': Breaks a sentence into Subject-Verb-Object 
        and checks if it contradicts known Semantic Memory facts.
        """
        # 1. Quick Keyword Extraction (Heuristic Parsing)
        # We look for "X is Y" or "X has Y" patterns
        words = sentence.upper().replace('.', '').split()
        
        # Simple relation extraction
        if " IS " in sentence.upper():
            parts = sentence.upper().split(" IS ")
            subj = parts[0].strip().split()[-1] # Last word of subject phrase
            obj = parts[1].strip().split()[0]   # First word of object phrase
            
            # Check 1: Do we know this connection?
            known_obj = self.memory.semantic.query_fact(subj, "IS_A")
            if known_obj and known_obj != obj:
                # Contradiction detected?
                # e.g., We know "SKY IS BLUE", text says "SKY IS GREEN"
                similarity = self.memory.semantic.get_similarity(known_obj, obj)
                if similarity < 0.3:
                    return False, f"Conflict: I thought {subj} was {known_obj}, but text says {obj}."
        
        # Check 2: Conceptual Similarity
        # If words in the sentence are usually totally unrelated, flag it.
        # e.g., "The algorithm ate the sandwich."
        for i in range(len(words)-1):
            w1, w2 = words[i], words[i+1]
            # If both are nouns (heuristic check), do they belong together?
            # (Skipping stop words would be better here, but this is a prototype)
            pass

        return True, "Proposition seems plausible."

    def solve_by_analogy(self, current_context: Dict[str, Any]) -> Optional[Action]:
        """
        Analogical Reasoning: 
        1. Look for a past success in Episodic Memory.
        2. If exact match fails, look for a *structural* match.
        3. Adapt the old solution to the new context.
        """
        episodes = self.memory.episodic.get_successes(n=100)
        best_score = 0.0
        best_episode = None

        # Extract keywords from current context
        current_keys = set(current_context.keys())
        current_values = set(str(v) for v in current_context.values())

        for ep in episodes:
            score = 0.0
            ep_keys = set(ep.context.keys())
            
            # Structural Similarity: Do they have the same *kinds* of parameters?
            # e.g., both have "target_location" even if values differ.
            common_keys = current_keys.intersection(ep_keys)
            score += len(common_keys) * 1.0
            
            # Content Similarity: Do they involve related concepts?
            # Here we check if context values are semantically close
            for v1 in current_values:
                for v2 in ep.context.values():
                    v2_str = str(v2)
                    if v1 == v2_str:
                        score += 2.0 # Exact match
                    else:
                        # Check semantic distance (using the new vector memory from previous turn)
                        sim = self.memory.semantic.get_similarity(v1, v2_str)
                        if sim > 0.6:
                            score += sim
            
            if score > best_score:
                best_score = score
                best_episode = ep

        if best_episode and best_score > 2.0:
            # We found a strong analogy!
            # Now we must ADAPT the action. 
            # If the old action used a parameter that matched the old context,
            # swap it for the new context's parameter.
            
            old_action = best_episode.action
            new_payload = old_action.payload
            
            # Simple heuristic adaptation: 
            # If the old payload was a value in the old context, replace it with the new context value
            for k, v in best_episode.context.items():
                if str(v) == str(old_action.payload) and k in current_context:
                    new_payload = current_context[k]
                    break
            
            return Action(
                action_type=old_action.action_type,
                payload=new_payload,
                metadata={"reasoning": "analogy", "source_episode": best_episode.timestamp},
                emergent=True
            )
            
        if best_episode is None:
            # Logic failed → try Abstraction rules as a trigger for deeper thinking
            for name in self.abstraction_engine.abstractions:
                if self.abstraction_engine.apply_abstraction(name, current_context):
                    rule = self.abstraction_engine.abstractions[name]
                    return Action(
                        action_type=ActionType.RECURSIVE_THINK,
                        payload={"hypothesis": rule.name, "rule_pattern": rule.pattern, "depth": 0},
                        emergent=True
                    )

        return None

    def hypothesize_cause(self, effect_concept: str) -> List[Tuple[str, float]]:
        """
        Abductive Reasoning:
        Given an effect, guess the cause based on Causal Model history.
        """
        # This requires reverse-lookup on the CausalModel
        # Since CausalModel stores Action -> Success/Fail, we look for actions
        # that strongly correlate with this concept appearing in successful contexts.
        
        candidates = []
        
        # Scan recent successful episodes
        successes = self.memory.episodic.get_successes(n=50)
        for ep in successes:
            # Did the effect appear in the outcome or context?
            # (Assuming you record post-state in context or have a way to track it)
            if effect_concept in str(ep.context) or effect_concept in str(ep.action.payload):
                candidates.append(ep.action.action_type.name)
        
        # Count frequency
        from collections import Counter
        counts = Counter(candidates)
        total = sum(counts.values())
        
        return [(action, count/total) for action, count in counts.most_common(3)]
    
    # [Add to ReasoningEngine class]
    def hybrid_solve(self, goal_context: Dict[str, Any]) -> Optional[Action]:
        """
        System 1 (Intuition) + System 2 (Logic).
        Uses Vector Memory to guess candidates, then Causal Model to verify them.
        """
        # --- SYSTEM 1: INTUITION (Fast) ---
        # "What implies success in this context?"
        # We query the vector space for concepts associated with the goal keys
        candidates = []
        for key in goal_context:
            # intuitive_leaps = get related concepts
            associations = self.memory.semantic.get_associated(key, top_k=5)
            candidates.extend([a[0] for a in associations])
        
        if not candidates:
            return None
            
        # --- SYSTEM 2: LOGIC (Slow) ---
        # "Does this candidate actually cause the desired outcome?"
        best_action = None
        best_score = -1.0
        
        for concept in set(candidates):
            # Check if this concept maps to a known ActionType
            try:
                # We try to map the concept string to an ActionType name
                # (e.g., "RESEARCH" -> ActionType.RESEARCH)
                action_type = ActionType[concept.upper()]
                
                # Verify with Causal Model
                causal_score = self.memory.causal.predict_success(action_type, goal_context)
                
                if causal_score > best_score:
                    best_score = causal_score
                    best_action = Action(action_type, payload=f"Hybrid solution: {concept}")
                    
            except KeyError:
                continue # Concept wasn't a valid action type
                
        if best_action and best_score > 0.3:
            return best_action
            
        return None
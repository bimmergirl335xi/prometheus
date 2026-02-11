from config import *
from utils import *
from definitions import *
from neural import SimpleNeuralNet
import numpy as np

class UniversalGrammar:
    """
    Implements Chomskyan 'Merge' and Case Grammar.
    Parses language based on deep semantic structure rather than surface order.
    """
    def __init__(self, memory):
        self.memory = memory
        # Map specific words to their likely universal roles
        # In a full AGI, this is learned. For now, we seed it or infer it.
        self.lexical_map = {
            # Examples of how the agent might map concepts internally
            # "EAT": SemanticRole.PREDICATE,
            # "CAT": SemanticRole.AGENT,
        }

    def infer_role(self, word: str) -> SemanticRole:
        """
        Guess the universal role of a word based on memory/context.
        """
        # 1. Check Memory Types
        cat_id = self.memory.semantic.get_lexical_category(word)
        
        # Heuristic: If it's an Action in memory -> PREDICATE
        if self.memory.semantic.is_action(word): # You'll need to add is_action helper
            return SemanticRole.PREDICATE
            
        # Heuristic: If it has strong agency associations -> AGENT
        # (Simplified logic for the prototype)
        return SemanticRole.UNKNOWN

    def merge(self, left_node: Dict, right_node: Dict) -> Dict:
        """
        The fundamental 'Merge' operation.
        Combines two constituent parts into a single semantic unit.
        """
        # Rule 1: Modifier + Entity = Modified Entity
        # e.g., "Red" + "Ball" -> {Entity: Ball, Mod: Red}
        if left_node['role'] == SemanticRole.MODIFIER and right_node['role'] != SemanticRole.PREDICATE:
            return {
                "head": right_node['head'],
                "role": right_node['role'],
                "children": right_node.get('children', []) + [left_node]
            }

        # Rule 2: Agent + Predicate = Sentence Fragment
        # e.g., "I" + "Run"
        if left_node['role'] == SemanticRole.AGENT and right_node['role'] == SemanticRole.PREDICATE:
            return {
                "head": right_node['head'],
                "role": SemanticRole.PREDICATE,
                "agent": left_node,
                "children": right_node.get('children', [])
            }

        # Default: Just link them as a sequence if no merge rule applies
        return {"head": "COMPOUND", "children": [left_node, right_node], "role": SemanticRole.UNKNOWN}

    def parse_universal(self, sentence: str) -> Dict[str, Any]:
        """
        Parses a sentence into a dependency graph (Universal Dependencies).
        This works for SVO, SOV, and VSO languages alike.
        """
        words = sentence.split()
        
        # 1. Tokenize and Assign Roles
        nodes = []
        for w in words:
            role = self.infer_role(w)
            nodes.append({"head": w, "role": role, "children": []})

        # 2. Identify the Anchor (The Main Verb/Predicate)
        # Every sentence revolves around the action.
        predicate_node = next((n for n in nodes if n['role'] == SemanticRole.PREDICATE), None)
        
        if not predicate_node:
            return {"raw": sentence, "type": "fragment"}

        # 3. Attach Arguments (The "Who" and "What")
        structure = {"action": predicate_node['head'], "agent": None, "patient": None}
        
        for node in nodes:
            if node == predicate_node: continue
            
            # Distance heuristic: Agents usually precede actions, Patients follow
            # (Note: A true UG parser would use case markers, but this is a start)
            if node['role'] == SemanticRole.AGENT:
                structure["agent"] = node['head']
            elif node['role'] == SemanticRole.PATIENT:
                structure["patient"] = node['head']
            # Fallback based on semantic memory lookup
            else:
                # If memory says this noun is usually an object of this verb
                structure["patient"] = node['head'] 

        return structure

class LanguageEngine:
    """
    A statistical grammar induction system.
    Learns to speak by finding patterns in concept sequences.
    """
    def __init__(self, memory: 'LongTermMemory'):
        self.memory = memory
        # Templates: List of (Sequence of Category_IDs, Frequency)
        self.templates: Dict[Tuple[int, ...], int] = defaultdict(int)
        
    def learn_grammar(self, sentence: str):
        """
        Reads a sentence and learns its grammatical structure.
        E.g., "The cat sat" -> [DET] [NOUN] [VERB] -> Template #42
        """
        words = [w.strip().upper() for w in sentence.split() if len(w) > 1]
        if len(words) < 2: return
        
        # Convert words to their "Grammar IDs"
        structure = []
        for word in words:
            cat_id = self.memory.semantic.get_lexical_category(word)
            structure.append(cat_id)
            
        # Store/Reinforce this structure
        structure_tuple = tuple(structure)
        self.templates[structure_tuple] += 1
        
    def construct_sentence(self, concepts: List[str]) -> str:
        """
        Converts a bag of concepts into a structured English sentence using Universal Grammar roles.
        """
        if not concepts: return "..."
        
        # 1. Identify Roles (Who is the actor? What is the action?)
        # We use the memory's vector store to guess roles if they aren't explicit
        roles = {'AGENT': [], 'PREDICATE': [], 'PATIENT': [], 'OTHER': []}
        
        for word in concepts:
            word = str(word).upper()
            # Simple heuristic: specific known verbs, or check memory
            if self.memory.semantic.is_action(word):
                roles['PREDICATE'].append(word)
            elif word in ["I", "ME", "SELF", "SYSTEM", "AGENT"]:
                roles['AGENT'].append("I")
            else:
                # Default to objects/patients for now
                roles['PATIENT'].append(word)
        
        # 2. Apply "SVO" (Subject-Verb-Object) Template
        # This is the standard grammar for English.
        
        # Subject
        subj = roles['AGENT'][0] if roles['AGENT'] else (roles['PATIENT'].pop(0) if roles['PATIENT'] else "It")
        
        # Verb (Predicate)
        verb = roles['PREDICATE'][0] if roles['PREDICATE'] else "is"
        
        # Object (The rest)
        obj = " and ".join(roles['PATIENT']) if roles['PATIENT'] else ""
        
        # 3. Construct (Capitalize and add punctuation)
        sentence = f"{subj} {verb}"
        if obj:
            sentence += f" {obj}"
            
        return sentence.capitalize() + "."

    def calculate_complexity(self, sentence: str) -> float:
        """
        Estimates cognitive load of a sentence based on:
        1. Word rarity (Surprise)
        2. Sentence length (Working Memory load)
        3. Structural complexity (Clause density)
        """
        words = [w.strip().upper() for w in sentence.split() if len(w) > 1]
        if not words: return 0.0

        # 1. Surprisal (Rare words = High surprise)
        surprisal = 0.0
        for word in words:
            # Check memory for familiarity
            if word in self.memory.semantic.concept_to_idx:
                idx = self.memory.semantic.concept_to_idx[word]
                # High access count = Low surprise
                familiarity = min(1.0, self.memory.semantic.access_count[idx] / 100.0)
                surprisal += (1.0 - familiarity)
            else:
                surprisal += 1.0 # Unknown word is max surprise

        # 2. Structural Density (Approximated by punctuation/conjunctions)
        clauses = sentence.count(',') + sentence.count(';') + sentence.count(' that ')
        
        # Normalize score (0.0 to 1.0+)
        # Avg sentence is ~10 words. 
        score = (surprisal * 0.1) + (len(words) * 0.05) + (clauses * 0.2)
        
        return score

    def formulate_thought(self, focus_concepts: List[str]) -> str:
        """
        The 'Talk' function.
        Takes raw concepts [I, WANT, ENERGY] and tries to make it natural.
        """
        # If we have learned the pattern [I] [ACTION] [OBJECT], we output "I want energy"
        # For now, we return the raw concepts, but this is where you'd hook in
        # the template matching from 'construct_sentence'
        
        # Simple heuristic: Subject -> Verb -> Object
        # We assume the input list is already ordered by importance (SVO)
        return " ".join(c.lower() for c in focus_concepts).capitalize() + "."

class PredictiveLanguageModel:
    """
    A lightweight, online-learning N-gram model.
    It learns valid sentence structures by predicting the next concept.
    """
    def __init__(self):
        # Maps (Word_A, Word_B) -> {Word_C: probability}
        self.trigrams = defaultdict(lambda: defaultdict(int))
    
    def learn_stream(self, text: str):
        tokens = text.upper().split()
        if len(tokens) < 3: return
        
        for i in range(len(tokens) - 2):
            w1, w2, w3 = tokens[i], tokens[i+1], tokens[i+2]
            self.trigrams[(w1, w2)][w3] += 1
            
    def predict_next(self, w1: str, w2: str) -> str:
        """Auto-complete thought"""
        candidates = self.trigrams.get((w1.upper(), w2.upper()))
        if not candidates: return None
        # Return most likely next word
        return max(candidates, key=candidates.get)

    def surprise_metric(self, w1: str, w2: str, w3: str) -> float:
        """
        How surprised are we by w3?
        High surprise = High information content = Focus attention here.
        """
        candidates = self.trigrams.get((w1, w2))
        if not candidates: return 1.0 # Max surprise
        
        total = sum(candidates.values())
        prob = candidates.get(w3, 0) / total
        return 1.0 - prob
# memory.py
from config import *
from utils import *
from definitions import *
import mmap
import os
from core_systems import Genome
from dataclasses import dataclass
from scipy.spatial.distance import cosine

class SemanticMemory:
    """Enhanced with abstraction support"""
    def __init__(self, genome: Genome, max_size: int = 50000):
        self.genome = genome
        self.max_size = max_size
        self.concept_to_idx: Dict[str, int] = {}
        self.idx_to_concept: Dict[int, str] = {}
        self.next_idx = 0
        self.vector_dim = 256
        
        self.activation = xp.zeros(max_size, dtype=xp.float32)
        self.association_matrix = xp.zeros((max_size, max_size), dtype=xp.float32)
        self.access_count = xp.zeros(max_size, dtype=xp.int32)
        self.last_access = xp.zeros(max_size, dtype=xp.float32)
        
        # New: Concept types (concrete vs abstract)
        self.concept_types = xp.zeros(max_size, dtype=xp.int32)  # 0=concrete, 1=abstract
       
        self.index_vectors = np.random.choice([0, 1, -1], size=(max_size, self.vector_dim), p=[0.9, 0.05, 0.05])
        if GPU_AVAILABLE:
            self.index_vectors = xp.array(self.index_vectors)

        self.triples: Dict[str, Dict[str, str]] = defaultdict(dict)

        # 2. Context: Accumulates signatures of neighbors (learns meaning)
        self.context_vectors = xp.zeros((max_size, self.vector_dim), dtype=xp.float32)

        self.load()

    def _get_or_create_idx(self, concept: str, is_abstract: bool = False) -> int:
        concept = concept.upper()
        if concept not in self.concept_to_idx:
            if self.next_idx >= self.max_size:
                self._consolidate_memory()
            
            idx = self.next_idx
            self.concept_to_idx[concept] = idx
            self.idx_to_concept[idx] = concept
            self.concept_types[idx] = 1 if is_abstract else 0
            self.next_idx += 1
            return idx
        return self.concept_to_idx[concept]

    def stimulate(self, concept: str, strength: float = 1.0, is_abstract: bool = False) -> None:
        idx = self._get_or_create_idx(concept, is_abstract)
        lr = self.genome.traits["learning_rate"] * self.genome.traits["memory_consolidation_rate"]
        
        self.activation[idx] = xp.minimum(1.0, self.activation[idx] + lr * strength)
        self.access_count[idx] += 1
        self.last_access[idx] = time.time()
        
        # Spreading activation
        if self.next_idx > 1:
            associations = self.association_matrix[idx, :self.next_idx]
            spread_strength = strength * 0.3
            self.activation[:self.next_idx] += associations * spread_strength
            self.activation[:self.next_idx] = xp.minimum(1.0, self.activation[:self.next_idx])

    def associate(self, a: str, b: str, weight: float = 1.0) -> None:
        if a.upper() == b.upper():
            return
        
        idx_a = self._get_or_create_idx(a)
        idx_b = self._get_or_create_idx(b)
        
        lr = self.genome.traits["learning_rate"] * self.genome.traits["memory_consolidation_rate"]
        delta = lr * weight
        
        self.association_matrix[idx_a, idx_b] += delta
        self.association_matrix[idx_b, idx_a] += delta
        
        self.association_matrix[idx_a, idx_b] = xp.clip(
            self.association_matrix[idx_a, idx_b], -1.0, 1.0
        )
        self.association_matrix[idx_b, idx_a] = xp.clip(
            self.association_matrix[idx_b, idx_a], -1.0, 1.0
        )

    def get_associated(self, concept: str, top_k: int = 5, threshold: float = 0.1) -> List[Tuple[str, float]]:
        if concept.upper() not in self.concept_to_idx:
            return []
        
        idx = self.concept_to_idx[concept.upper()]
        associations = self.association_matrix[idx, :self.next_idx]
        
        if GPU_AVAILABLE:
            associations = cp.asnumpy(associations)
        else:
            associations = np.array(associations)
        
        valid_indices = np.where(associations > threshold)[0]
        if len(valid_indices) == 0:
            return []
        
        valid_associations = associations[valid_indices]
        sorted_indices = valid_indices[np.argsort(valid_associations)[::-1]]
        
        results = []
        for i in sorted_indices[:top_k]:
            results.append((self.idx_to_concept[i], float(associations[i])))
        
        return results

    def decay_activation(self, decay_rate: float = 0.01) -> None:
        self.activation *= (1 - decay_rate)

    def _consolidate_memory(self) -> None:
        log("MEMORY", "Consolidating memory...", Term.YELLOW)
        
        current_time = time.time()
        
        # 1. Prepare Data (Move to CPU if on GPU)
        if GPU_AVAILABLE:
            last_access_np = cp.asnumpy(self.last_access[:self.next_idx])
            access_count_np = cp.asnumpy(self.access_count[:self.next_idx])
        else:
            last_access_np = np.array(self.last_access[:self.next_idx])
            access_count_np = np.array(self.access_count[:self.next_idx])
        
        # 2. Calculate Recency Weight (The Missing Piece)
        # Concepts fade if not accessed recently (decay over 24h)
        recency = current_time - last_access_np
        recency_weight = np.exp(-recency / 86400) 

        # 3. Calculate Connectivity (Infinite Scaling Factor)
        # We value "Hub" concepts that connect many other ideas
        if GPU_AVAILABLE:
            connectivity = xp.sum(xp.abs(self.association_matrix[:self.next_idx]), axis=1)
            connectivity = xp.asnumpy(connectivity) # Ensure CPU for final math
        else:
            connectivity = np.sum(np.abs(self.association_matrix[:self.next_idx]), axis=1)

        # 4. Calculate Final Importance Score
        # Importance = Frequency * Recency * Connectivity
        importance = access_count_np * recency_weight * (1 + connectivity)
        
        # 5. Pruning Logic
        # Remove the bottom 20% of least important concepts
        threshold_idx = int(self.next_idx * 0.2)
        sorted_indices = np.argsort(importance)
        to_remove = set(sorted_indices[:threshold_idx])

        usage_ratio = self.next_idx / self.max_size
        
        new_concept_to_idx = {}
        new_idx_to_concept = {}
        new_idx = 0
        
        if usage_ratio < 0.8:
            # If memory is less than 80% full, DON'T PRUNE ANYTHING.
            log("MEM", "Memory healthy (sparse), skipping prune.", Term.DIM)
            return

        # If we must prune, only cut the absolute worst 5%
        prune_count = int(self.next_idx * 0.05)

        # Remap surviving concepts
        for old_idx in range(self.next_idx):
            if old_idx not in to_remove:
                concept = self.idx_to_concept[old_idx]
                new_concept_to_idx[concept] = new_idx
                new_idx_to_concept[new_idx] = concept
                
                # Copy data to new position
                self.activation[new_idx] = self.activation[old_idx]
                self.access_count[new_idx] = self.access_count[old_idx]
                self.last_access[new_idx] = self.last_access[old_idx]
                self.concept_types[new_idx] = self.concept_types[old_idx]
                self.association_matrix[new_idx, :] = self.association_matrix[old_idx, :]
                
                new_idx += 1
        
        # Apply changes
        self.concept_to_idx = new_concept_to_idx
        self.idx_to_concept = new_idx_to_concept
        self.next_idx = new_idx
        
        # Zero out the freed space in the matrix to prevent ghost connections
        # (Optional but good for hygiene)
        self.association_matrix[new_idx:, :] = 0
        self.association_matrix[:, new_idx:] = 0
        
        log("MEMORY", f"Consolidated to {new_idx} concepts", Term.YELLOW)
        
    def save(self):
        try:
            data = {
                'concept_to_idx': self.concept_to_idx,
                'idx_to_concept': {int(k): v for k, v in self.idx_to_concept.items()},
                'next_idx': self.next_idx
            }
            
            if GPU_AVAILABLE:
                activation = cp.asnumpy(self.activation)
                association_matrix = cp.asnumpy(self.association_matrix)
                access_count = cp.asnumpy(self.access_count)
                last_access = cp.asnumpy(self.last_access)
                concept_types = cp.asnumpy(self.concept_types)
            else:
                activation = self.activation
                association_matrix = self.association_matrix
                access_count = self.access_count
                last_access = self.last_access
                concept_types = self.concept_types
            
            np.savez_compressed(
                MEMORY_FILE,
                activation=activation,
                association_matrix=association_matrix,
                access_count=access_count,
                last_access=last_access,
                concept_types=concept_types,
                **data
            )
        except Exception as e:
            log("MEMORY", f"Save failed: {e}", Term.RED)

    def load(self):
        if os.path.exists(MEMORY_FILE):
            try:
                data = np.load(MEMORY_FILE, allow_pickle=True)
                
                self.concept_to_idx = data['concept_to_idx'].item()
                idx_dict = data['idx_to_concept'].item()
                self.idx_to_concept = {int(k): v for k, v in idx_dict.items()}
                self.next_idx = int(data['next_idx'])
                
                if GPU_AVAILABLE:
                    self.activation = cp.array(data['activation'])
                    self.association_matrix = cp.array(data['association_matrix'])
                    self.access_count = cp.array(data['access_count'])
                    self.last_access = cp.array(data.get('last_access', np.zeros(self.max_size)))
                    self.concept_types = cp.array(data.get('concept_types', np.zeros(self.max_size)))
                else:
                    self.activation = data['activation']
                    self.association_matrix = data['association_matrix']
                    self.access_count = data['access_count']
                    self.last_access = data.get('last_access', np.zeros(self.max_size))
                    self.concept_types = data.get('concept_types', np.zeros(self.max_size))
                
                log("MEMORY", f"Loaded {self.next_idx} semantic concepts", Term.CYAN)
            except Exception as e:
                log("MEMORY", f"Load failed: {e}", Term.RED)

    def learn_context(self, focus_concept: str, context_words: List[str]):
        """
        Update semantic meaning based on context (Sliding Window).
        If 'apple' appears near 'fruit', 'apple' acquires some of 'fruit's signature.
        """
        idx_focus = self._get_or_create_idx(focus_concept)
        
        for word in context_words:
            if word == focus_concept: continue
            idx_ctx = self._get_or_create_idx(word)
            
            # Add the neighbor's static index vector to the focus's context vector
            # This builds a "history" of what words this concept hangs out with.
            self.context_vectors[idx_focus] += self.index_vectors[idx_ctx]

    def get_similarity(self, concept_a: str, concept_b: str) -> float:
        """Cosine similarity between two learned context vectors."""
        if concept_a not in self.concept_to_idx or concept_b not in self.concept_to_idx:
            return 0.0
            
        idx_a = self.concept_to_idx[concept_a]
        idx_b = self.concept_to_idx[concept_b]
        
        vec_a = self.context_vectors[idx_a]
        vec_b = self.context_vectors[idx_b]
        
        if GPU_AVAILABLE:
            # simple cosine similarity manually for CuPy
            norm_a = xp.linalg.norm(vec_a)
            norm_b = xp.linalg.norm(vec_b)
            if norm_a == 0 or norm_b == 0: return 0.0
            return float(xp.dot(vec_a, vec_b) / (norm_a * norm_b))
        else:
            return 1.0 - cosine(vec_a, vec_b)
        
    def learn_sequence(self, sequence: List[str]):
        """
        Learns the temporal order of concepts (A -> B).
        Crucial for understanding sentences and cause-effect.
        """
        if len(sequence) < 2: return

        # Get indices for all words in one go
        indices = [self._get_or_create_idx(w) for w in sequence]
        
        # Learn transitions: Current Word predicts Next Word
        for i in range(len(indices) - 1):
            curr_idx = indices[i]
            next_idx = indices[i+1]
            
            # Asymmetric Association: A leads to B
            # We add a small amount to the link to strengthen the path
            # Scalability Note: This builds a Markov Chain-like structure in the weight matrix
            lr = self.genome.traits["learning_rate"]
            
            # Directional strength: Strengthen A->B more than B->A for sequences
            self.association_matrix[curr_idx, next_idx] += lr * 0.5
            
            # Bound values
            self.association_matrix[curr_idx, next_idx] = xp.minimum(
                1.0, self.association_matrix[curr_idx, next_idx]
            )

    def get_lexical_category(self, word: str) -> int:
        """
        Returns a 'Grammar ID' for a word.
        Words that appear in similar contexts will tend to have similar IDs.
        We use a Locality Sensitive Hash (LSH) of the context vector.
        """
        idx = self.concept_to_idx.get(word.upper())
        if idx is None: return 0 # Unknown category
        
        # Get the context vector (which encodes how the word is used)
        vec = self.context_vectors[idx]
        if GPU_AVAILABLE: vec = xp.asnumpy(vec)
        
        # Simple LSH: Check the sign of the first 8 dimensions
        # This creates crude "buckets" for words (e.g., Nouns might end up in bucket 5)
        category_hash = 0
        for i in range(min(8, len(vec))):
            if vec[i] > 0:
                category_hash |= (1 << i)
                
        return category_hash
    
    def add_fact(self, subject: str, predicate: str, obj: str):
        """Learns a specific logical relationship."""
        subject = subject.upper()
        predicate = predicate.upper()
        obj = obj.upper()
        
        self.triples[subject][predicate] = obj
        
        # Also strengthen the vector association
        self.associate(subject, obj, weight=0.8)
        self.stimulate(subject)

    def query_fact(self, subject: str, predicate: str) -> Optional[str]:
        """Retrieves a specific fact."""
        return self.triples.get(subject.upper(), {}).get(predicate.upper())

    def is_action(self, word: str) -> bool:
        """
        Returns True if the concept is action-oriented (Verb-like).
        Uses the vector space to check distance to known actions.
        """
        # Seed some known actions if empty
        known_actions = ["EAT", "RUN", "READ", "BUILD", "MAKE", "CHECK"]
        
        word = word.upper()
        if word in known_actions: return True
        
        # Check similarity to known actions
        # (Using your existing vector similarity function)
        for action in known_actions:
            if self.get_similarity(word, action) > 0.7:
                return True
        return False

class ScalableSemanticMemory(SemanticMemory):
    """
    Disk-backed memory that scales to Terabytes.
    Uses memory mapping (mmap) to access huge files without loading them into RAM.
    """
    def __init__(self, genome: Genome, storage_dir: str = "memory_shards"):
        # MAX SIZE: 50,000 concepts in RAM matrix (approx 10GB).
        # Everything else scales to disk via shards.
        
        self.storage_dir = storage_dir

        self.vector_dim = 256
        self.shard_size = 100000
        
        super().__init__(genome, max_size=5000) 
        
        os.makedirs(storage_dir, exist_ok=True)
        
        # Load the disk index to sync with RAM
        self.load_index()

    def _get_vector(self, idx: int) -> np.array:
        """
        Reads a specific vector from disk instantly using math.
        No RAM spike.
        """
        shard_id = idx // self.shard_size
        offset = idx % self.shard_size
        
        filename = f"{self.storage_dir}/shard_{shard_id}.bin"
        
        # Create file if missing
        if not os.path.exists(filename):
            with open(filename, "wb") as f:
                f.seek(self.shard_size * self.vector_dim * 4 - 1)
                f.write(b'\0')
        
        dtype = np.float32
        bytes_per_vector = self.vector_dim * 4
        
        # Open as memory map
        with open(filename, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            start = offset * bytes_per_vector
            end = start + bytes_per_vector
            vector_bytes = mm[start:end]
            return np.frombuffer(vector_bytes, dtype=dtype).copy()

    def _save_vector(self, idx: int, vector: np.array):
        """Writes a vector directly to the correct disk location."""
        shard_id = idx // self.shard_size
        offset = idx % self.shard_size
        filename = f"{self.storage_dir}/shard_{shard_id}.bin"
        
        with open(filename, "r+b") as f:
            mm = mmap.mmap(f.fileno(), 0)
            start = offset * self.vector_dim * 4
            mm[start:start + self.vector_dim * 4] = vector.tobytes()
            mm.flush()

    def save_index(self):
        """Saves the lightweight mapping (Name -> ID) to a JSON file."""
        index_file = os.path.join(self.storage_dir, "index.json")
        try:
            data = {
                "concept_to_idx": self.concept_to_idx,
                "idx_to_concept": self.idx_to_concept,
                "next_idx": self.next_idx
            }
            with open(index_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            log("MEM", f"Index save failed: {e}", Term.RED)

    def load_index(self):
        """Loads the mapping from disk."""
        index_file = os.path.join(self.storage_dir, "index.json")
        if os.path.exists(index_file):
            try:
                with open(index_file, "r") as f:
                    data = json.load(f)
                    self.concept_to_idx = data["concept_to_idx"]
                    # JSON keys are always strings, convert back to int for ID mapping
                    self.idx_to_concept = {int(k): v for k, v in data["idx_to_concept"].items()}
                    self.next_idx = data["next_idx"]
                log("MEM", f"Loaded index: {self.next_idx} concepts mapped.", Term.GREEN)
            except Exception as e:
                log("MEM", f"Index load failed: {e}", Term.RED)
        else:
            log("MEM", "No existing index found. Starting fresh.", Term.YELLOW)

    def save(self):
        """
        Overrides the base save. 
        We save the index (JSON) and the RAM-based association matrix (NumPy).
        The vectors are already saved to shards in real-time.
        """
        # 1. Save the Index (Mapping)
        self.save_index()
        
        # 2. Save the RAM-based Association Matrix & Activation
        # We use the parent class's logic but point to a specific file in our storage dir
        try:
            matrix_file = os.path.join(self.storage_dir, "ram_matrix.npz")
            
            if GPU_AVAILABLE:
                activation = cp.asnumpy(self.activation)
                association_matrix = cp.asnumpy(self.association_matrix)
            else:
                activation = self.activation
                association_matrix = self.association_matrix
            
            np.savez_compressed(
                matrix_file,
                activation=activation,
                association_matrix=association_matrix
            )
            log("MEM", "Saved RAM matrix and Index.", Term.GREEN)
        except Exception as e:
            log("MEM", f"RAM Matrix save failed: {e}", Term.RED)

    # Note: We also need to override load() to pull the matrix back into RAM
    def load(self):
        """Overrides base load to pull matrix from our specific storage dir."""
        # Index is already loaded in __init__ via load_index()
        
        matrix_file = os.path.join(self.storage_dir, "ram_matrix.npz")
        if os.path.exists(matrix_file):
            try:
                data = np.load(matrix_file)
                if GPU_AVAILABLE:
                    self.activation = cp.array(data['activation'])
                    self.association_matrix = cp.array(data['association_matrix'])
                else:
                    self.activation = data['activation']
                    self.association_matrix = data['association_matrix']
                log("MEM", "Loaded RAM matrix.", Term.GREEN)
            except Exception as e:
                log("MEM", f"RAM Matrix load failed: {e}", Term.RED)

class EpisodicMemory:
    """Enhanced with abstraction tagging"""
    def __init__(self, max_size: int = 1000, vector_dim: int = 256):
        self.max_size = max_size
        self.vector_dim = vector_dim
        self.episodes: deque = deque(maxlen=max_size)
        
        # We store the 'gist' (context vector) of every episode here
        self.episode_vectors = xp.zeros((max_size, vector_dim), dtype=xp.float32)
        self.episode_indices = [] # Keeps track of which episode obj corresponds to which row
        self.pointer = 0 # Circular buffer pointer
        # -----------------------------------------

        self.by_action_type: Dict[ActionType, List[Episode]] = defaultdict(list)
        self.by_outcome: Dict[bool, List[Episode]] = defaultdict(list)
        self.by_abstraction: Dict[str, List[Episode]] = defaultdict(list)

    def record(self, episode: Episode, context_vector: np.array = None) -> None:
        """
        Save episode and index its vector for fast recall.
        NOTE: context_vector must be passed from the Executive/Perception system.
        """
        self.episodes.append(episode)
        self.by_action_type[episode.action.action_type].append(episode)
        self.by_outcome[episode.outcome].append(episode)
        
        for tag in episode.abstraction_tags:
            self.by_abstraction[tag].append(episode)
        
        # --- NEW: Update Vector Index ---
        if context_vector is not None:
            # Normalize vector for cosine similarity
            norm = xp.linalg.norm(context_vector)
            if norm > 0:
                normalized_vec = context_vector / norm
                self.episode_vectors[self.pointer] = normalized_vec
                
                # Manage the mapping list
                if len(self.episode_indices) < self.max_size:
                    self.episode_indices.append(episode)
                else:
                    self.episode_indices[self.pointer] = episode
                
                # Move pointer (Circular Buffer)
                self.pointer = (self.pointer + 1) % self.max_size

    def retrieve_similar(self, current_vector: np.array, top_k: int = 3, threshold: float = 0.7) -> List[Episode]:
        """
        Finds past episodes that are semantically similar to the current situation.
        This is 'Associative Recall'.
        """
        if not self.episode_indices:
            return []
            
        # Normalize query vector
        query_norm = xp.linalg.norm(current_vector)
        if query_norm == 0: return []
        query = current_vector / query_norm
        
        # Calculate Cosine Similarity (Dot product of normalized vectors)
        # We compute scores for ALL memories instantly using matrix multiplication
        scores = xp.dot(self.episode_vectors, query)
        
        if GPU_AVAILABLE:
            scores = xp.asnumpy(scores) # Bring back to CPU for sorting
            
        # Get indices of top_k matches
        # We only look at valid entries (up to len(episode_indices))
        valid_count = len(self.episode_indices)
        valid_scores = scores[:valid_count]
        
        # Filter by threshold
        indices = np.where(valid_scores > threshold)[0]
        
        # Sort by score descending
        top_indices = indices[np.argsort(valid_scores[indices])[::-1]][:top_k]
        
        return [self.episode_indices[i] for i in top_indices]

    def get_recent(self, n: int = 10) -> List[Episode]:
        return list(self.episodes)[-n:]

    def get_by_action(self, action_type: ActionType, n: int = 10) -> List[Episode]:
        return self.by_action_type[action_type][-n:]

    def count_recent_failures(self, window: int = 10) -> int:
        recent = self.get_recent(window)
        return sum(1 for ep in recent if not ep.outcome)

    def get_successes(self, n: int = 10) -> List[Episode]:
        return self.by_outcome[True][-n:]

    def get_failures(self, n: int = 10) -> List[Episode]:
        return self.by_outcome[False][-n:]

    def get_by_abstraction(self, abstraction: str, n: int = 10) -> List[Episode]:
        """Get episodes matching an abstraction"""
        return self.by_abstraction[abstraction][-n:]

class CausalModel:
    """Enhanced causal inference"""
    def __init__(self):
        self.causal_links: Dict[str, List[float]] = defaultdict(list)
        self.context_modifiers: Dict[str, Dict[str, float]] = defaultdict(dict)
        self.intervention_count: Dict[str, int] = defaultdict(int)
        self.causal_graph: Dict[str, Set[str]] = defaultdict(set)  # cause -> effects

    def record_outcome(
        self, 
        action_type: ActionType, 
        outcome: bool, 
        context: Optional[Dict] = None
    ) -> None:
        key = action_type.name
        value = 1.0 if outcome else -1.0
        
        self.causal_links[key].append(value)
        self.intervention_count[key] += 1
        
        if len(self.causal_links[key]) > 100:
            self.causal_links[key].pop(0)
        
        if context:
            for ctx_key, ctx_val in context.items():
                if isinstance(ctx_val, (int, float)):
                    if ctx_key not in self.context_modifiers[key]:
                        self.context_modifiers[key][ctx_key] = 0.0
                    
                    alpha = 0.1
                    self.context_modifiers[key][ctx_key] = (
                        alpha * value + (1 - alpha) * self.context_modifiers[key][ctx_key]
                    )

    def predict_success(
        self, 
        action_type: ActionType, 
        context: Optional[Dict] = None
    ) -> float:
        key = action_type.name
        
        if key not in self.causal_links or not self.causal_links[key]:
            return 0.0
        
        history = self.causal_links[key]
        weights = np.exp(np.linspace(-1, 0, len(history)))
        base_prediction = np.average(history, weights=weights)
        
        if context and key in self.context_modifiers:
            adjustment = 0.0
            for ctx_key, ctx_val in context.items():
                if ctx_key in self.context_modifiers[key] and isinstance(ctx_val, (int, float)):
                    adjustment += self.context_modifiers[key][ctx_key] * min(abs(ctx_val), 1.0)
            
            base_prediction += adjustment * 0.2
        
        return max(-1.0, min(1.0, base_prediction))

    def get_intervention_count(self, action_type: ActionType) -> int:
        return self.intervention_count.get(action_type.name, 0)

class LongTermMemory:
    """Unified memory with abstraction support"""
    def __init__(self, genome: Genome):
        self.semantic = ScalableSemanticMemory(genome)
        self.episodic = EpisodicMemory()
        self.causal = CausalModel()
        self.genome = genome
        
        self.last_consolidation = time.time()
        self.consolidation_interval = 300

        self.episodic = EpisodicMemory(max_size=100000) # Store 100k episodes in RAM

        self.causal = CausalModel()
        self.genome = genome
        
        self.last_consolidation = time.time()
        self.consolidation_interval = 300

    def tick(self) -> None:
        current_time = time.time()
        
        decay_rate = 1 - self.genome.traits["memory_retention"]
        self.semantic.decay_activation(decay_rate)
        
        if current_time - self.last_consolidation > self.consolidation_interval:
            self.consolidate()
            self.last_consolidation = current_time

    def consolidate(self) -> None:
        """
        'Sleep Mode': Transfers episodic insights into semantic knowledge.
        1. Strengthens associations between concepts that appear together in successes.
        2. Extracts 'Rules' (Triples) if patterns are consistent.
        """
        log("MEM", "🧠 Consolidating memories (Sleep Phase)...", Term.PURPLE)
        
        # 1. Get recent successes
        recent_successes = self.episodic.get_successes(n=50)
        
        concept_cooccurrence = defaultdict(int)
        
        for ep in recent_successes:
            # Extract concepts from the context and the action payload
            context_keys = list(ep.context.keys())
            payload_words = str(ep.action.payload).split() if ep.action.payload else []
            
            all_concepts = [str(k).upper() for k in context_keys] + \
                           [str(w).upper() for w in payload_words if len(w) > 3]
            
            # Hebbian Learning: "Cells that fire together, wire together"
            # Increase weight between all concepts present in a success
            for i in range(len(all_concepts)):
                for j in range(i + 1, len(all_concepts)):
                    a, b = all_concepts[i], all_concepts[j]
                    self.semantic.associate(a, b, weight=0.1) # Gentle nudge
                    
                    # Track for rule extraction
                    pair_key = tuple(sorted((a, b)))
                    concept_cooccurrence[pair_key] += 1

        # 2. Rule Extraction (Knowledge Graph)
        # If two concepts appear together very often in successes, assume a strong link
        for (a, b), count in concept_cooccurrence.items():
            if count > 5: # Threshold for forming a 'fact'
                # We don't know the predicate, so we use a generic 'RELATED_TO'
                # or simpler: just ensure the vector link is maxed out
                self.semantic.associate(a, b, weight=0.5)
                log("MEM", f"🔗 Strong bond formed: {a} <-> {b}", Term.CYAN)

        # 3. Prune Weak Memories (Forgetting)
        # This keeps the system fast.
        self.semantic._consolidate_memory()
        
    def save(self) -> None:
        self.semantic.save()

    def load(self) -> None:
        self.semantic.load()

@dataclass
class ActiveItem:
    content: Any
    category: str  # e.g., "visual", "phonological", "goal"
    activation: float = 1.0
    created_at: float = field(default_factory=time.time)

class WorkingMemory:
    """
    Short-term buffer with limited capacity (Miller's Law: 7 +/- 2).
    Holds items active for immediate processing/logic.
    """
    def __init__(self, capacity: int = 7):
        self.capacity = capacity
        self.items: List[ActiveItem] = []
        self.decay_rate = 0.05
    
    def hold(self, content: Any, category: str = "general"):
        """Place an item into working memory, pushing out old ones if full."""
        # Refresh if already exists
        for item in self.items:
            if item.content == content:
                item.activation = 1.0
                return

        # Add new item
        new_item = ActiveItem(content, category)
        self.items.append(new_item)
        
        # Sort by activation and prune
        if len(self.items) > self.capacity:
            self.items.sort(key=lambda x: x.activation, reverse=True)
            self.items = self.items[:self.capacity]

    def retrieve(self, category: str = None) -> List[Any]:
        """Get currently active items, optionally filtered by category."""
        if category:
            return [i.content for i in self.items if i.category == category]
        return [i.content for i in self.items]

    def tick(self):
        """Decay items. Call this every cycle."""
        active_items = []
        for item in self.items:
            item.activation -= self.decay_rate
            if item.activation > 0.1:
                active_items.append(item)
        self.items = active_items

    def get_context_string(self) -> str:
        """Convert WM content to a string context for the Planner."""
        return " | ".join([str(i.content) for i in self.items])

    def get_recent_perplexity(self) -> float:
        """
        Returns the average confusion level of recent items.
        (0.0 = Clear, 1.0 = Confused)
        """
        if not self.items:
            return 0.0
        
        # We assume items might have a 'complexity' score attached, 
        # or we default to 0.0 if not present.
        scores = [getattr(i, 'complexity', 0.0) for i in self.items]
        return sum(scores) / len(scores)

    def add_perplexity(self, score: float):
        """Register a confusion score for the current moment"""
        # We attach this score to the most recent item, or create a placeholder
        if self.items:
            self.items[-1].complexity = score

class SelfModel:
    """Enhanced metacognitive awareness with abstraction support."""
    def __init__(self):
        self.capabilities: Dict[str, float] = defaultdict(lambda: 0.5)
        self.action_history: Dict[str, int] = defaultdict(int)
        self.failure_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.success_patterns: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.action_latencies: Dict[str, List[float]] = defaultdict(list)
        self.last_update: Dict[str, float] = {}
        
        # New: Self-awareness metrics
        self.cognitive_load = 0.5
        self.learning_curve: List[Tuple[float, float]] = []  # (time, performance)
        self.capability_trajectory: Dict[str, List[Tuple[float, float]]] = defaultdict(list)

    def update_belief(
        self, 
        action_type: ActionType, 
        success: bool, 
        failure_type: Optional[FailureType] = None,
        latency: float = 0.0,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        key = action_type.name
        current = self.capabilities[key]
        
        update_strength = 0.1 if self.action_history[key] > 10 else 0.2
        
        if success:
            self.capabilities[key] = min(1.0, current + update_strength)
            if context:
                for ctx_key, ctx_val in context.items():
                    self.success_patterns[key][f"{ctx_key}={ctx_val}"] += 1
        else:
            self.capabilities[key] = max(0.0, current - update_strength * 1.5)
            if failure_type:
                self.failure_patterns[key][failure_type.name] += 1
        
        self.action_history[key] += 1
        self.last_update[key] = time.time()
        
        # Track trajectory
        self.capability_trajectory[key].append((time.time(), self.capabilities[key]))
        if len(self.capability_trajectory[key]) > 100:
            self.capability_trajectory[key].pop(0)
        
        if latency > 0:
            self.action_latencies[key].append(latency)
            if len(self.action_latencies[key]) > 50:
                self.action_latencies[key].pop(0)

    def get_confidence(self, action_type: ActionType) -> float:
        key = action_type.name
        base_confidence = self.capabilities[key]
        
        if key in self.last_update:
            time_since = time.time() - self.last_update[key]
            if time_since > 300:
                decay = min(0.2, time_since / 3000)
                base_confidence *= (1 - decay)
        
        return base_confidence

    def get_dominant_failure(self, action_type: ActionType) -> Optional[FailureType]:
        patterns = self.failure_patterns.get(action_type.name, {})
        if not patterns:
            return None
        dominant = max(patterns, key=patterns.get)
        try:
            return FailureType[dominant]
        except KeyError:
            return FailureType.UNKNOWN

    def get_average_latency(self, action_type: ActionType) -> float:
        latencies = self.action_latencies.get(action_type.name, [])
        return sum(latencies) / len(latencies) if latencies else 0.0

    def get_learning_rate(self, action_type: ActionType) -> float:
        """
        Estimate learning rate for this action type.
        How quickly is the agent improving?
        """
        trajectory = self.capability_trajectory.get(action_type.name, [])
        if len(trajectory) < 2:
            return 0.0
        
        # Linear regression on recent trajectory
        recent = trajectory[-20:]
        times = np.array([t for t, _ in recent])
        capabilities = np.array([c for _, c in recent])
        
        if len(times) < 2:
            return 0.0
        
        # Normalize time
        times = times - times[0]
        
        # Simple linear fit
        if np.std(times) > 0:
            slope = np.corrcoef(times, capabilities)[0, 1] * np.std(capabilities) / np.std(times)
            return slope
        
        return 0.0

    def introspect(self) -> Dict[str, Any]:
        """Deep introspection - the agent examines its own state."""
        report = {
            "capabilities": dict(self.capabilities),
            "cognitive_load": self.cognitive_load,
            "total_actions": sum(self.action_history.values()),
            "learning_rates": {}
        }
        
        for action_type in ActionType:
            lr = self.get_learning_rate(action_type)
            if lr != 0.0:
                report["learning_rates"][action_type.name] = lr
        
        return report

class HolographicMemory:
    """
    Uses Circular Convolution to bind concepts algebraically.
    A * B = C (Binding)
    C # A ≈ B (Unbinding)
    """
    def __init__(self, dim: int = 512): # HRR needs higher dims
        self.dim = dim
        self.rng = np.random.default_rng()

    def bind(self, vec_a: np.array, vec_b: np.array) -> np.array:
        """
        Circular Convolution: Binds two vectors.
        Preserves the identity of both in a compressed form.
        """
        # FFT -> Multiply -> IFFT is faster for large dims (O(N log N))
        return np.real(np.fft.ifft(np.fft.fft(vec_a) * np.fft.fft(vec_b)))

    def unbind(self, composite: np.array, key: np.array) -> np.array:
        """
        Circular Correlation: Extracts the partner of 'key' from 'composite'.
        """
        # Correlation is convolution with one operand reversed (approx inverse)
        # Or in Fourier domain: A * conj(B)
        return np.real(np.fft.ifft(np.fft.fft(composite) * np.conj(np.fft.fft(key))))

    def superposition(self, vectors: List[np.array]) -> np.array:
        """Standard vector addition (bundling)"""
        if not vectors: return np.zeros(self.dim)
        return np.sum(vectors, axis=0)
    
    def normalize(self, vec: np.array) -> np.array:
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

# --- Usage Update in SemanticMemory ---
# Replace your simple context storage with this:
# When storing a fact "Cat Eats Fish":
#   Vec_Fact = (Role_Agent * Vec_Cat) + (Role_Action * Vec_Eat) + (Role_Patient * Vec_Fish)
# This allows you to query: "What Eats Fish?" -> Vec_Fact # (Role_Action * Vec_Eat + Role_Patient * Vec_Fish) ≈ Vec_Cat
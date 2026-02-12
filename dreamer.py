# dreamer.py
import time
import random
import numpy as np
from typing import List, Dict, Any, Tuple
from config import *
from utils import log, Term
from definitions import Episode, Action, ActionType
from memory import LongTermMemory

class Dreamer:
    """
    Offline Simulation Engine.
    Replays episodic memories with counterfactual mutations to
    densify the Causal Model and Generalize skills.
    """
    def __init__(self, memory: LongTermMemory, genome: Any):
        self.memory = memory
        self.genome = genome
        self.active = False

    async def dream_cycle(self, duration: float = 0.5):
        """
        Run a simulation loop. 
        Selects high-surprise episodes and mutates them.
        """
        start_time = time.time()
        
        # 1. Select 'Interesting' Memories (High surprise or failure)
        # We look for episodes where prediction error was high
        candidates = self.memory.episodic.get_failures(n=20) + \
                     self.memory.episodic.get_recent(n=10)
        
        if not candidates:
            return

        episode = random.choice(candidates)
        
        # 2. Generate Counterfactual: "What if I did X instead?"
        # We assume the Context remains the same, but Action changes.
        alternative_action = self._mutate_action(episode.action)
        
        # 3. Predict Outcome using Causal Model
        # (We don't actually execute, we ask the model what IT thinks would happen)
        predicted_success = self.memory.causal.predict_success(
            alternative_action.action_type, 
            episode.context
        )
        
        # 4. Update Policy / Values based on this 'Dream'
        # If the model thinks this alternative would have worked better,
        # we strengthen the association for future planning.
        if predicted_success > 0.6 and not episode.outcome:
            log("DREAM", f"💭 Counterfactual: {alternative_action.action_type.name} might have fixed failure in Ep {episode.timestamp:.0f}", Term.PURPLE)
            
            # Synthesize a "Dream Episode" and store it as a lesson
            # We mark it as 'imagined' so we don't confuse it with reality
            dream_ep = Episode(
                timestamp=time.time(),
                action=alternative_action,
                context=episode.context,
                outcome=True, # We assume success for the sake of learning the path
                metadata={"source": "dream", "confidence": predicted_success}
            )
            # Store in a separate buffer or weighted lower in main memory
            self.memory.episodic.record(dream_ep)

    def _mutate_action(self, action: Action) -> Action:
        """Randomly swaps the action type or payload."""
        new_type = random.choice(list(ActionType))
        return Action(new_type, action.payload, emergent=True)
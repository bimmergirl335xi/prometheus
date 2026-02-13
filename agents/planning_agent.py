
class PlanningAgent:
    """
    Responsible for breaking down high-level goals into executable steps.
    Uses the causal model to predict outcomes.
    """
    def __init__(self, 
                 genome: Genome, 
                 memory: LongTermMemory, 
                 self_model: SelfModel, 
                 skill_library: SkillLibrary,
                 abstraction_engine: AbstractionEngine,
                 world_model: 'PredictiveWorldModel'):
        self.genome = genome
        self.memory = memory
        self.self_model = self_model
        self.skill_library = skill_library
        self.abstraction_engine = abstraction_engine
        self.world_model = world_model

        self.reasoner = ReasoningEngine(memory, abstraction_engine)

    def set_executor(self, executor: 'ActionExecutor'):
        """
        Dependency Injection: Allows the planner to access the executor 
        for running simulations without causing a circular import error.
        """
        self.executor = executor

    async def generate_plan_tree(self, goal: Goal, depth: int = 3, breadth: int = 3) -> List[Action]:
        """
        Tree of Thoughts (ToT) Search.
        Explores future states to find the optimal path.
        """
        log("PLAN", f"🌳 Growing thought tree for: {goal.action.action_type.name}", Term.CYAN)
        
        # State = (Accumulated Actions, Current Context Score)
        # Queue = List of paths
        queue = [([], 0.0)] 
        
        best_path = []
        best_score = -float('inf')

        current_context_vec = self.executor.executive._get_state_vector() # Access executive state

        for d in range(depth):
            candidates = []
            
            # 1. Expand (Breadth)
            while queue:
                path, score = queue.pop(0)
                
                # Generate 'breadth' number of next logical steps
                possible_next_actions = self._propose_next_actions(path, breadth)
                
                for action in possible_next_actions:
                    # 2. Evaluate (Simulate)
                    # Use World Model to imagine the result of this specific action sequence
                    predicted_state_vec = self.executor.executive.world_model.imagine_sequence(
                        current_context_vec, 
                        path + [action]
                    )
                    
                    # Score the state against the Goal (Similarity check)
                    # (Assuming goal has a target vector representation)
                    state_val = self._evaluate_state(predicted_state_vec, goal)
                    
                    candidates.append((path + [action], state_val))

            # 3. Prune
            # Keep only the top 'breadth' paths for the next layer
            candidates.sort(key=lambda x: x[1], reverse=True)
            queue = candidates[:breadth]
            
            if queue and queue[0][1] > best_score:
                best_score = queue[0][1]
                best_path = queue[0][0]

        log("PLAN", f"✅ ToT Complete. Best path length: {len(best_path)} Score: {best_score:.2f}", Term.GREEN)
        return best_path

    def _propose_next_actions(self, current_path: List[Action], n: int) -> List[Action]:
        """Uses Skill Library + Randomness to propose valid next moves."""
        # Simple heuristic: What skills fit the current context?
        valid_skills = self.skill_library.find_applicable_skills(self.memory.get_context()) # Simplified
        
        actions = []
        for _ in range(n):
            if valid_skills:
                skill = random.choice(valid_skills)
                actions.append(Action(ActionType.EXECUTE_SKILL, skill.name))
            else:
                # Fallback to generic actions
                actions.append(Action(ActionType.RESEARCH, "fallback_strategy"))
        return actions

    def predict_success(self, goal: 'Goal') -> float:
        """Predict the likelihood of a goal succeeding based on history"""
        # Base prediction from causal model
        causal_pred = self.memory.causal.predict_success(
            goal.action.action_type, 
            goal.context
        )
        
        # Adjust based on self-confidence in that capability
        self_conf = self.self_model.get_confidence(goal.action.action_type)
        
        # Weighted average
        return (causal_pred * 0.4) + (self_conf * 0.6)

    # [Add to PlanningAgent in agents.py]
    def recall_solution(self, goal_action_type: ActionType, context_keys: List[str]) -> Optional[str]:
        """
        Find a past episode with the same action and similar context that succeeded.
        Returns the specific payload used previously.
        """
        # Retrieve successful episodes of this type
        successes = self.memory.episodic.get_by_action(goal_action_type)
        successes = [e for e in successes if e.outcome]
        
        best_match = None
        best_score = 0.0
        
        for ep in successes:
            # Simple context overlap score
            match_count = sum(1 for k in context_keys if k in ep.context)
            if match_count > best_score:
                best_score = match_count
                best_match = ep
        
        if best_match and best_score > 0:
            log("PLAN", f"💡 Recalled solution from {best_match.age():.1f}s ago", Term.CYAN)
            return best_match.action.payload
            
        return None

    def should_decompose(self, goal: 'Goal') -> bool:
        """Decide if a goal is too complex and needs breakdown"""
        if goal.action.action_type in [ActionType.PROCESS_INPUT, ActionType.RESEARCH]:
            # Complex tasks often need decomposition
            return self.predict_success(goal) < self.genome.traits["confidence_threshold"]
        return False

    def decompose_goal(self, goal: 'Goal') -> List['Goal']:
        """Break a goal into subgoals"""
        subgoals = []
        parent_prio = goal.priority
        
        if goal.action.action_type == ActionType.PROCESS_INPUT:
            # Standard problem solving strategy
            # 1. Research/Understand
            # 2. Hypothesize/Plan
            # 3. Execute/Respond
            
            subgoals.append(Goal(
                priority=parent_prio + 0.1, 
                action=Action(ActionType.RESEARCH, goal.action.payload)
            ))
            
            # If we have relevant skills, use them
            best_skill = self.skill_library.get_best_skill(goal.context)
            if best_skill:
                # We interpret "using a skill" as executing its code/steps
                # For now, we wrap it in a generic execution action
                subgoals.append(Goal(
                    priority=parent_prio,
                    action=Action(ActionType.EXECUTE_SKILL, best_skill.name) # Placeholder logic
                ))
                pass

            if not subgoals and not best_skill:
                log("PLAN", "🤔 No standard skill found. Attempting reasoning...", Term.PURPLE)
            
            # A. Try Analogy (Solve based on similar past experience)
            analogous_action = self.reasoner.solve_by_analogy(goal.context)
            if analogous_action:
                log("PLAN", f"💡 Found analogy! Adapting strategy.", Term.CYAN)
                subgoals.append(Goal(
                    priority=parent_prio,
                    action=analogous_action,
                    context=goal.context
                ))
            
            # B. Try Deduction (If the goal is to find a relation)
            elif "relation" in goal.context:
                target = goal.context.get("target")
                source = goal.context.get("source")
                if target and source:
                    score, path = self.reasoner.deduce_relation(source, target)
                    if score > 0.1:
                        log("PLAN", f"🔗 Deduced path: {' -> '.join(path)}", Term.CYAN)
                        # Create a plan to verify this deduction
                        subgoals.append(Goal(
                            priority=parent_prio,
                            action=Action(ActionType.RESEARCH, f"Verify connection: {path}"),
                            context=goal.context
                        ))
            
        return subgoals

    def generate_autonomous_goals(self) -> List['Goal']:
        """
        Generate goals based on intrinsic drives.
        Now includes MOTOR BABBLING: Randomly trying tools to see what they do.
        """
        goals = []
        
        # 1. Motor Babbling (The Infant Phase)
        # If curiosity is high, try a random action with a random concept
        if self.genome.values["curiosity"] > 0.3:
            # Pick a random available action type
            # We filter out internal-only actions to focus on external effects
            options = [
                ActionType.SPEAK, 
                ActionType.RESEARCH, 
                ActionType.READ_FILE, 
                ActionType.EXPLORE
            ]
            action_type = random.choice(options)
            
            # Pick a random concept to use as the "Payload"
            payload = "init"
            if self.memory.semantic.idx_to_concept:
                idx = random.choice(list(self.memory.semantic.idx_to_concept.keys()))
                payload = self.memory.semantic.idx_to_concept[idx]
            
            # Create a "Just try it" goal
            # We give it slightly lower priority than survival, but high enough to happen often
            log("PLAN", f"👶 Motor Babbling: Trying {action_type.name} with '{payload}'", Term.DIM)
            goals.append(Goal(
                priority=2.0, 
                action=Action(action_type, payload, emergent=True)
            ))

        # 2. Memory Consolidation (Maintenance)
        if self.memory.semantic.next_idx > 100 and random.random() < 0.1:
             goals.append(Goal(priority=1.5, action=Action(ActionType.CONSOLIDATE, None, emergent=True)))

        # 3. Reflection (Self-Analysis)
        if random.random() < 0.2:
            goals.append(Goal(priority=1.0, action=Action(ActionType.REFLECT, "idle_reflection", emergent=True)))

        return goals

    def simulate(self, action: Action) -> Tuple[bool, str]:
        """
        Run a mental simulation of an action to check for safety/outcome.
        Returns (success, message).
        """
        # Safety check for self-modification
        if action.action_type == ActionType.SELF_MODIFY:
            # Check if modification is within bounds (logic mirrored from Genome)
            if "=" in str(action.payload):
                try:
                    trait, val = str(action.payload).split("=")
                    trait = trait.strip()
                    val = float(val)
                    bounds = self.genome._get_safety_bounds(trait)
                    if not (bounds[0] <= val <= bounds[1]):
                        return False, f"Simulation: Value {val} for {trait} out of bounds {bounds}"
                except:
                    pass
        
        return True, "Simulation safe"

    def recall_solution(self, goal_action_type: ActionType, context_keys: List[str]) -> Optional[str]:
        """
        Find a past episode with the same action and similar context that succeeded.
        Returns the specific payload used previously.
        """
        # Retrieve successful episodes of this type
        successes = self.memory.episodic.get_by_action(goal_action_type)
        successes = [e for e in successes if e.outcome]
        
        best_match = None
        best_score = 0.0
        
        for ep in successes:
            # simple context overlap score
            match_count = sum(1 for k in context_keys if k in ep.context)
            if match_count > best_score:
                best_score = match_count
                best_match = ep
        
        if best_match and best_score > 0:
            log("PLAN", f"💡 Recalled solution from {best_match.age():.1f}s ago", Term.CYAN)
            return best_match.action.payload
            
        return None

    def find_transferable_skill(self, context: Dict[str, Any]) -> Optional[str]:
        """
        Generalization: If no exact skill exists, find a skill 
        used in a semantically similar context.
        """
        # Get vector for current context keys
        current_concepts = list(context.keys())
        
        best_skill = None
        best_similarity = 0.0
        
        # Iterate through all known skills
        for name, skill in self.skill_library.skills.items():
            # Check skill's 'preconditions' (concepts it usually works with)
            for pre in skill.preconditions:
                # Check similarity against current context
                for curr in current_concepts:
                    sim = self.memory.semantic.get_similarity(pre, curr)
                    
                    if sim > 0.7 and sim > best_similarity:
                        best_similarity = sim
                        best_skill = name
                        
        if best_skill and best_similarity > 0.7:
            log("PLAN", f"💡 Generalization: Transferring '{best_skill}' to new domain (Sim: {best_similarity:.2f})", Term.CYAN)
            return best_skill
            
        return None

    def select_best_action(self, goal: 'Goal', candidates: List[Action]) -> Action:
        """
        Model-Based RL (lightweight).
        NOTE: Removed dependency on obsolete self.executor.executive.
        Uses world_model directly with simple vectorization.
        """
        # NOTE: If world_model isn't available for some reason, fail safe.
        if not hasattr(self, "world_model") or self.world_model is None:
            return candidates[0]

        # NOTE: Minimal state vector (placeholder): we use a zero state for now.
        # You can later swap this with a real state encoder from WorkingMemory/Semantic activation.
        current_state = xp.zeros((256,), dtype=xp.float32)

        best_action = None
        best_score = -float('inf')

        for action in candidates:
            # NOTE: Minimal action vector: hash action type into first few dims
            action_vec = xp.zeros((64,), dtype=xp.float32)
            try:
                idx = int(action.action_type.value) % 64
                action_vec[idx] = 1.0
            except Exception:
                action_vec[0] = 1.0

            trajectory = self.world_model.imagine(current_state, action_vec, steps=3)

            predicted_value = 0.0
            cumulative_uncertainty = 0.0
            curiosity_drive = self.genome.values["curiosity"]

            for step in trajectory:
                u = float(step.get("uncertainty", 0.0))
                cumulative_uncertainty += u

                # NOTE: Curiosity flips the sign (explore vs safety)
                if curiosity_drive > 0.7:
                    predicted_value += u * 2.0
                else:
                    predicted_value -= u

            if cumulative_uncertainty > 2.0 and curiosity_drive < 0.5:
                continue

            if predicted_value > best_score:
                best_score = predicted_value
                best_action = action

        return best_action or candidates[0]

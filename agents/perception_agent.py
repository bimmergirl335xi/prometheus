class PerceptionAgent:
    """
    Processes raw input from the Workspace and converts it into actionable signals.
    """
    def __init__(self, 
                 workspace: ActiveGlobalWorkspace, 
                 memory: LongTermMemory,
                 abstraction_engine: AbstractionEngine):
        self.workspace = workspace
        self.memory = memory
        self.abstraction_engine = abstraction_engine
        self.language_engine = LanguageEngine(memory)
        self.universal_grammar = UniversalGrammar(memory)

    def perceive(self) -> Optional[Tuple[float, str, Any]]:
        """Check workspace for new salient information (NON-CONSUMING)"""

        # NOTE: Stream-of-consciousness should NEVER steal from the Mesh/Executor queue.
        # It should only enrich Working Memory / Semantic activation.
        self.associate_stream()

        # NOTE: The Mesh consumes via workspace.get_most_salient().
        # The Executor consumes via a dedicated "pop_next_executable()" (added below).
        return None

    def associate_stream(self):
        """
        The 'Stream of Consciousness'. 
        Uses items in Working Memory to pull related LTM concepts back into WM.
        """
        # Get current focus items
        active_items = self.workspace.working_memory.retrieve()
        
        for item in active_items:
            if isinstance(item, str):
                # 1. Check direct associations
                related = self.memory.semantic.get_associated(item, top_k=2)
                
                # 2. Check semantic similarity (The vector improvement)
                # We interpret similar vectors as "reminders"
                
                for concept, weight in related:
                    # Chance to bubble up into working memory
                    if weight > 0.5 and random.random() < 0.3:
                        # "Reminded" of something
                        log("MEM", f"💭 Association: {item} -> {concept}", Term.DIM)
                        self.workspace.working_memory.hold(concept, "associative")

    def interpret(self, signal: Tuple[float, str, Any]) -> Optional[Action]:
        """Convert a raw signal into a cognitive action"""
        priority, source, content = signal
        
        if source == "USER":
            text = str(content)
            
            # LEARN GRAMMAR: Every user input is a lesson in English
            self.language_engine.learn_grammar(text)

        # Stimulate memory with keywords from content
        if isinstance(content, str):
            words = content.split()
            for word in words:
                if len(word) > 3:
                    self.memory.semantic.stimulate(word, strength=0.2)

        # 1. User Commands
        if source == "USER":
            text = str(content).strip()

            if isinstance(content, dict) and content.get("type") == "file":
                return Action(ActionType.READ_FILE, content["path"], metadata={"source": "user"})
            
            text = str(content)

        if text.lower().startswith(("read ", "open ", "load ")):
                try:
                    parts = text.split(" ", 1)
                    if len(parts) > 1:
                        filepath = parts[1].strip()
                        # Cleanup quotes
                        filepath = filepath.replace('"', '').replace("'", "")
                        return Action(ActionType.READ_FILE, filepath, metadata={"source": "user"})
                except Exception:
                    pass

        if source == "USER":
            text = str(content)
            
            # Instead of just splitting strings, we understand the EVENT.
            structural_parse = self.universal_grammar.parse_universal(text)
            
            if structural_parse.get("action"):
                log("LANG", f"Parsed Event: {structural_parse}", Term.GREEN)

            # System Modification
            if text.startswith("SYS_MOD"):
                try:
                    # Parse "SYS_MOD: trait = value"
                    _, assignment = text.split(":", 1)
                    trait, value = assignment.split("=")
                    trait = trait.strip()
                    value = float(value.strip())
                    return Action(ActionType.SELF_MODIFY, f"{trait}={value}", metadata={"source": "user"})
                except ValueError:
                    log("PERCEPT", "Invalid syntax for SYS_MOD", Term.RED)
                    return None
                pass

            # Checkpoint
            elif text.startswith("CHECKPOINT"):
                name = text.split(" ", 1)[1] if " " in text else None
                return Action(ActionType.CHECKPOINT, name)

            # Build Skill
            elif text.startswith("BUILD_SKILL"):
                skill_name = text.split(" ", 1)[1] if " " in text else "new_skill"
                return Action(ActionType.BUILD_SKILL, skill_name)

            # General Inquiry / Task
            else:
                return Action(ActionType.PROCESS_INPUT, text, metadata={"source": "user"})

        # 2. Network Results
        elif source == "NETWORK":
            return Action(ActionType.LEARN, content, metadata={"source": "network"})

        return None
    
    def extract_skill_from_text(self, text: str) -> Optional[Action]:
        """
        Social Learning: Scans text for instructional patterns 
        (e.g., "To do X: Step 1, Step 2...") and compiles a Skill.
        """
        # Regex to find "To [SkillName]: [Steps]"
        pattern = r"To\s+(\w+):(.+)"
        match = re.search(pattern, text, re.IGNORECASE)
        
        if match:
            skill_name = match.group(1).strip()
            steps_text = match.group(2).strip()
            
            # Simple parser: split by commas or numbers
            steps = [s.strip() for s in re.split(r'[,;]|\d+\.', steps_text) if len(s) > 3]
            
            if len(steps) > 1:
                # We construct a new skill programmatically
                code_block = "def execute(context):\n"
                for step in steps:
                    code_block += f"    print('Executing step: {step}')\n"
                    # In a real scenario, we would map 'step' words to internal functions
                
                log("SOCIAL", f"🧠 Learned skill '{skill_name}' from text!", Term.GREEN)
                
                # Return an action to formally build this skill
                return Action(
                    ActionType.BUILD_SKILL, 
                    payload=skill_name
                )
        return None

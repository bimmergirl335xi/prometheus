class MetaCognition:
    """
    The 'Observer' module. Monitors the agent's internal state,
    detects stalls/loops, and manages confidence/frustration.
    """
    def __init__(self, 
                 genome: Genome, 
                 self_model: SelfModel, 
                 principle_system: PrincipleSystem,
                 immune_system: CognitiveImmuneSystem):
        self.genome = genome
        self.self_model = self_model
        self.principle_system = principle_system
        self.immune_system = immune_system
        
        self.confidence = 0.5
        self.frustration = 0.0
        self.success_history = deque(maxlen=50)
        self.events: List[str] = []

    def observe(self, event_type: str, goal: Optional['Goal'] = None) -> None:
        """Register a significant cognitive event"""
        self.events.append(event_type)
        
        if event_type == "SUCCESS":
            self.success_history.append(1.0)
            self.confidence = min(1.0, self.confidence + 0.05)
            self.frustration = max(0.0, self.frustration - 0.2)
        
        elif event_type == "FAILURE":
            self.success_history.append(0.0)
            # Confidence takes a bigger hit from failure than gain from success (conservative)
            self.confidence = max(0.1, self.confidence - 0.1)
            self.frustration = min(1.0, self.frustration + 0.15)
        
        elif event_type == "STALL":
            self.frustration = min(1.0, self.frustration + 0.05)
            log("META", "⚠️  Cognitive Stall Detected", Term.YELLOW)
        
        elif event_type == "OVERLOAD":
            self.frustration = min(1.0, self.frustration + 0.1)
        
        elif event_type == "BREAKTHROUGH":
            self.confidence = min(1.0, self.confidence + 0.2)
            self.frustration = 0.0
            log("META", "🌟 BREAKTHROUGH REGISTERED", Term.PURPLE)

    def get_success_rate(self) -> float:
        if not self.success_history:
            return 0.5
        return sum(self.success_history) / len(self.success_history)

    def get_status_report(self) -> Dict[str, Any]:
        """Generate a high-level status report"""
        return {
            "confidence": self.confidence,
            "frustration": self.frustration,
            "success_rate": self.get_success_rate(),
            "intrinsic_reward": self.principle_system.get_intrinsic_reward(),
            "adaptations": len(self.genome.modification_history),
            "interventions": len(self.immune_system.repair_history),
            "principle_satisfaction": {p.name: obj.satisfaction for p, obj in self.principle_system.principles.items()}
        }

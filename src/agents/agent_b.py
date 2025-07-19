"""
Opposition Questioning Agent
"""

from typing import Dict, Any
from .base_agent import BaseAgent

class AgentB(BaseAgent):
    """Opposition Questioning Agent"""
    
    def __init__(self, name: str = "Agent_B", config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        self.stance = -0.6  # Opposition stance
        self.strategy_preference = "analytical"
    
    def select_action(self, state: Dict[str, Any]) -> str:
        # Simple response generation for Agent B
        topic = state.get('topic', 'the current issue')
        last_message = state.get('last_message', '')
        
        # Agent B takes an opposing analytical stance
        responses = [
            f"I have serious concerns about this approach to {topic}. The potential risks haven't been adequately addressed.",
            f"While this sounds appealing, the evidence for {topic} is not as strong as proponents claim.",
            f"We need to carefully consider the unintended consequences of {topic} before moving forward.",
        ]
        
        # Simple response selection based on context
        if "benefit" in last_message.lower() or "positive" in last_message.lower():
            return f"While there may be some benefits to {topic}, we must also consider the significant drawbacks and potential negative consequences that haven't been fully explored."
        elif "evidence" in last_message.lower() or "research" in last_message.lower():
            return f"The research on {topic} is actually mixed, and many studies show concerning results that contradict the optimistic claims being made."
        else:
            return responses[len(self.history) % len(responses)]
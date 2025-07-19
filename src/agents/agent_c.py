"""
Neutral Analytical Agent
"""

from typing import Dict, Any
from .base_agent import BaseAgent

class AgentC(BaseAgent):
    """Neutral Analytical Agent"""
    
    def __init__(self, name: str = "Agent_C", config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        self.stance = 0.0  # Neutral stance
        self.strategy_preference = "empathetic"
    
    def select_action(self, state: Dict[str, Any]) -> str:
        # Simple response generation for Agent C
        topic = state.get('topic', 'the current issue')
        last_message = state.get('last_message', '')
        
        # Agent C takes a neutral, empathetic stance
        responses = [
            f"I can see valid points on both sides of {topic}. Perhaps we should consider a balanced approach.",
            f"Both perspectives on {topic} have merit. Let's explore how we might find common ground.",
            f"This is a complex issue regarding {topic}. I think we need to understand all stakeholders' concerns.",
        ]
        
        # Simple response selection based on context
        if "concern" in last_message.lower() or "risk" in last_message.lower():
            return f"I understand the concerns about {topic}. These are valid worries that deserve careful consideration alongside the potential benefits."
        elif "support" in last_message.lower() or "benefit" in last_message.lower():
            return f"The benefits of {topic} are certainly worth considering. At the same time, we should also acknowledge the legitimate concerns that have been raised."
        else:
            return responses[len(self.history) % len(responses)]
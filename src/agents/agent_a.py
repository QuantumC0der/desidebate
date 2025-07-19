"""
Proactive Support Agent
"""

from typing import Dict, Any
from .base_agent import BaseAgent

class AgentA(BaseAgent):
    """Proactive Support Agent"""
    
    def __init__(self, name: str = "Agent_A", config: Dict[str, Any] = None):
        super().__init__(name, config or {})
        self.stance = 0.8  # Strong support stance
        self.strategy_preference = "aggressive"
    
    def select_action(self, state: Dict[str, Any]) -> str:
        # Simple response generation for Agent A
        topic = state.get('topic', 'the current issue')
        last_message = state.get('last_message', '')
        
        # Agent A takes a strong supportive stance
        responses = [
            f"I strongly support this position on {topic}. The evidence clearly shows the benefits outweigh any concerns.",
            f"This is absolutely the right approach for {topic}. We need to move forward decisively.",
            f"The data supports this stance on {topic}. We cannot ignore the positive impact this will have.",
        ]
        
        # Simple response selection based on context
        if "concern" in last_message.lower() or "problem" in last_message.lower():
            return f"While I understand the concerns raised, I believe the benefits of {topic} far outweigh the risks. We have evidence showing positive outcomes."
        elif "data" in last_message.lower() or "research" in last_message.lower():
            return f"The research strongly supports my position on {topic}. Multiple studies demonstrate the effectiveness of this approach."
        else:
            return responses[len(self.history) % len(responses)]
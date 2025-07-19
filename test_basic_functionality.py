#!/usr/bin/env python3
"""
Basic functionality test for Desi Debate
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_imports():
    """Test if all core modules can be imported"""
    print("Testing imports...")
    
    try:
        from src.utils.config_loader import ConfigLoader
        print("✓ ConfigLoader")
    except ImportError as e:
        print(f"❌ ConfigLoader: {e}")
        return False
    
    try:
        from src.gpt_interface.gpt_client import chat
        print("✓ GPT Client")
    except ImportError as e:
        print(f"❌ GPT Client: {e}")
        return False
    
    try:
        from src.agents.agent_a import AgentA
        from src.agents.agent_b import AgentB
        from src.agents.agent_c import AgentC
        print("✓ Agents")
    except ImportError as e:
        print(f"❌ Agents: {e}")
        return False
    
    try:
        from src.rag.simple_retriever import SimpleRetriever
        print("✓ RAG Retriever")
    except ImportError as e:
        print(f"❌ RAG Retriever: {e}")
        return False
    
    try:
        from src.orchestrator.parallel_orchestrator import ParallelOrchestrator
        print("✓ Orchestrator")
    except ImportError as e:
        print(f"❌ Orchestrator: {e}")
        return False
    
    return True

def test_config_loading():
    """Test configuration loading"""
    print("\nTesting configuration...")
    
    try:
        from src.utils.config_loader import ConfigLoader
        config = ConfigLoader.load('debate')
        print(f"✓ Loaded debate config: {len(config)} sections")
        return True
    except Exception as e:
        print(f"❌ Config loading failed: {e}")
        return False

def test_agents():
    """Test agent initialization"""
    print("\nTesting agents...")
    
    try:
        from src.agents.agent_a import AgentA
        from src.agents.agent_b import AgentB
        from src.agents.agent_c import AgentC
        
        agent_a = AgentA()
        agent_b = AgentB()
        agent_c = AgentC()
        
        print(f"✓ Agent A: {agent_a.name}")
        print(f"✓ Agent B: {agent_b.name}")
        print(f"✓ Agent C: {agent_c.name}")
        
        # Test simple response
        state = {
            'topic': 'artificial intelligence regulation',
            'last_message': 'This is a test message'
        }
        
        response_a = agent_a.select_action(state)
        print(f"✓ Agent A response: {response_a[:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Agent test failed: {e}")
        return False

def test_retriever():
    """Test RAG retriever"""
    print("\nTesting retriever...")
    
    try:
        from src.rag.simple_retriever import SimpleRetriever
        retriever = SimpleRetriever()
        
        results = retriever.retrieve("artificial intelligence", top_k=2)
        print(f"✓ Retrieved {len(results)} results")
        
        if results:
            print(f"✓ Sample result: {results[0].text[:50]}...")
        
        return True
    except Exception as e:
        print(f"❌ Retriever test failed: {e}")
        return False

def test_orchestrator():
    """Test orchestrator initialization"""
    print("\nTesting orchestrator...")
    
    try:
        from src.orchestrator.parallel_orchestrator import ParallelOrchestrator
        orchestrator = ParallelOrchestrator()
        
        # Test agent initialization
        agent_configs = [
            {'id': 'Agent_A', 'initial_stance': 0.8, 'initial_conviction': 0.7},
            {'id': 'Agent_B', 'initial_stance': -0.6, 'initial_conviction': 0.6},
            {'id': 'Agent_C', 'initial_stance': 0.0, 'initial_conviction': 0.5}
        ]
        
        agents = orchestrator.initialize_agents(agent_configs)
        print(f"✓ Initialized {len(agents)} agents")
        
        return True
    except Exception as e:
        print(f"❌ Orchestrator test failed: {e}")
        return False

def main():
    print("Desi Debate - Basic Functionality Test")
    print("=" * 40)
    
    tests = [
        test_imports,
        test_config_loading,
        test_agents,
        test_retriever,
        test_orchestrator
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print(f"\nTest Results: {passed}/{total} passed")
    
    if passed == total:
        print("✓ All tests passed! System is ready.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
🎭 Interactive Demo Helper for Desi Debate
Provides guided demo scenarios for hackathon judges and evaluators
"""

import json
import time
import webbrowser
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional

class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

class InteractiveDemo:
    """Interactive demo guide for hackathon presentations"""
    
    def __init__(self):
        self.scenarios = self._load_scenarios()
        self.current_scenario = None
        self.demo_start_time = None
        
    def _load_scenarios(self) -> Dict:
        """Load demo scenarios from JSON file"""
        try:
            with open('demo_scenarios.json', 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"{Colors.RED}❌ demo_scenarios.json not found{Colors.END}")
            return {"hackathon_demo_scenarios": {"scenarios": []}}
        except json.JSONDecodeError as e:
            print(f"{Colors.RED}❌ Error parsing demo_scenarios.json: {e}{Colors.END}")
            return {"hackathon_demo_scenarios": {"scenarios": []}}
    
    def print_header(self):
        """Print attractive demo header"""
        print(f"\n{Colors.PURPLE}{'='*70}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}🎭 DESI DEBATE - INTERACTIVE DEMO GUIDE 🏆{Colors.END}")
        print(f"{Colors.PURPLE}{'='*70}{Colors.END}")
        print(f"{Colors.WHITE}AI-Powered Multi-Agent Debate System{Colors.END}")
        print(f"{Colors.YELLOW}Guided Demo for Hackathon Judges{Colors.END}")
        print(f"{Colors.PURPLE}{'='*70}{Colors.END}\n")
    
    def show_main_menu(self):
        """Display main demo menu"""
        print(f"{Colors.BLUE}📋 Demo Options:{Colors.END}")
        print(f"  {Colors.WHITE}1.{Colors.END} Quick 8-minute demo sequence")
        print(f"  {Colors.WHITE}2.{Colors.END} Individual scenario selection")
        print(f"  {Colors.WHITE}3.{Colors.END} System status check")
        print(f"  {Colors.WHITE}4.{Colors.END} Launch web interface")
        print(f"  {Colors.WHITE}5.{Colors.END} View technical details")
        print(f"  {Colors.WHITE}6.{Colors.END} Exit")
        print()
    
    def run_quick_demo(self):
        """Run the optimized 8-minute demo sequence"""
        print(f"{Colors.BOLD}{Colors.GREEN}🚀 Starting Quick Demo Sequence (8 minutes){Colors.END}")
        print(f"{Colors.PURPLE}{'='*50}{Colors.END}")
        
        sequence = self.scenarios.get("hackathon_demo_scenarios", {}).get("quick_demo_sequence", {})
        if not sequence:
            print(f"{Colors.RED}❌ Quick demo sequence not found{Colors.END}")
            return
        
        self.demo_start_time = time.time()
        
        for step_info in sequence.get("sequence", []):
            step_num = step_info["step"]
            duration = step_info["duration"]
            action = step_info["action"]
            talking_points = step_info["talking_points"]
            
            print(f"\n{Colors.BLUE}Step {step_num}: {action}{Colors.END}")
            print(f"{Colors.YELLOW}Duration: {duration}{Colors.END}")
            print(f"{Colors.WHITE}Talking Points:{Colors.END}")
            for point in talking_points:
                print(f"  • {point}")
            
            input(f"\n{Colors.GREEN}Press Enter when ready to continue...{Colors.END}")
        
        elapsed = time.time() - self.demo_start_time
        print(f"\n{Colors.GREEN}✅ Demo completed in {elapsed/60:.1f} minutes{Colors.END}")
    
    def show_scenarios(self):
        """Display available demo scenarios"""
        scenarios = self.scenarios.get("hackathon_demo_scenarios", {}).get("scenarios", [])
        
        if not scenarios:
            print(f"{Colors.RED}❌ No scenarios available{Colors.END}")
            return
        
        print(f"{Colors.BLUE}📋 Available Demo Scenarios:{Colors.END}\n")
        
        for i, scenario in enumerate(scenarios, 1):
            print(f"{Colors.WHITE}{i}. {scenario['title']}{Colors.END}")
            print(f"   Topic: {scenario['topic']}")
            print(f"   Duration: {scenario['duration_estimate']}")
            print(f"   Features: {', '.join(scenario['key_features'][:2])}...")
            print()
        
        try:
            choice = input(f"{Colors.YELLOW}Select scenario (1-{len(scenarios)}) or 'b' for back: {Colors.END}")
            if choice.lower() == 'b':
                return
            
            scenario_idx = int(choice) - 1
            if 0 <= scenario_idx < len(scenarios):
                self.run_scenario(scenarios[scenario_idx])
            else:
                print(f"{Colors.RED}❌ Invalid selection{Colors.END}")
        except ValueError:
            print(f"{Colors.RED}❌ Invalid input{Colors.END}")
    
    def run_scenario(self, scenario: Dict):
        """Run a specific demo scenario"""
        self.current_scenario = scenario
        
        print(f"\n{Colors.BOLD}{Colors.CYAN}🎯 {scenario['title']}{Colors.END}")
        print(f"{Colors.PURPLE}{'='*50}{Colors.END}")
        
        print(f"{Colors.WHITE}Topic:{Colors.END} {scenario['topic']}")
        print(f"{Colors.WHITE}Description:{Colors.END} {scenario['description']}")
        print(f"{Colors.WHITE}Estimated Duration:{Colors.END} {scenario['duration_estimate']}")
        print()
        
        print(f"{Colors.BLUE}🔍 Key Features to Highlight:{Colors.END}")
        for feature in scenario['key_features']:
            print(f"  • {feature}")
        print()
        
        print(f"{Colors.BLUE}💡 Judge Talking Points:{Colors.END}")
        for point in scenario['judge_talking_points']:
            print(f"  • {point}")
        print()
        
        print(f"{Colors.BLUE}⚙️ Setup Instructions:{Colors.END}")
        for instruction in scenario['setup_instructions']:
            print(f"  • {instruction}")
        print()
        
        print(f"{Colors.YELLOW}Expected Outcome:{Colors.END} {scenario['expected_outcome']}")
        print()
        
        choice = input(f"{Colors.GREEN}Ready to run this scenario? (y/n): {Colors.END}")
        if choice.lower() == 'y':
            self._execute_scenario(scenario)
    
    def _execute_scenario(self, scenario: Dict):
        """Execute the selected scenario"""
        print(f"\n{Colors.GREEN}🚀 Executing: {scenario['title']}{Colors.END}")
        
        # Check if web interface is running
        if not self._check_web_interface():
            print(f"{Colors.YELLOW}⚠️  Web interface not detected. Starting...{Colors.END}")
            self._start_web_interface()
        
        # Open browser to the demo
        print(f"{Colors.BLUE}🌐 Opening web interface...{Colors.END}")
        webbrowser.open('http://localhost:5000')
        
        # Provide step-by-step guidance
        print(f"\n{Colors.BOLD}📋 Demo Steps:{Colors.END}")
        print(f"1. {Colors.WHITE}Set topic:{Colors.END} {scenario['topic']}")
        print(f"2. {Colors.WHITE}Click 'Start Debate'{Colors.END}")
        print(f"3. {Colors.WHITE}Watch for:{Colors.END} {', '.join(scenario['key_features'][:2])}")
        print(f"4. {Colors.WHITE}Point out:{Colors.END} {scenario['judge_talking_points'][0]}")
        
        input(f"\n{Colors.GREEN}Press Enter when demo is complete...{Colors.END}")
        
        # Show results summary
        print(f"\n{Colors.GREEN}✅ Scenario completed!{Colors.END}")
        print(f"{Colors.BLUE}Next steps:{Colors.END}")
        print(f"  • Export the debate results")
        print(f"  • Show performance metrics")
        print(f"  • Discuss technical architecture")
    
    def check_system_status(self):
        """Check system status and readiness"""
        print(f"{Colors.BLUE}🔍 System Status Check{Colors.END}")
        print(f"{Colors.PURPLE}{'='*30}{Colors.END}")
        
        checks = [
            ("Python Environment", self._check_python),
            ("Dependencies", self._check_dependencies),
            ("Configuration", self._check_configuration),
            ("Web Interface", self._check_web_interface),
            ("Demo Files", self._check_demo_files)
        ]
        
        passed = 0
        for check_name, check_func in checks:
            print(f"  {check_name}...", end=' ')
            try:
                if check_func():
                    print(f"{Colors.GREEN}✅{Colors.END}")
                    passed += 1
                else:
                    print(f"{Colors.RED}❌{Colors.END}")
            except Exception as e:
                print(f"{Colors.RED}❌ ({str(e)[:30]}...){Colors.END}")
        
        print(f"\n{Colors.WHITE}Status: {passed}/{len(checks)} checks passed{Colors.END}")
        
        if passed == len(checks):
            print(f"{Colors.GREEN}🎉 System ready for demonstration!{Colors.END}")
        else:
            print(f"{Colors.YELLOW}⚠️  Some issues detected. Run setup script:{Colors.END}")
            print(f"  python hackathon_setup.py")
    
    def _check_python(self) -> bool:
        """Check Python version"""
        return sys.version_info >= (3, 8)
    
    def _check_dependencies(self) -> bool:
        """Check if core dependencies are available"""
        try:
            import flask
            import yaml
            return True
        except ImportError:
            return False
    
    def _check_configuration(self) -> bool:
        """Check configuration files"""
        return Path('.env').exists() and Path('configs/debate.yaml').exists()
    
    def _check_web_interface(self) -> bool:
        """Check if web interface is running"""
        try:
            import requests
            response = requests.get('http://localhost:5000/api/health', timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _check_demo_files(self) -> bool:
        """Check if demo files exist"""
        required_files = ['demo_scenarios.json', 'DEMO.md', 'hackathon_setup.py']
        return all(Path(f).exists() for f in required_files)
    
    def _start_web_interface(self):
        """Start the web interface"""
        try:
            print(f"{Colors.BLUE}Starting Flask server...{Colors.END}")
            subprocess.Popen([sys.executable, 'run_flask.py'], 
                           stdout=subprocess.DEVNULL, 
                           stderr=subprocess.DEVNULL)
            time.sleep(3)  # Give server time to start
        except Exception as e:
            print(f"{Colors.RED}❌ Failed to start web interface: {e}{Colors.END}")
    
    def launch_web_interface(self):
        """Launch web interface in browser"""
        print(f"{Colors.BLUE}🌐 Launching Web Interface{Colors.END}")
        
        if not self._check_web_interface():
            print(f"{Colors.YELLOW}Starting web server...{Colors.END}")
            self._start_web_interface()
            time.sleep(2)
        
        webbrowser.open('http://localhost:5000')
        print(f"{Colors.GREEN}✅ Web interface opened at http://localhost:5000{Colors.END}")
        
        print(f"\n{Colors.BLUE}💡 Quick Demo Tips:{Colors.END}")
        print(f"  • Use pre-loaded topic or try: 'Should AI be regulated?'")
        print(f"  • Click 'Start Debate' to begin")
        print(f"  • Watch agent status panels for real-time updates")
        print(f"  • Export results when complete")
    
    def show_technical_details(self):
        """Show technical architecture details"""
        print(f"{Colors.BLUE}🔧 Technical Architecture Overview{Colors.END}")
        print(f"{Colors.PURPLE}{'='*40}{Colors.END}")
        
        print(f"{Colors.WHITE}Core Components:{Colors.END}")
        print(f"  • {Colors.CYAN}RAG System:{Colors.END} FAISS/Chroma vector databases")
        print(f"  • {Colors.CYAN}GNN Model:{Colors.END} GraphSAGE + GAT (67.77% accuracy)")
        print(f"  • {Colors.CYAN}RL Agent:{Colors.END} PPO with 4 strategy types")
        print(f"  • {Colors.CYAN}Orchestrator:{Colors.END} Parallel async processing")
        
        print(f"\n{Colors.WHITE}Performance Metrics:{Colors.END}")
        print(f"  • Response Time: <2 seconds")
        print(f"  • Persuasion Prediction: 67.77% accuracy")
        print(f"  • Strategy Classification: 64.47% accuracy")
        print(f"  • System Reliability: >99% uptime")
        
        print(f"\n{Colors.WHITE}Innovation Highlights:{Colors.END}")
        print(f"  • First RAG+GNN+RL integration for debates")
        print(f"  • Multi-task learning architecture")
        print(f"  • Graceful degradation without API keys")
        print(f"  • Real-time social prediction")
        
        print(f"\n{Colors.BLUE}📚 Documentation:{Colors.END}")
        print(f"  • README.md - Project overview")
        print(f"  • ARCHITECTURE.md - Technical deep dive")
        print(f"  • DEMO.md - Demo guide")
        print(f"  • demo_scenarios.json - Scenario configurations")
    
    def run(self):
        """Main demo interface loop"""
        self.print_header()
        
        while True:
            self.show_main_menu()
            
            try:
                choice = input(f"{Colors.YELLOW}Select option (1-6): {Colors.END}")
                
                if choice == '1':
                    self.run_quick_demo()
                elif choice == '2':
                    self.show_scenarios()
                elif choice == '3':
                    self.check_system_status()
                elif choice == '4':
                    self.launch_web_interface()
                elif choice == '5':
                    self.show_technical_details()
                elif choice == '6':
                    print(f"\n{Colors.GREEN}Thanks for using Desi Debate! 🎭✨{Colors.END}")
                    break
                else:
                    print(f"{Colors.RED}❌ Invalid option{Colors.END}")
                    
            except KeyboardInterrupt:
                print(f"\n\n{Colors.YELLOW}Demo interrupted. Goodbye! 👋{Colors.END}")
                break
            except Exception as e:
                print(f"{Colors.RED}❌ Error: {str(e)}{Colors.END}")
            
            print()  # Add spacing between menu cycles

def main():
    """Main entry point"""
    try:
        demo = InteractiveDemo()
        demo.run()
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Goodbye! 👋{Colors.END}")
    except Exception as e:
        print(f"{Colors.RED}❌ Unexpected error: {str(e)}{Colors.END}")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""
🏆 Hackathon One-Click Setup Script for Desi Debate
Comprehensive environment validation and automated setup for judges and evaluators
"""

import os
import sys
import subprocess
import time
import platform
from pathlib import Path
from typing import List, Tuple, Dict, Optional

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

class HackathonSetup:
    """Comprehensive setup and validation for hackathon demonstration"""
    
    def __init__(self):
        self.start_time = time.time()
        self.errors = []
        self.warnings = []
        self.system_info = self._get_system_info()
        
    def _get_system_info(self) -> Dict[str, str]:
        """Gather system information for diagnostics"""
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            'architecture': platform.machine(),
            'processor': platform.processor() or 'Unknown'
        }
    
    def print_header(self):
        """Print attractive header for hackathon demo"""
        print(f"\n{Colors.PURPLE}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}🎭 DESI DEBATE - HACKATHON SETUP 🏆{Colors.END}")
        print(f"{Colors.PURPLE}{'='*60}{Colors.END}")
        print(f"{Colors.WHITE}AI-Powered Multi-Agent Debate System{Colors.END}")
        print(f"{Colors.YELLOW}RAG + GNN + RL Integration{Colors.END}")
        print(f"{Colors.PURPLE}{'='*60}{Colors.END}\n")
        
        print(f"{Colors.BLUE}System Information:{Colors.END}")
        for key, value in self.system_info.items():
            print(f"  {key.replace('_', ' ').title()}: {Colors.WHITE}{value}{Colors.END}")
        print()
    
    def check_python_version(self) -> bool:
        """Validate Python version requirements"""
        print(f"{Colors.BLUE}🐍 Checking Python Version...{Colors.END}")
        
        required_version = (3, 8)
        current_version = sys.version_info[:2]
        
        if current_version >= required_version:
            print(f"  {Colors.GREEN}✓ Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} (Required: 3.8+){Colors.END}")
            return True
        else:
            error_msg = f"Python {required_version[0]}.{required_version[1]}+ required, found {current_version[0]}.{current_version[1]}"
            print(f"  {Colors.RED}❌ {error_msg}{Colors.END}")
            self.errors.append(error_msg)
            return False
    
    def check_directory_structure(self) -> bool:
        """Validate and create necessary directories"""
        print(f"{Colors.BLUE}📁 Setting up Directory Structure...{Colors.END}")
        
        required_dirs = [
            "data/models",
            "data/processed", 
            "data/raw",
            "data/chroma",
            "logs",
            "cache",
            "src/rag/data/rag"
        ]
        
        success = True
        for directory in required_dirs:
            try:
                Path(directory).mkdir(parents=True, exist_ok=True)
                print(f"  {Colors.GREEN}✓ {directory}{Colors.END}")
            except Exception as e:
                print(f"  {Colors.RED}❌ {directory}: {str(e)}{Colors.END}")
                self.errors.append(f"Failed to create directory {directory}: {str(e)}")
                success = False
        
        return success
    
    def install_dependencies(self) -> Tuple[bool, int, int]:
        """Install Python dependencies with progress tracking"""
        print(f"{Colors.BLUE}📦 Installing Dependencies...{Colors.END}")
        
        # Core dependencies (must work)
        core_packages = [
            "torch>=2.0.0",
            "numpy>=1.24.0",
            "scipy>=1.10.0",
            "PyYAML>=6.0",
            "python-dotenv>=1.0.0",
            "flask>=3.0.0",
            "flask-cors>=4.0.0",
            "openai>=1.0.0",
            "tqdm>=4.65.0",
            "tiktoken>=0.5.0",
            "networkx>=3.0",
            "aiohttp>=3.8.0"
        ]
        
        # Optional packages (enhance functionality)
        optional_packages = [
            "torch-geometric>=2.3.0",
            "scikit-learn>=1.3.0", 
            "matplotlib>=3.7.0",
            "transformers>=4.30.0",
            "langchain>=0.1.0",
            "langchain-community>=0.0.20",
            "langchain-openai>=0.0.5",
            "chromadb>=0.4.0",
            "faiss-cpu>=1.7.0",
            "seaborn>=0.12.0"
        ]
        
        def install_package_list(packages: List[str], package_type: str) -> int:
            """Install a list of packages and return success count"""
            print(f"  {Colors.YELLOW}Installing {package_type} packages...{Colors.END}")
            success_count = 0
            
            for i, package in enumerate(packages, 1):
                package_name = package.split('>=')[0].split('==')[0]
                print(f"    [{i}/{len(packages)}] {package_name}...", end=' ')
                
                try:
                    result = subprocess.run(
                        [sys.executable, "-m", "pip", "install", package, "--quiet"],
                        capture_output=True,
                        text=True,
                        timeout=120  # 2 minute timeout per package
                    )
                    
                    if result.returncode == 0:
                        print(f"{Colors.GREEN}✓{Colors.END}")
                        success_count += 1
                    else:
                        print(f"{Colors.RED}❌{Colors.END}")
                        if package_type == "core":
                            self.errors.append(f"Failed to install core package: {package_name}")
                        else:
                            self.warnings.append(f"Failed to install optional package: {package_name}")
                            
                except subprocess.TimeoutExpired:
                    print(f"{Colors.RED}❌ (timeout){Colors.END}")
                    if package_type == "core":
                        self.errors.append(f"Timeout installing core package: {package_name}")
                    else:
                        self.warnings.append(f"Timeout installing optional package: {package_name}")
                        
                except Exception as e:
                    print(f"{Colors.RED}❌ (error){Colors.END}")
                    if package_type == "core":
                        self.errors.append(f"Error installing core package {package_name}: {str(e)}")
                    else:
                        self.warnings.append(f"Error installing optional package {package_name}: {str(e)}")
            
            return success_count
        
        # Install packages
        core_success = install_package_list(core_packages, "core")
        optional_success = install_package_list(optional_packages, "optional")
        
        # Summary
        core_total = len(core_packages)
        optional_total = len(optional_packages)
        
        print(f"  {Colors.WHITE}Core Dependencies: {core_success}/{core_total}{Colors.END}")
        print(f"  {Colors.WHITE}Optional Dependencies: {optional_success}/{optional_total}{Colors.END}")
        
        core_ok = core_success >= (core_total * 0.8)  # 80% of core packages must work
        return core_ok, core_success, optional_success
    
    def setup_environment_file(self) -> bool:
        """Create and configure .env file"""
        print(f"{Colors.BLUE}⚙️  Setting up Environment Configuration...{Colors.END}")
        
        env_example = Path(".env.example")
        env_file = Path(".env")
        
        # Create .env.example if it doesn't exist
        if not env_example.exists():
            env_content = """# Desi Debate Environment Configuration
# OpenAI API Key (optional - system works with fallbacks)
OPENAI_API_KEY=your-openai-api-key-here

# Debug mode
DEBUG=false

# Flask configuration
FLASK_ENV=production
FLASK_DEBUG=false

# Model configurations
USE_GPU=false
BATCH_SIZE=32
MAX_TOKENS=150
"""
            env_example.write_text(env_content)
            print(f"  {Colors.GREEN}✓ Created .env.example{Colors.END}")
        
        # Create .env if it doesn't exist
        if not env_file.exists():
            env_example_content = env_example.read_text()
            env_file.write_text(env_example_content)
            print(f"  {Colors.GREEN}✓ Created .env file{Colors.END}")
            print(f"  {Colors.YELLOW}⚠️  Add your OpenAI API key to .env for enhanced features{Colors.END}")
            self.warnings.append("OpenAI API key not configured - using fallback mode")
        else:
            print(f"  {Colors.GREEN}✓ .env file exists{Colors.END}")
        
        # Validate environment
        try:
            from dotenv import load_dotenv
            load_dotenv()
            
            api_key = os.getenv("OPENAI_API_KEY")
            if api_key and api_key != "your-openai-api-key-here" and len(api_key) > 10:
                print(f"  {Colors.GREEN}✓ OpenAI API key configured{Colors.END}")
            else:
                print(f"  {Colors.YELLOW}⚠️  OpenAI API key not set (fallback mode enabled){Colors.END}")
                self.warnings.append("System will use fallback implementations without API key")
                
        except Exception as e:
            print(f"  {Colors.RED}❌ Environment validation failed: {str(e)}{Colors.END}")
            self.errors.append(f"Environment setup error: {str(e)}")
            return False
        
        return True
    
    def run_system_tests(self) -> bool:
        """Run comprehensive system validation tests"""
        print(f"{Colors.BLUE}🧪 Running System Validation Tests...{Colors.END}")
        
        # Add src to Python path
        sys.path.insert(0, str(Path(__file__).parent / "src"))
        
        tests = [
            ("Import Tests", self._test_imports),
            ("Configuration Loading", self._test_config),
            ("Agent Initialization", self._test_agents),
            ("RAG Retriever", self._test_retriever),
            ("Orchestrator", self._test_orchestrator),
            ("Web Interface", self._test_flask_import)
        ]
        
        passed_tests = 0
        total_tests = len(tests)
        
        for test_name, test_func in tests:
            print(f"  Testing {test_name}...", end=' ')
            try:
                if test_func():
                    print(f"{Colors.GREEN}✓{Colors.END}")
                    passed_tests += 1
                else:
                    print(f"{Colors.RED}❌{Colors.END}")
            except Exception as e:
                print(f"{Colors.RED}❌ ({str(e)[:50]}...){Colors.END}")
                self.errors.append(f"{test_name} failed: {str(e)}")
        
        success_rate = passed_tests / total_tests
        print(f"  {Colors.WHITE}Test Results: {passed_tests}/{total_tests} passed ({success_rate:.1%}){Colors.END}")
        
        return success_rate >= 0.8  # 80% of tests must pass
    
    def _test_imports(self) -> bool:
        """Test core module imports"""
        try:
            from src.utils.config_loader import ConfigLoader
            from src.gpt_interface.gpt_client import chat
            from src.agents.agent_a import AgentA
            from src.rag.simple_retriever import SimpleRetriever
            from src.orchestrator.parallel_orchestrator import ParallelOrchestrator
            return True
        except ImportError:
            return False
    
    def _test_config(self) -> bool:
        """Test configuration loading"""
        try:
            from src.utils.config_loader import ConfigLoader
            config = ConfigLoader.load('debate')
            return len(config) > 0
        except:
            return False
    
    def _test_agents(self) -> bool:
        """Test agent initialization"""
        try:
            from src.agents.agent_a import AgentA
            agent = AgentA()
            return hasattr(agent, 'name') and hasattr(agent, 'select_action')
        except:
            return False
    
    def _test_retriever(self) -> bool:
        """Test RAG retriever"""
        try:
            from src.rag.simple_retriever import SimpleRetriever
            retriever = SimpleRetriever()
            results = retriever.retrieve("test", top_k=1)
            return isinstance(results, list)
        except:
            return False
    
    def _test_orchestrator(self) -> bool:
        """Test orchestrator initialization"""
        try:
            from src.orchestrator.parallel_orchestrator import ParallelOrchestrator
            orchestrator = ParallelOrchestrator()
            return hasattr(orchestrator, 'initialize_agents')
        except:
            return False
    
    def _test_flask_import(self) -> bool:
        """Test Flask application import"""
        try:
            import flask
            return True
        except ImportError:
            return False
    
    def create_startup_scripts(self) -> bool:
        """Create convenient startup scripts"""
        print(f"{Colors.BLUE}🚀 Creating Startup Scripts...{Colors.END}")
        
        # Ensure scripts directory exists
        scripts_dir = Path("scripts")
        scripts_dir.mkdir(exist_ok=True)
        
        # Windows batch script
        windows_script = scripts_dir / "hackathon_demo.bat"
        windows_content = """@echo off
echo Starting Desi Debate Hackathon Demo...
echo =====================================
echo.
echo Opening web browser to http://localhost:5000
echo Press Ctrl+C to stop the server
echo.
start http://localhost:5000
python run_flask.py
pause
"""
        windows_script.write_text(windows_content)
        print(f"  {Colors.GREEN}✓ Created {windows_script}{Colors.END}")
        
        # Unix shell script
        unix_script = scripts_dir / "hackathon_demo.sh"
        unix_content = """#!/bin/bash
echo "Starting Desi Debate Hackathon Demo..."
echo "====================================="
echo ""
echo "Opening web browser to http://localhost:5000"
echo "Press Ctrl+C to stop the server"
echo ""

# Try to open browser
if command -v xdg-open > /dev/null; then
    xdg-open http://localhost:5000 &
elif command -v open > /dev/null; then
    open http://localhost:5000 &
fi

python run_flask.py
"""
        unix_script.write_text(unix_content)
        
        # Make executable on Unix systems
        if platform.system() != "Windows":
            try:
                os.chmod(unix_script, 0o755)
            except:
                pass
        
        print(f"  {Colors.GREEN}✓ Created {unix_script}{Colors.END}")
        
        return True
    
    def print_summary(self, setup_success: bool):
        """Print comprehensive setup summary"""
        elapsed_time = time.time() - self.start_time
        
        print(f"\n{Colors.PURPLE}{'='*60}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.CYAN}🏆 HACKATHON SETUP COMPLETE 🏆{Colors.END}")
        print(f"{Colors.PURPLE}{'='*60}{Colors.END}")
        
        print(f"{Colors.WHITE}Setup Time: {elapsed_time:.1f} seconds{Colors.END}")
        
        if setup_success:
            print(f"{Colors.GREEN}{Colors.BOLD}✅ SYSTEM READY FOR DEMONSTRATION!{Colors.END}")
            print(f"\n{Colors.BLUE}🚀 Quick Start Options:{Colors.END}")
            print(f"  1. {Colors.WHITE}python run_flask.py{Colors.END}")
            print(f"  2. {Colors.WHITE}scripts/hackathon_demo.bat{Colors.END} (Windows)")
            print(f"  3. {Colors.WHITE}./scripts/hackathon_demo.sh{Colors.END} (Linux/Mac)")
            
            print(f"\n{Colors.BLUE}🌐 Demo URL:{Colors.END}")
            print(f"  {Colors.WHITE}http://localhost:5000{Colors.END}")
            
            print(f"\n{Colors.BLUE}📋 Demo Topics to Try:{Colors.END}")
            print(f"  • Should AI development be regulated by government?")
            print(f"  • Is social media's impact on society positive or negative?")
            print(f"  • Should universal basic income be implemented?")
            
            if self.warnings:
                print(f"\n{Colors.YELLOW}⚠️  Warnings ({len(self.warnings)}):{Colors.END}")
                for warning in self.warnings[:3]:  # Show first 3 warnings
                    print(f"  • {warning}")
                if len(self.warnings) > 3:
                    print(f"  • ... and {len(self.warnings) - 3} more")
        else:
            print(f"{Colors.RED}{Colors.BOLD}❌ SETUP FAILED{Colors.END}")
            print(f"\n{Colors.RED}Errors ({len(self.errors)}):{Colors.END}")
            for error in self.errors:
                print(f"  • {error}")
        
        print(f"\n{Colors.BLUE}📚 Documentation:{Colors.END}")
        print(f"  • README.md - Project overview and features")
        print(f"  • DEMO.md - Structured demo guide for judges")
        print(f"  • ARCHITECTURE.md - Technical deep dive")
        print(f"  • TROUBLESHOOTING.md - Common issues and solutions")
        
        print(f"\n{Colors.PURPLE}{'='*60}{Colors.END}")
        
        if setup_success:
            print(f"{Colors.GREEN}Ready to impress the judges! 🎭✨{Colors.END}")
        else:
            print(f"{Colors.RED}Please fix the errors above and run setup again.{Colors.END}")
        
        print(f"{Colors.PURPLE}{'='*60}{Colors.END}\n")
    
    def run_setup(self) -> bool:
        """Run complete hackathon setup process"""
        self.print_header()
        
        steps = [
            ("Python Version Check", self.check_python_version),
            ("Directory Structure", self.check_directory_structure),
            ("Dependency Installation", lambda: self.install_dependencies()[0]),
            ("Environment Configuration", self.setup_environment_file),
            ("System Validation", self.run_system_tests),
            ("Startup Scripts", self.create_startup_scripts)
        ]
        
        success_count = 0
        total_steps = len(steps)
        
        for step_name, step_func in steps:
            print(f"{Colors.BOLD}Step {success_count + 1}/{total_steps}: {step_name}{Colors.END}")
            try:
                if step_func():
                    success_count += 1
                    print(f"{Colors.GREEN}✅ {step_name} completed successfully{Colors.END}\n")
                else:
                    print(f"{Colors.RED}❌ {step_name} failed{Colors.END}\n")
            except Exception as e:
                print(f"{Colors.RED}❌ {step_name} failed with exception: {str(e)}{Colors.END}\n")
                self.errors.append(f"{step_name} failed: {str(e)}")
        
        setup_success = success_count >= (total_steps * 0.8)  # 80% success rate required
        self.print_summary(setup_success)
        
        return setup_success

def main():
    """Main entry point for hackathon setup"""
    try:
        setup = HackathonSetup()
        success = setup.run_setup()
        return 0 if success else 1
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Setup interrupted by user{Colors.END}")
        return 1
    except Exception as e:
        print(f"\n{Colors.RED}Setup failed with unexpected error: {str(e)}{Colors.END}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
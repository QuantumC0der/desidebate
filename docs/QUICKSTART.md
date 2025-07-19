# Quick Start Guide

*English*

Get Desi Debate up and running in 5 minutes!

## Prerequisites

- Python 3.8+
- 8GB+ RAM
- Git

## Installation Steps

### 1. Clone the Project
```bash
git clone https://github.com/your-username/Desi_Debate.git
cd Desi_Debate
```

### 2. Create Virtual Environment
```bash
# Using conda (recommended)
conda create -n desi_debate python=3.8
conda activate desi_debate

# Or using venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set API Key (Optional)
If you want to use full RAG functionality:
```bash
cp env.example .env
# Edit .env file, add your OpenAI API Key
```

## Quick Run

### Method 1: Use Web UI (Recommended)
```bash
# Windows
scripts\start_flask.bat

# Linux/Mac
chmod +x scripts/start_flask.sh
./scripts/start_flask.sh

# Or run directly
python run_flask.py
```

Open your browser and visit http://localhost:5000

### Method 2: Quick Demo
```bash
# Run demo with default models
python quick_demo.py
```

## Using Web UI

### 1. Initialize System
- Open http://localhost:5000
- System will auto-initialize

### 2. Set Debate Topic
Enter your discussion topic, for example:
- "Should artificial intelligence be regulated by government?"
- "Is universal basic income feasible?"
- "Is social media's impact positive or negative?"

### 3. Start Debate
- Click "Next Round" button
- Observe the debate process between three agents
- Watch real-time stance and belief changes

### 4. Analyze Results
- System automatically determines victory
- Export complete debate records
- View detailed scoring breakdown

### 5. Understanding the Scoring System
The system evaluates debate performance through:
- **Stance Firmness**: Ability to maintain clear position
- **Persuasiveness**: Capacity to influence others' viewpoints
- **Resistance**: Defense ability when under attack
- **Overall Performance**: Whether able to make opponents surrender

For detailed scoring mechanism, see [Debate Scoring System Documentation](DEBATE_SCORING_SYSTEM.md)

## Training Models (Optional)

If you have raw data and want to train your own models:

### Quick Training (Demo Scale)
```bash
# Train small-scale models (~10 minutes)
python train_all.py --all --demo
```

### Full Training
```bash
# Train complete models (~30-60 minutes)
python train_all.py --all
```

## Verify Installation

Run system integrity test:
```bash
python test_system_integrity.py
```

Expected output:
```
✅ GPT Interface Test Passed
✅ RAG System Test Passed
✅ GNN System Test Passed
✅ RL System Test Passed
✅ System Integration Test Passed
```

## FAQ

### Q: Can I run without GPU?
A: Yes! The system will automatically use CPU. Training will be slower, but inference speed impact is minimal.

### Q: Is OpenAI API Key required?
A: Not required. Without API Key, simple indexing will be used with slightly limited functionality.

### Q: How to change debate parameters?
A: Edit `configs/debate.yaml` file to adjust rounds, agent count, etc.

### Q: System using too much memory?
A: You can reduce batch size in config files or use `--demo` mode.

## Next Steps

- Check [Training Guide](TRAINING_GUIDE.md) to learn model training
- Check [API Documentation](API_REFERENCE.md) to integrate into your applications
- Check [Deployment Guide](DEPLOYMENT.md) for production deployment

## Need Help?

- Submit [GitHub Issue](https://github.com/your-username/Desi_Debate/issues)
- Email us at your-email@example.com
- Check [Complete Documentation](../README.md)

---

Congratulations! You have successfully run Desi Debate. Start exploring the world of intelligent debate!
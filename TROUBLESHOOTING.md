# Troubleshooting Guide

## Common Issues and Solutions

### 1. Import Errors

**Problem**: `ModuleNotFoundError` or import errors when running the application.

**Solution**:
```bash
# Make sure you're in the project root directory
cd Desi_Debate

# Run the setup script
python setup_environment.py

# Test basic functionality
python test_basic_functionality.py
```

### 2. OpenAI API Key Issues

**Problem**: "No API key configured" or API authentication errors.

**Solution**:
1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit `.env` and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your-actual-api-key-here
   ```

3. Verify the key is loaded:
   ```bash
   python -c "from dotenv import load_dotenv; import os; load_dotenv(); print('API Key:', os.getenv('OPENAI_API_KEY')[:10] + '...' if os.getenv('OPENAI_API_KEY') else 'Not found')"
   ```

### 3. Missing Dependencies

**Problem**: Package import errors or missing modules.

**Solution**:
```bash
# Install all dependencies
pip install -r requirements.txt

# For GPU support (optional)
pip install faiss-gpu torch-geometric

# For development
pip install pytest black flake8
```

### 4. Flask Server Won't Start

**Problem**: Flask server fails to start or crashes immediately.

**Solution**:
1. Check if port 5000 is available:
   ```bash
   # Windows
   netstat -an | findstr :5000
   
   # Linux/Mac
   lsof -i :5000
   ```

2. Try a different port:
   ```bash
   python run_flask.py --port 5001
   ```

3. Check the logs for specific errors:
   ```bash
   python run_flask.py --debug
   ```

### 5. Model Loading Issues

**Problem**: "Model not found" or model loading errors.

**Solution**:
The system is designed to work without pre-trained models by using fallback implementations:

1. **GNN Model**: Uses dummy social vectors if model files are missing
2. **RL Policy**: Uses heuristic-based strategy selection
3. **RAG Index**: Falls back to simple retriever with basic examples

To train your own models:
```bash
# Train all models
python train_all.py

# Train individual components
python train_all.py --gnn
python train_all.py --rl
python train_all.py --rag
```

### 6. Memory Issues

**Problem**: Out of memory errors or slow performance.

**Solution**:
1. Reduce batch sizes in config files
2. Use CPU instead of GPU:
   ```bash
   export CUDA_VISIBLE_DEVICES=""
   ```
3. Limit the number of parallel workers in `configs/system.yaml`

### 7. Configuration Issues

**Problem**: Configuration loading errors or invalid settings.

**Solution**:
1. Validate YAML syntax:
   ```bash
   python -c "import yaml; yaml.safe_load(open('configs/debate.yaml'))"
   ```

2. Reset to default configuration:
   ```bash
   # Backup current config
   cp configs/debate.yaml configs/debate.yaml.backup
   
   # The system will create default configs if files are missing
   rm configs/debate.yaml
   python test_basic_functionality.py
   ```

### 8. Web Interface Issues

**Problem**: Web interface doesn't load or shows errors.

**Solution**:
1. Check if the Flask server is running:
   ```bash
   curl http://localhost:5000/api/health
   ```

2. Clear browser cache and cookies

3. Check browser console for JavaScript errors

4. Try accessing directly:
   ```
   http://localhost:5000
   ```

### 9. Debate Generation Issues

**Problem**: Agents don't respond or generate poor quality responses.

**Solution**:
1. Verify OpenAI API key is working:
   ```bash
   python -c "from src.gpt_interface.gpt_client import chat; print(chat('Hello'))"
   ```

2. Check agent initialization:
   ```bash
   python test_basic_functionality.py
   ```

3. Review debate configuration in `configs/debate.yaml`

### 10. Performance Issues

**Problem**: Slow response times or timeouts.

**Solution**:
1. Reduce the number of retrieval results in `configs/rag.yaml`
2. Decrease max_tokens in GPT requests
3. Use simpler models or reduce complexity
4. Enable caching in configuration

## Getting Help

If you're still experiencing issues:

1. **Check the logs**: Look in the `logs/` directory for detailed error messages
2. **Run diagnostics**: Use `python test_system_integrity.py` for a comprehensive check
3. **Minimal reproduction**: Try with a fresh environment and minimal configuration
4. **System requirements**: Ensure you have Python 3.8+, sufficient RAM (8GB+), and a stable internet connection

## Debug Mode

For detailed debugging, run with debug flags:

```bash
# Enable debug logging
export DEBUG=true

# Run with verbose output
python run_flask.py --debug --verbose

# Check system integrity
python test_system_integrity.py --verbose
```

## Environment Reset

If all else fails, reset your environment:

```bash
# Clean Python cache
find . -name "*.pyc" -delete
find . -name "__pycache__" -type d -exec rm -rf {} +

# Reinstall dependencies
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

# Reset configuration
rm -rf logs/* cache/*
python setup_environment.py
```
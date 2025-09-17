# 🚀 Quick Start Guide - Spectacular Demo

## Prerequisites

1. **Python 3.8+** installed
2. **OpenAI API key** (free tier works fine)
3. **Internet connection** for API calls

## Setup (2 minutes)

### 1. Install Dependencies
```bash
cd demos
pip install -r requirements.txt
```

### 2. Set Your API Key

**Option A: Edit config file**
Open `demo_config.yaml` and replace:
```yaml
api_key: "your-openai-api-key-here"
```
with your actual API key from https://platform.openai.com/api-keys

**Option B: Environment variable**
```bash
export OPENAI_API_KEY="your-actual-api-key"
```

## Run Your First Demo (30 seconds)

```bash
python simple_bayesian_demo.py
```

This will:
- Process the default physics question
- Show every reasoning step in real-time
- Generate a complete markdown report
- Save everything to `output/spectacular_reasoning_trace.md`

## Try Different Examples

```bash
# Physics - Newton's Laws
python simple_bayesian_demo.py --config examples/physics_demo.yaml

# Quantum Physics - Complex topic with more recursion
python simple_bayesian_demo.py --config examples/quantum_demo.yaml

# AI/ML - Neural networks and learning
python simple_bayesian_demo.py --config examples/ai_demo.yaml
```

## What You'll See

### 1. Real-time Processing
```
🧠 Starting Bayesian Evidence Network Processing...
📝 Step 1: Query Analysis
🤖 Querying gpt-3.5-turbo...
✅ Response received (347 characters)
📚 Step 2: Knowledge Retrieval
🔄 Knowledge retrieval needs more depth...
🤖 Querying gpt-3.5-turbo...
✅ Response received (892 characters)
🔍 Step 3: Reasoning Validation
...
```

### 2. Network Visualization
```
🧠 BAYESIAN EVIDENCE NETWORK STATE
==================================================

┌─ query_analysis ✅ ──────────────────┐
│ Type: input_processor            │
│ Confidence: [████████████████████] 0.850 │
│ Uncertainty:[████░░░░░░░░░░░░░░░░] 0.240 │
│ Evidence: 1 items, Recursive: 0     │
│ Analyzes the user query...           │
└────────────────────────────────────────┘
```

### 3. Complete Markdown Report
The generated report includes:
- Executive Summary with quality assessment
- Step-by-step reasoning with confidence changes
- ASCII visualizations and concept diagrams
- Network state analysis with performance metrics
- Technical appendix explaining Bayesian inference

## Create Your Own Configuration

Copy `demo_config.yaml` and modify:

```yaml
query:
  prompt: "Your question here"
  context: 
    - "Context about your question"
    - "What kind of explanation you need"

bayesian_network:
  convergence_threshold: 0.8    # How confident before accepting
  uncertainty_threshold: 0.3    # How much uncertainty is OK
  max_recursive_depth: 3        # Max recursive loops

output:
  filename: "my_custom_analysis.md"
  detail_level: "verbose"       # verbose, normal, minimal
```

## Understanding the Output

### Confidence Scores (0.0 - 1.0)
- **0.9+**: Excellent - Very high confidence
- **0.8-0.9**: Good - High confidence
- **0.7-0.8**: Satisfactory - Adequate confidence
- **0.6-0.7**: Moderate - Some uncertainty
- **<0.6**: Poor - High uncertainty

### Node States
- ✅ **Converged**: Node satisfied with current evidence
- 🔄 **Recursive Loop**: Node gathering more evidence
- ❓ **Needs More Evidence**: Low confidence, needs input
- ❌ **Failed**: Node encountered error

### Network Coherence
Overall system confidence calculated by weighing all nodes:
- Higher coherence = more reliable results
- Lower coherence = treat results cautiously

## Troubleshooting

### "Authentication failed"
- Check your OpenAI API key is correct
- Make sure you have credits in your OpenAI account

### "Rate limit hit"
- The demo automatically waits and retries
- Free tier has request limits - be patient

### "No module named..."
- Run `pip install -r requirements.txt`
- Make sure you're in the demos directory

### Empty or error responses
- Check your internet connection
- Verify OpenAI service status
- Try a simpler query first

## What Makes This Special?

Unlike regular chatbots, this demo shows you:

1. **Every reasoning step** - Nothing is hidden
2. **Confidence tracking** - How sure the system is
3. **Recursive processing** - When it needs more evidence
4. **Evidence accumulation** - How beliefs change over time
5. **Visual validation** - Diagrams that prove understanding

This is a simplified version of the full Spectacular system's Bayesian Evidence Network with complete transparency for learning and debugging.

## Next Steps

Once you understand how this works, explore:
- The full Spectacular system with web interface
- Advanced Bayesian network concepts
- Custom node types and validation methods
- Integration with your own knowledge bases

Happy reasoning! 🧠✨

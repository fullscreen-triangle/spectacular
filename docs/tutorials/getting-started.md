---
layout: page
title: Getting Started
permalink: /tutorials/getting-started/
---

# Getting Started with Spectacular

## Quick Start Guide

This tutorial will walk you through setting up and using Spectacular for your first data visualization project.

## Installation

### Prerequisites
- Python 3.8+
- Node.js 16+
- 16+ GB RAM
- CUDA-compatible GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/spectacular.git
cd spectacular

# Install dependencies
make setup

# Install cognitive modules
pip install scikit-fuzzy pymc tensorflow-probability gym stable-baselines3

# Install Hugging Face integration
pip install transformers datasets torch torchvision

# Configure environment
cp .env.example .env
# Edit .env with your API keys and configuration

# Initialize the system
make init-system
```

## Your First Visualization

### Basic Usage

```python
from spectacular import SpectacularSystem

# Initialize the system
system = SpectacularSystem()

# Create a simple visualization
result = await system.generate_visualization(
    query="Create a bar chart showing sales by region",
    context={
        "data_schema": {
            "region": "string",
            "sales": "number"
        },
        "data": [
            {"region": "North", "sales": 100000},
            {"region": "South", "sales": 85000},
            {"region": "East", "sales": 120000},
            {"region": "West", "sales": 95000}
        ]
    }
)

# Get the generated D3.js code
print(result.d3_code)
```

### Advanced Example

```python
# Complex visualization with multiple cognitive modules
result = await system.generate_visualization(
    query="Create an interactive scatter plot with regression line showing correlation between temperature and ice cream sales, with seasonal color coding",
    context={
        "data_schema": {
            "temperature": "number",
            "sales": "number", 
            "season": "string",
            "date": "date"
        },
        "constraints": {
            "interactivity": True,
            "statistical_analysis": True,
            "accessibility": True
        },
        "user_expertise": "intermediate"
    }
)

# Access reasoning trace
for step in result.reasoning_trace:
    print(f"Module: {step.module}")
    print(f"Confidence: {step.confidence}")
    print(f"Output: {step.output}")
```

## Understanding the System

### Cognitive Modules

Spectacular uses five specialized modules:

1. **Pretoria (Fuzzy Logic)**: Handles uncertainty and linguistic variables
2. **Mzekezeke (Bayesian)**: Probabilistic reasoning and evidence integration
3. **Zengeza (MDP)**: Sequential decision making and optimization
4. **Nicotine (Sketching)**: Context maintenance and progressive refinement
5. **Diadochi (Model Combination)**: Intelligent coordination of multiple AI models

### Configuration

```yaml
# spectacular_config.yaml
metacognitive:
  depth: 3
  timeout: 30
  debug_mode: false

modules:
  pretoria:
    rule_base: "rules/d3_generation.rules"
    inference_method: "mamdani"
  
  mzekezeke:
    network_file: "networks/visualization_network.json"
    inference_algorithm: "variable_elimination"
  
  zengeza:
    policy_file: "policies/visualization_policy.pkl"
    exploration_rate: 0.1
```

## Common Patterns

### Data-Driven Visualizations

```python
# Automatic chart type selection
result = await system.generate_visualization(
    query="Visualize this data effectively",
    context={"data": your_data}
)
```

### Interactive Dashboards

```python
# Multi-panel dashboard
result = await system.generate_visualization(
    query="Create a dashboard with multiple related charts",
    context={
        "data": complex_dataset,
        "layout": "grid",
        "interactions": ["brushing", "linking", "filtering"]
    }
)
```

### Custom Styling

```python
# Branded visualizations
result = await system.generate_visualization(
    query="Create a professional chart for quarterly report",
    context={
        "data": quarterly_data,
        "style": {
            "theme": "corporate",
            "colors": ["#1f77b4", "#ff7f0e", "#2ca02c"],
            "font_family": "Arial"
        }
    }
)
```

## Best Practices

### Query Writing

**Good queries are:**
- Specific about the visualization type
- Clear about the data relationships to show
- Explicit about interactivity needs
- Descriptive about the target audience

**Examples:**

```python
# ✅ Good
"Create an interactive scatter plot showing the correlation between advertising spend and sales revenue, with points colored by campaign type"

# ❌ Too vague
"Make a chart with my data"
```

### Data Preparation

```python
# Ensure data is clean and well-structured
context = {
    "data_schema": {
        "x_variable": "number",
        "y_variable": "number",
        "category": "string"
    },
    "data_quality": "clean",  # or "noisy", "sparse"
    "data_size": "medium"     # or "small", "large"
}
```

### Performance Optimization

```python
# For large datasets
system = SpectacularSystem(
    config={
        "optimization": {
            "lazy_loading": True,
            "data_sampling": True,
            "progressive_rendering": True
        }
    }
)
```

## Troubleshooting

### Common Issues

1. **Low-quality output**: Provide more specific queries and better data context
2. **Slow generation**: Reduce cognitive depth or enable caching
3. **Memory issues**: Use data sampling for large datasets

### Debugging

```python
# Enable debug mode
system = SpectacularSystem(debug_mode=True)

# Check reasoning trace
for step in result.reasoning_trace:
    if step.confidence < 0.7:
        print(f"Low confidence in {step.module}: {step.output}")
```

### Getting Help

- Check the [API Reference](../api-reference.md)
- Review [Cognitive Architecture](../cognitive-architecture.md) for advanced usage
- See [Examples](../examples/) for common patterns
- Join our [Community Discord](https://discord.gg/spectacular)

## Next Steps

1. Try the [Advanced Tutorial](advanced-tutorial.md)
2. Explore the [Example Gallery](../examples/)
3. Read about [Cognitive Architecture](../cognitive-architecture.md)
4. Learn about [Customization](customization.md)

Happy visualizing! 🎨✨ 
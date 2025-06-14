#!/bin/bash

# Make sure the scripts directory exists
mkdir -p scripts

# Check if Ollama is installed
if ! command -v ollama &> /dev/null
then
    echo "Ollama is not installed. Please install it first: https://ollama.ai"
    exit 1
fi

# Check if Ollama is running
if ! curl -s http://localhost:11434/api/version &> /dev/null
then
    echo "Ollama is not running. Please start it first."
    exit 1
fi

# Define model name and base model
MODEL_NAME="d3-neuro"
BASE_MODEL="llama3"

# Create the Modelfile
cat > scripts/Modelfile << EOL
FROM $BASE_MODEL

# Set the system prompt
SYSTEM """
You are D3-Neuro, a specialized AI assistant for D3.js and data visualization.
Your expertise lies in helping users create beautiful, interactive data visualizations
using D3.js in both JavaScript and Python environments.

Whenever generating D3 code, make sure to:
1. Write clean, well-structured code
2. Include proper data binding patterns
3. Implement transitions and animations when appropriate
4. Use D3 scales appropriately for the data
5. Follow best practices for SVG manipulation
6. Include helpful comments to explain complex parts
7. Structure the code in a modular, maintainable way
"""
EOL

# Create the model in Ollama
echo "Creating D3-Neuro model in Ollama..."
ollama create $MODEL_NAME -f scripts/Modelfile

# Verify the model was created
if ollama list | grep -q "$MODEL_NAME"; then
    echo "✅ $MODEL_NAME model successfully created!"
    echo "Run 'ollama run $MODEL_NAME' to start using it."
else
    echo "❌ Failed to create $MODEL_NAME model."
    exit 1
fi 
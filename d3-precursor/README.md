# D3-Precursor

Knowledge extraction and processing for D3 visualizations.

## Overview

D3-Precursor is a Python library for extracting, processing, and structuring knowledge from D3.js codebase and documentation. It's designed to build a comprehensive knowledge base for the D3-Neuro domain-specific LLM.

## Features

- Parse JavaScript and Python D3 code to extract API structures
- Process D3 documentation to extract API documentation and examples
- Extract common visualization patterns and techniques
- Generate embeddings for semantic search
- Build a structured knowledge base

## Installation

### From Source

```bash
git clone https://github.com/yourusername/d3-neuro.git
cd d3-neuro/d3-precursor
pip install -e .
```

### Requirements

See `requirements.txt` for the full list of dependencies.

## Usage

### Process D3 Codebase

```bash
# From the scripts directory
python process_codebase.py --d3-js-dir /path/to/d3/js --d3-py-dir /path/to/d3/python --docs-dir /path/to/d3/docs --examples-dir /path/to/d3/examples
```

### Use in Python

```python
from d3_precursor.parsers import D3JSParser, D3DocParser
from d3_precursor.embeddings import D3Encoder

# Parse D3.js codebase
js_parser = D3JSParser("/path/to/d3/js")
api_defs = js_parser.parse_directory()

# Parse documentation
doc_parser = D3DocParser("/path/to/d3/docs")
api_docs = doc_parser.process_directory()

# Generate embeddings
encoder = D3Encoder()
embeddings = encoder.encode_batch(api_docs['api_elements'], 'api')
```

## Module Structure

- `parsers/`: Code for parsing D3 code and documentation
- `extractors/`: Extract structured knowledge from parsed data
- `processors/`: Clean and normalize extracted data
- `knowledge_base/`: Define schema and storage for knowledge
- `embeddings/`: Generate vector embeddings for knowledge items
- `scripts/`: Utility scripts for processing D3 codebase

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Commit your changes (`git commit -am 'Add my feature'`)
4. Push to the branch (`git push origin feature/my-feature`)
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
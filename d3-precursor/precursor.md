# D3-Precursor Implementation Specifications

## Overview

The `d3-precursor` module is responsible for extracting, processing, and structuring D3.js knowledge. It analyzes the D3 codebase to build a comprehensive knowledge base with semantic embeddings for use by the LLM.

## Core Components

### 1. Parsers

The parsing modules extract structured information from D3 code and documentation.

#### `js_parser.py`

```python
import os
import json
import esprima  # For JavaScript parsing
from typing import Dict, List, Any, Optional

class D3JSParser:
    """Parse D3.js source code to extract API structure and usage patterns."""
    
    def __init__(self, source_path: str):
        self.source_path = source_path
        self.ast_cache = {}
        self.api_definitions = {}
        self.module_structure = {}
        
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """Parse a JavaScript file into AST representation."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        try:
            ast = esprima.parseScript(content, {'loc': True, 'comment': True})
            self.ast_cache[file_path] = ast
            return ast.toDict()
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return {}
            
    def parse_directory(self, dir_path: Optional[str] = None) -> Dict[str, Any]:
        """Parse all JavaScript files in a directory."""
        if dir_path is None:
            dir_path = self.source_path
            
        results = {}
        for root, _, files in os.walk(dir_path):
            for file in files:
                if file.endswith('.js'):
                    file_path = os.path.join(root, file)
                    results[file_path] = self.parse_file(file_path)
                    
        return results
    
    def extract_api_definitions(self) -> Dict[str, Any]:
        """Extract API definitions from parsed ASTs."""
        definitions = {}
        
        for file_path, ast in self.ast_cache.items():
            # Process declarations, exports, and function definitions
            file_definitions = self._process_declarations(ast)
            definitions[file_path] = file_definitions
            
        self.api_definitions = definitions
        return definitions
    
    def _process_declarations(self, ast: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process declarations in AST to extract API components."""
        definitions = []
        
        # Implementation to traverse AST and extract:
        # - Function declarations
        # - Variable declarations with function expressions
        # - Export statements
        # - Class definitions
        
        return definitions
    
    def analyze_usage_patterns(self) -> Dict[str, Any]:
        """Identify common D3 usage patterns in the codebase."""
        patterns = {
            'method_chaining': [],
            'selections': [],
            'data_binding': [],
            'scales': [],
            'transitions': []
        }
        
        # Implementation to detect common patterns in D3 code
        
        return patterns
```

#### `py_parser.py`

```python
import ast
import os
from typing import Dict, List, Any, Optional

class D3PyParser:
    """Parse Python D3 bindings and examples."""
    
    def __init__(self, source_path: str):
        self.source_path = source_path
        self.ast_cache = {}
        self.api_definitions = {}
        
    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """Parse a Python file into AST representation."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        try:
            tree = ast.parse(content)
            self.ast_cache[file_path] = tree
            return self._ast_to_dict(tree)
        except Exception as e:
            print(f"Error parsing {file_path}: {e}")
            return {}
    
    def _ast_to_dict(self, node) -> Dict[str, Any]:
        """Convert AST node to dictionary representation."""
        # Implementation to convert Python AST to dictionary
        return {}
        
    def extract_api_definitions(self) -> Dict[str, Any]:
        """Extract API definitions from parsed Python files."""
        definitions = {}
        
        # Implementation to extract class and function definitions
        
        return definitions
```

#### `doc_parser.py`

```python
import os
import re
import markdown
from bs4 import BeautifulSoup
from typing import Dict, List, Any, Optional

class D3DocParser:
    """Parse D3 documentation in Markdown and HTML formats."""
    
    def __init__(self, docs_path: str):
        self.docs_path = docs_path
        self.api_docs = {}
        self.examples = []
        
    def parse_markdown(self, file_path: str) -> Dict[str, Any]:
        """Parse markdown documentation into structured format."""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        html = markdown.markdown(content)
        soup = BeautifulSoup(html, 'html.parser')
        
        # Extract structured information from markdown
        result = {
            'title': self._extract_title(soup),
            'sections': self._extract_sections(soup),
            'api_elements': self._extract_api_elements(soup),
            'examples': self._extract_examples(soup)
        }
        
        return result
    
    def _extract_title(self, soup) -> str:
        """Extract title from parsed document."""
        h1 = soup.find('h1')
        return h1.text if h1 else ""
    
    def _extract_sections(self, soup) -> List[Dict[str, Any]]:
        """Extract sections from document."""
        sections = []
        # Implementation to extract sections
        return sections
    
    def _extract_api_elements(self, soup) -> List[Dict[str, Any]]:
        """Extract API documentation elements."""
        elements = []
        # Implementation to extract API elements
        return elements
    
    def _extract_examples(self, soup) -> List[Dict[str, Any]]:
        """Extract code examples from documentation."""
        examples = []
        # Implementation to extract code examples
        return examples
    
    def extract_api_docs(self) -> Dict[str, Any]:
        """Extract and structure API documentation."""
        api_docs = {}
        
        # Process all documentation files
        for root, _, files in os.walk(self.docs_path):
            for file in files:
                if file.endswith('.md'):
                    file_path = os.path.join(root, file)
                    parsed = self.parse_markdown(file_path)
                    
                    # Map API elements to structured docs
                    for element in parsed['api_elements']:
                        api_name = element.get('name', '')
                        if api_name:
                            api_docs[api_name] = element
        
        self.api_docs = api_docs
        return api_docs
```

### 2. Extractors

The extractor modules transform parsed data into structured knowledge.

#### `api_extractor.py`

```python
from typing import Dict, List, Any

class D3APIExtractor:
    """Extract and categorize D3 API components."""
    
    def __init__(self, js_parser_results: Dict[str, Any], py_parser_results: Dict[str, Any] = None):
        self.js_parser_results = js_parser_results
        self.py_parser_results = py_parser_results or {}
        self.api_methods = {}
        self.api_modules = {}
        self.api_categories = {}
        
    def extract_methods(self) -> Dict[str, Any]:
        """Extract method signatures, parameters, and return types."""
        methods = {}
        
        # Process JavaScript API methods
        for file_path, definitions in self.js_parser_results.items():
            file_methods = self._process_js_methods(definitions)
            methods.update(file_methods)
            
        # Process Python API methods if available
        if self.py_parser_results:
            for file_path, definitions in self.py_parser_results.items():
                file_methods = self._process_py_methods(definitions)
                methods.update(file_methods)
                
        self.api_methods = methods
        return methods
    
    def _process_js_methods(self, definitions: Dict[str, Any]) -> Dict[str, Any]:
        """Process JavaScript API method definitions."""
        methods = {}
        
        # Implementation to extract method signatures
        
        return methods
    
    def _process_py_methods(self, definitions: Dict[str, Any]) -> Dict[str, Any]:
        """Process Python API method definitions."""
        methods = {}
        
        # Implementation to extract method signatures
        
        return methods
    
    def extract_modules(self) -> Dict[str, Any]:
        """Extract and structure D3 module organization."""
        modules = {}
        
        # Implementation to identify module structure
        
        self.api_modules = modules
        return modules
    
    def categorize_api(self) -> Dict[str, Any]:
        """Categorize API by functionality."""
        categories = {
            'selection': [],
            'scales': [],
            'axes': [],
            'shapes': [],
            'layouts': [],
            'transitions': [],
            'data_loading': [],
            'formatting': [],
            'geo': [],
            'colors': [],
            'other': []
        }
        
        # Categorize API methods
        for method_name, method_info in self.api_methods.items():
            category = self._determine_category(method_name, method_info)
            categories[category].append(method_name)
            
        self.api_categories = categories
        return categories
    
    def _determine_category(self, method_name: str, method_info: Dict[str, Any]) -> str:
        """Determine the category for a method."""
        # Implementation to classify methods into categories
        return 'other'
```

#### `pattern_extractor.py`

```python
from typing import Dict, List, Any

class D3PatternExtractor:
    """Extract common usage patterns from D3 code."""
    
    def __init__(self, parser_results: Dict[str, Any]):
        self.parser_results = parser_results
        self.patterns = {}
        
    def extract_patterns(self) -> Dict[str, Any]:
        """Extract common D3 usage patterns."""
        patterns = {
            'method_chaining': self._extract_method_chaining(),
            'data_binding': self._extract_data_binding(),
            'enter_update_exit': self._extract_enter_update_exit(),
            'scale_usage': self._extract_scale_usage(),
            'transition_patterns': self._extract_transition_patterns()
        }
        
        self.patterns = patterns
        return patterns
    
    def _extract_method_chaining(self) -> List[Dict[str, Any]]:
        """Extract method chaining patterns."""
        patterns = []
        
        # Implementation to identify method chaining
        
        return patterns
    
    def _extract_data_binding(self) -> List[Dict[str, Any]]:
        """Extract data binding patterns."""
        patterns = []
        
        # Implementation to identify data binding
        
        return patterns
    
    def _extract_enter_update_exit(self) -> List[Dict[str, Any]]:
        """Extract enter-update-exit patterns."""
        patterns = []
        
        # Implementation to identify enter-update-exit patterns
        
        return patterns
    
    def _extract_scale_usage(self) -> List[Dict[str, Any]]:
        """Extract scale usage patterns."""
        patterns = []
        
        # Implementation to identify scale usage
        
        return patterns
    
    def _extract_transition_patterns(self) -> List[Dict[str, Any]]:
        """Extract transition patterns."""
        patterns = []
        
        # Implementation to identify transition patterns
        
        return patterns
```

### 3. Processors

The processor modules clean, normalize, and structure the extracted data.

#### `code_processor.py`

```python
import re
from typing import List, Dict, Any

class CodeProcessor:
    """Process code for normalization and segmentation."""
    
    def __init__(self):
        self.js_normalizers = [
            self._remove_comments,
            self._normalize_whitespace,
            self._normalize_semicolons
        ]
        self.py_normalizers = [
            self._remove_comments,
            self._normalize_whitespace,
            self._normalize_indentation
        ]
        
    def normalize_js(self, code: str) -> str:
        """Normalize JavaScript code formatting."""
        for normalizer in self.js_normalizers:
            code = normalizer(code)
        return code
    
    def normalize_python(self, code: str) -> str:
        """Normalize Python code formatting."""
        for normalizer in self.py_normalizers:
            code = normalizer(code)
        return code
    
    def _remove_comments(self, code: str) -> str:
        """Remove comments from code."""
        # Implementation to remove comments
        return code
    
    def _normalize_whitespace(self, code: str) -> str:
        """Normalize whitespace in code."""
        # Implementation to normalize whitespace
        return code
    
    def _normalize_semicolons(self, code: str) -> str:
        """Normalize semicolon usage in JavaScript."""
        # Implementation to normalize semicolons
        return code
    
    def _normalize_indentation(self, code: str) -> str:
        """Normalize indentation in Python code."""
        # Implementation to normalize indentation
        return code
    
    def segment_by_function(self, code: str, language: str = 'js') -> List[Dict[str, Any]]:
        """Segment code into function-level blocks."""
        segments = []
        
        if language == 'js':
            segments = self._segment_js_by_function(code)
        elif language == 'python':
            segments = self._segment_py_by_function(code)
            
        return segments
    
    def _segment_js_by_function(self, code: str) -> List[Dict[str, Any]]:
        """Segment JavaScript code by function."""
        segments = []
        
        # Implementation to segment JS code by function
        
        return segments
    
    def _segment_py_by_function(self, code: str) -> List[Dict[str, Any]]:
        """Segment Python code by function."""
        segments = []
        
        # Implementation to segment Python code by function
        
        return segments
```

### 4. Knowledge Base

The knowledge base modules define the schema and storage for the structured knowledge.

#### `schema.py`

```python
from enum import Enum
from typing import Dict, List, Any, Optional, Union
from pydantic import BaseModel, Field

class ApiElementType(str, Enum):
    """Type of API element."""
    FUNCTION = "function"
    METHOD = "method"
    CLASS = "class"
    CONSTANT = "constant"
    MODULE = "module"

class ParameterType(str, Enum):
    """Type of function parameter."""
    STRING = "string"
    NUMBER = "number"
    BOOLEAN = "boolean"
    ARRAY = "array"
    OBJECT = "object"
    FUNCTION = "function"
    ANY = "any"

class Parameter(BaseModel):
    """Function parameter definition."""
    name: str
    type: Union[ParameterType, List[ParameterType]]
    description: str
    optional: bool = False
    default_value: Optional[Any] = None

class ReturnValue(BaseModel):
    """Function return value definition."""
    type: Union[ParameterType, List[ParameterType]]
    description: str

class Example(BaseModel):
    """Code example definition."""
    code: str
    description: str
    language: str = "javascript"
    
class ApiElement(BaseModel):
    """API element definition."""
    name: str
    type: ApiElementType
    module: str
    description: str
    parameters: List[Parameter] = Field(default_factory=list)
    return_value: Optional[ReturnValue] = None
    examples: List[Example] = Field(default_factory=list)
    related: List[str] = Field(default_factory=list)
    visualization_types: List[str] = Field(default_factory=list)
    
class UsagePattern(BaseModel):
    """Usage pattern definition."""
    name: str
    description: str
    code_snippets: List[Example] = Field(default_factory=list)
    use_cases: List[str] = Field(default_factory=list)
    api_methods: List[str] = Field(default_factory=list)
    
class VisualizationType(BaseModel):
    """Visualization type definition."""
    name: str
    description: str
    components: List[str] = Field(default_factory=list)
    code_patterns: List[str] = Field(default_factory=list)
    examples: List[Example] = Field(default_factory=list)
    api_dependencies: List[str] = Field(default_factory=list)
```

#### `storage.py`

```python
import os
import json
import yaml
from typing import Dict, List, Any, Optional, Union, TypeVar, Generic
from uuid import uuid4

T = TypeVar('T')

class KnowledgeStorage(Generic[T]):
    """Store and retrieve knowledge base items."""
    
    def __init__(self, storage_path: str, format: str = 'json'):
        self.storage_path = storage_path
        self.format = format
        os.makedirs(storage_path, exist_ok=True)
        
    def store_item(self, item: T, item_type: str) -> str:
        """Store a knowledge item by type."""
        item_id = str(uuid4())
        type_dir = os.path.join(self.storage_path, item_type)
        os.makedirs(type_dir, exist_ok=True)
        
        file_path = os.path.join(type_dir, f"{item_id}.{self.format}")
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if self.format == 'json':
                if hasattr(item, 'dict'):
                    json.dump(item.dict(), f, indent=2)
                else:
                    json.dump(item, f, indent=2)
            elif self.format == 'yaml':
                if hasattr(item, 'dict'):
                    yaml.dump(item.dict(), f)
                else:
                    yaml.dump(item, f)
        
        return item_id
    
    def store_batch(self, items: List[T], item_type: str) -> List[str]:
        """Store multiple items of the same type."""
        return [self.store_item(item, item_type) for item in items]
    
    def get_item(self, item_id: str, item_type: str) -> Optional[Dict[str, Any]]:
        """Retrieve an item by ID and type."""
        file_path = os.path.join(self.storage_path, item_type, f"{item_id}.{self.format}")
        
        if not os.path.exists(file_path):
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            if self.format == 'json':
                return json.load(f)
            elif self.format == 'yaml':
                return yaml.safe_load(f)
        
        return None
    
    def get_all_items(self, item_type: str) -> List[Dict[str, Any]]:
        """Retrieve all items of a specific type."""
        type_dir = os.path.join(self.storage_path, item_type)
        
        if not os.path.exists(type_dir):
            return []
        
        items = []
        for file_name in os.listdir(type_dir):
            if file_name.endswith(f'.{self.format}'):
                item_id = file_name.split('.')[0]
                item = self.get_item(item_id, item_type)
                if item:
                    items.append(item)
                    
        return items
```

### 5. Embeddings

The embedding modules generate vector representations for knowledge retrieval.

#### `encoder.py`

```python
import torch
from typing import List, Dict, Any, Union
from sentence_transformers import SentenceTransformer
import numpy as np

class D3Encoder:
    """Generate embeddings from D3 code and documentation."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-mpnet-base-v2"):
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
    def encode_text(self, text: str) -> np.ndarray:
        """Encode text into embeddings."""
        return self.model.encode(text, show_progress_bar=False)
    
    def encode_code(self, code: str) -> np.ndarray:
        """Encode code with special handling for code structures."""
        # For D3-specific code, we might want to preprocess
        # to emphasize API calls and patterns
        processed_code = self._preprocess_code(code)
        return self.encode_text(processed_code)
    
    def _preprocess_code(self, code: str) -> str:
        """Preprocess code to emphasize important structures."""
        # Implementation to highlight API calls, etc.
        return code
    
    def encode_batch(self, items: List[Union[str, Dict[str, Any]]], item_type: str) -> List[np.ndarray]:
        """Encode a batch of items."""
        texts = []
        
        if item_type == 'text':
            texts = items
        elif item_type == 'code':
            texts = [self._preprocess_code(item) for item in items]
        elif item_type == 'api':
            texts = [self._format_api_for_embedding(item) for item in items]
            
        return self.model.encode(texts, show_progress_bar=True)
    
    def _format_api_for_embedding(self, api_item: Dict[str, Any]) -> str:
        """Format API item for embedding generation."""
        # Implementation to format API details
        return ""
```

#### `index.py`

```python
import os
import numpy as np
import faiss
import pickle
from typing import List, Dict, Any, Optional, Tuple

class EmbeddingIndex:
    """Index and search embeddings."""
    
    def __init__(self, index_path: Optional[str] = None, dimension: int = 768):
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(dimension)  # L2 distance index
        self.metadata = []
        self.index_path = index_path
        
        if index_path and os.path.exists(index_path):
            self.load(index_path)
    
    def add_embeddings(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]) -> None:
        """Add embeddings to the index with metadata."""
        if len(embeddings) != len(metadata):
            raise ValueError("Embeddings and metadata must have the same length")
            
        # Convert embeddings to float32 if needed
        if embeddings.dtype != np.float32:
            embeddings = embeddings.astype(np.float32)
            
        self.index.add(embeddings)
        self.metadata.extend(metadata)
        
        if self.index_path:
            self.save(self.index_path)
    
    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[int, float, Dict[str, Any]]]:
        """Search index for similar embeddings."""
        if query_embedding.dtype != np.float32:
            query_embedding = query_embedding.astype(np.float32)
            
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for i, (idx, dist) in enumerate(zip(indices[0], distances[0])):
            if idx < len(self.metadata):
                results.append((idx, float(dist), self.metadata[idx]))
                
        return results
    
    def save(self, path: str) -> None:
        """Save index and metadata to disk."""
        directory = os.path.dirname(path)
        os.makedirs(directory, exist_ok=True)
        
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.metadata", 'wb') as f:
            pickle.dump(self.metadata, f)
    
    def load(self, path: str) -> None:
        """Load index and metadata from disk."""
        if os.path.exists(f"{path}.index") and os.path.exists(f"{path}.metadata"):
            self.index = faiss.read_index(f"{path}.index")
            with open(f"{path}.metadata", 'rb') as f:
                self.metadata = pickle.load(f)
```

## Implementation Plan

### Phase 1: Data Collection and Parsing

1. Fetch D3.js codebase and documentation
2. Implement parsers for JavaScript and Python code
3. Implement documentation parser
4. Test and validate parsing accuracy

### Phase 2: Knowledge Extraction

1. Implement API extractors
2. Implement pattern extractors
3. Implement concept extractors
4. Build relationship mapping

### Phase 3: Knowledge Structuring

1. Define knowledge base schema
2. Implement storage mechanisms
3. Populate knowledge base with extracted information
4. Validate knowledge integrity

### Phase 4: Embedding Generation

1. Setup embedding models
2. Generate embeddings for API definitions
3. Generate embeddings for code patterns
4. Build search indices

### Phase 5: Integration with LLM

1. Implement query processing
2. Build retrieval mechanisms
3. Connect with d3-receptor component
4. Test end-to-end knowledge flow

## Usage Examples

### Parsing D3 Codebase

```python
from d3_precursor.parsers.js_parser import D3JSParser

# Initialize parser with path to D3 codebase
parser = D3JSParser('path/to/d3/src')

# Parse all JavaScript files
ast_results = parser.parse_directory()

# Extract API definitions
api_definitions = parser.extract_api_definitions()

# Analyze usage patterns
patterns = parser.analyze_usage_patterns()
```

### Building Knowledge Base

```python
from d3_precursor.extractors.api_extractor import D3APIExtractor
from d3_precursor.knowledge_base.storage import KnowledgeStorage

# Extract API information
extractor = D3APIExtractor(js_parser_results)
methods = extractor.extract_methods()
modules = extractor.extract_modules()
categories = extractor.categorize_api()

# Store in knowledge base
storage = KnowledgeStorage('path/to/knowledge', format='json')
for method_name, method_info in methods.items():
    storage.store_item(method_info, 'api')
```

### Generating Embeddings

```python
from d3_precursor.embeddings.encoder import D3Encoder
from d3_precursor.embeddings.index import EmbeddingIndex

# Initialize encoder
encoder = D3Encoder()

# Generate embeddings for API items
api_items = storage.get_all_items('api')
api_texts = [encoder._format_api_for_embedding(item) for item in api_items]
api_embeddings = encoder.encode_batch(api_texts, 'text')

# Create searchable index
index = EmbeddingIndex('path/to/indices/api_index', dimension=768)
index.add_embeddings(api_embeddings, api_items)
```

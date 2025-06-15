---
layout: page
title: API Reference
permalink: /api-reference/
---

# Spectacular API Reference

## Table of Contents

1. [Core Architecture](#core-architecture)
2. [Metacognitive Orchestrator API](#metacognitive-orchestrator-api)
3. [Pretoria Module API](#pretoria-module-api)
4. [Mzekezeke Module API](#mzekezeke-module-api)
5. [Zengeza Module API](#zengeza-module-api)
6. [Nicotine Module API](#nicotine-module-api)
7. [Diadochi Module API](#diadochi-module-api)
8. [Hugging Face Integration API](#hugging-face-integration-api)
9. [Client Libraries](#client-libraries)
10. [Error Handling](#error-handling)

## Core Architecture

### SpectacularSystem

The main entry point for the Spectacular metacognitive system.

```python
from spectacular import SpectacularSystem

system = SpectacularSystem(
    config_path="config/spectacular.yaml",
    debug_mode=False,
    metacognitive_depth=3
)
```

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `config_path` | `str` | `"config/default.yaml"` | Path to system configuration file |
| `debug_mode` | `bool` | `False` | Enable debug logging and tracing |
| `metacognitive_depth` | `int` | `3` | Depth of metacognitive reasoning loops |
| `enable_gpu` | `bool` | `True` | Enable GPU acceleration for neural components |

#### Methods

##### `async generate_visualization(query: str, context: Dict = None) -> VisualizationResult`

Generate a D3.js visualization from natural language query.

**Parameters:**
- `query` (str): Natural language description of desired visualization
- `context` (Dict, optional): Additional context including data schema, constraints

**Returns:**
- `VisualizationResult`: Contains generated code, reasoning trace, and metadata

**Example:**
```python
result = await system.generate_visualization(
    query="Create an interactive scatter plot showing correlation between age and income",
    context={
        "data_schema": {"age": "number", "income": "number", "category": "string"},
        "constraints": {"max_points": 1000, "interactive": True}
    }
)
print(result.d3_code)
print(result.confidence_score)
```

## Metacognitive Orchestrator API

### MetacognitiveOrchestrator

Coordinates all system modules through self-aware reasoning.

```python
from spectacular.orchestrator import MetacognitiveOrchestrator

orchestrator = MetacognitiveOrchestrator(
    modules=["pretoria", "mzekezeke", "zengeza", "nicotine", "diadochi"],
    reasoning_strategy="adaptive"
)
```

#### Methods

##### `async orchestrate_query(query: ProcessedQuery) -> OrchestrationResult`

Main orchestration method that coordinates module interactions.

**Parameters:**
- `query` (ProcessedQuery): Preprocessed query with extracted features

**Returns:**
- `OrchestrationResult`: Contains module responses, reasoning chain, and final synthesis

##### `evaluate_reasoning_quality(result: OrchestrationResult) -> QualityMetrics`

Metacognitive evaluation of reasoning quality.

**Parameters:**
- `result` (OrchestrationResult): Result to evaluate

**Returns:**
- `QualityMetrics`: Quality scores and improvement suggestions

## Pretoria Module API

### FuzzyLogicEngine

Implements fuzzy logic programming for prompt generation and code synthesis.

```python
from spectacular.pretoria import FuzzyLogicEngine

fuzzy_engine = FuzzyLogicEngine(
    rule_base_path="rules/d3_generation.rules",
    membership_functions="config/fuzzy_mf.yaml"
)
```

#### Methods

##### `apply_fuzzy_rules(input_variables: Dict) -> FuzzyOutput`

Apply fuzzy rules to input variables.

**Parameters:**
- `input_variables` (Dict): Dictionary of linguistic variables and their values

**Returns:**
- `FuzzyOutput`: Fuzzy inference results with defuzzified outputs

**Example:**
```python
result = fuzzy_engine.apply_fuzzy_rules({
    "query_complexity": 0.8,
    "domain_specificity": 0.6,
    "user_expertise": 0.3
})
print(result.prompt_strategy)  # "multi_step"
print(result.reasoning_depth)  # "deep"
```

##### `add_rule(antecedent: str, consequent: str, weight: float = 1.0)`

Add a new fuzzy rule to the rule base.

**Parameters:**
- `antecedent` (str): Rule condition (IF part)
- `consequent` (str): Rule conclusion (THEN part)
- `weight` (float): Rule weight (0.0-1.0)

## Mzekezeke Module API

### BayesianEvidenceNetwork

Implements Bayesian evidence network with optimizable objective function.

```python
from spectacular.mzekezeke import BayesianEvidenceNetwork

bayesian_net = BayesianEvidenceNetwork(
    network_topology="config/network_structure.json",
    learning_rate=0.01
)
```

#### Methods

##### `update_beliefs(evidence: Dict[str, float]) -> BeliefState`

Update network beliefs based on new evidence.

**Parameters:**
- `evidence` (Dict[str, float]): Evidence variables and their observed values

**Returns:**
- `BeliefState`: Updated belief state with posterior probabilities

##### `optimize_network(training_data: List[EvidenceCase]) -> OptimizationResult`

Optimize network parameters using training data.

**Parameters:**
- `training_data` (List[EvidenceCase]): Training cases with evidence and outcomes

**Returns:**
- `OptimizationResult`: Optimization metrics and updated parameters

## Zengeza Module API

### MarkovDecisionProcess

Implements MDP for state transitions and utility optimization.

```python
from spectacular.zengeza import MarkovDecisionProcess

mdp = MarkovDecisionProcess(
    state_space_config="config/states.yaml",
    action_space_config="config/actions.yaml",
    discount_factor=0.95
)
```

#### Methods

##### `select_action(current_state: State) -> Action`

Select optimal action for current state using learned policy.

**Parameters:**
- `current_state` (State): Current system state

**Returns:**
- `Action`: Optimal action with confidence score

##### `update_policy(experience: Experience) -> PolicyUpdateResult`

Update policy based on experience.

**Parameters:**
- `experience` (Experience): State, action, reward, next state tuple

**Returns:**
- `PolicyUpdateResult`: Policy update metrics and convergence info

## Nicotine Module API

### ContextualSketchingEngine

Handles initial idea sketching and context maintenance.

```python
from spectacular.nicotine import ContextualSketchingEngine

sketching_engine = ContextualSketchingEngine(
    sketch_validation_model="models/sketch_validator.pkl",
    context_window_size=50
)
```

#### Methods

##### `create_initial_sketch(query: str, domain_hints: List[str]) -> SketchResult`

Generate initial visualization sketch from query.

**Parameters:**
- `query` (str): Natural language query
- `domain_hints` (List[str]): Domain-specific hints for sketch generation

**Returns:**
- `SketchResult`: Initial sketch with confidence metrics

##### `validate_context_coherence(sketch_history: List[Sketch]) -> CoherenceMetrics`

Validate context coherence across sketch evolution.

**Parameters:**
- `sketch_history` (List[Sketch]): History of sketch refinements

**Returns:**
- `CoherenceMetrics`: Coherence scores and potential issues

## Diadochi Module API

### DiadochiOrchestrator

Intelligent model combination system for domain expertise coordination.

```python
from spectacular.diadochi import DiadochiOrchestrator, DomainExpertise

orchestrator = DiadochiOrchestrator()
```

#### Methods

##### `register_domain_expert(domain_id: str, model: Any, expertise: DomainExpertise)`

Register a domain expert model.

**Parameters:**
- `domain_id` (str): Unique identifier for the domain
- `model` (Any): Model instance or API client
- `expertise` (DomainExpertise): Domain expertise configuration

##### `process_query(query: str, integration_pattern: IntegrationPattern) -> DiadochiResult`

Process query using intelligent model combination.

**Parameters:**
- `query` (str): Input query
- `integration_pattern` (IntegrationPattern): How to combine expert responses

**Returns:**
- `DiadochiResult`: Synthesized response from multiple experts

#### Integration Patterns

```python
from spectacular.diadochi import IntegrationPattern

# Available patterns
IntegrationPattern.ROUTER_ENSEMBLE      # Route to single best expert
IntegrationPattern.SEQUENTIAL_CHAIN     # Chain experts sequentially
IntegrationPattern.MIXTURE_OF_EXPERTS   # Weighted combination
IntegrationPattern.SYSTEM_PROMPTS       # Single model, multiple prompts
IntegrationPattern.KNOWLEDGE_DISTILL    # Distill knowledge from experts
```

## Hugging Face Integration API

### HuggingFaceOrchestrator

Provides access to pre-trained models for enhanced plot construction.

```python
from spectacular.hf_integration import HuggingFaceOrchestra

hf_orchestrator = HuggingFaceOrchestrator(
    cache_dir="models/hf_cache",
    api_key="your_hf_api_key"
)
```

#### Methods

##### `generate_code(query: str, context: Dict, model_preference: str = "auto") -> CodeGenerationResult`

Generate code using Hugging Face models.

**Parameters:**
- `query` (str): Natural language description
- `context` (Dict): Context information including data schema
- `model_preference` (str): Preferred model ("auto", "codet5", "codegen", etc.)

**Returns:**
- `CodeGenerationResult`: Generated code with model confidence scores

##### `analyze_sketch(sketch_image: np.ndarray) -> SketchAnalysisResult`

Analyze sketch using vision-language models.

**Parameters:**
- `sketch_image` (np.ndarray): Image array of the sketch

**Returns:**
- `SketchAnalysisResult`: Semantic understanding of the sketch

## Client Libraries

### JavaScript/TypeScript Client

```typescript
import { SpectacularClient } from '@spectacular/client';

const client = new SpectacularClient({
  apiUrl: 'https://api.spectacular.dev',
  apiKey: 'your-api-key'
});

// Generate visualization
const result = await client.generateVisualization({
  query: "Create a bar chart of sales by region",
  dataSchema: { region: "string", sales: "number" },
  interactivity: true
});
```

### Python Client

```python
from spectacular_client import SpectacularClient

client = SpectacularClient(
    api_url="https://api.spectacular.dev",
    api_key="your-api-key"
)

result = client.generate_visualization(
    query="Create a bar chart of sales by region",
    data_schema={"region": "string", "sales": "number"},
    interactivity=True
)
```

## Error Handling

### Exception Types

#### `SpectacularError`
Base exception class for all Spectacular errors.

#### `MetacognitiveError`
Errors in metacognitive reasoning process.

#### `ModuleError`
Errors specific to individual modules (Pretoria, Mzekezeke, etc.).

#### `APIError`
Errors in API communication or rate limiting.

### Error Response Format

```json
{
  "error": {
    "type": "MetacognitiveError",
    "message": "Failed to achieve reasoning convergence",
    "details": {
      "module": "orchestrator",
      "reasoning_depth": 5,
      "convergence_threshold": 0.95,
      "achieved_convergence": 0.87
    },
    "suggestions": [
      "Reduce reasoning depth",
      "Adjust convergence threshold", 
      "Provide more context"
    ]
  }
}
```

## Rate Limits and Quotas

| Endpoint | Rate Limit | Quota |
|----------|------------|-------|
| `/generate` | 10 req/min | 1000 req/day |
| `/analyze` | 20 req/min | 2000 req/day |
| `/sketch` | 30 req/min | 3000 req/day |

## Authentication

All API endpoints require authentication via API key:

```bash
curl -H "Authorization: Bearer your-api-key" \
     -H "Content-Type: application/json" \
     -d '{"query": "Create a scatter plot"}' \
     https://api.spectacular.dev/generate
```

## WebSocket API

For real-time interaction and streaming responses:

```javascript
const ws = new WebSocket('wss://api.spectacular.dev/stream');

ws.send(JSON.stringify({
  type: 'generate',
  query: 'Create an interactive timeline',
  stream_reasoning: true
}));

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'reasoning_step') {
    console.log('Reasoning:', data.step);
  } else if (data.type === 'final_result') {
    console.log('Generated code:', data.code);
  }
};
``` 
# Pretoria Fuzzy Rules Documentation

## Overview

The Pretoria module contains over 500 fuzzy rules that govern code generation strategies, prompt optimization, and reasoning depth selection. This document outlines the rule categories, syntax, and examples.

## Rule Categories

### 1. Query Analysis Rules (R001-R100)

Rules that analyze incoming queries for complexity, domain specificity, and user expertise.

#### Query Complexity Assessment
```fuzzy
R001: IF query_length IS long AND technical_terms IS many
      THEN query_complexity IS high (weight: 0.9)

R002: IF data_dimensions IS high AND chart_types IS multiple  
      THEN query_complexity IS very_high (weight: 0.95)

R003: IF query_contains_negation IS true OR conditional_statements IS present
      THEN query_complexity IS medium_high (weight: 0.8)
```

#### Domain Specificity Detection
```fuzzy
R010: IF domain_keywords IS scientific AND statistical_terms IS present
      THEN domain_specificity IS high (weight: 0.9)

R011: IF business_terms IS present AND kpi_mentions IS frequent
      THEN domain_specificity IS business_focused (weight: 0.85)

R012: IF geographic_terms IS present AND map_keywords IS mentioned
      THEN domain_specificity IS geospatial (weight: 0.9)
```

### 2. Code Generation Strategy Rules (R101-R200)

Rules determining the approach for D3.js code generation.

#### Generation Approach Selection
```fuzzy
R101: IF query_complexity IS high AND user_expertise IS low
      THEN generation_strategy IS template_based AND code_comments IS extensive (weight: 0.9)

R102: IF interactivity_required IS high AND performance_critical IS true
      THEN generation_strategy IS optimized AND event_handling IS efficient (weight: 0.8)

R103: IF data_size IS large AND real_time_updates IS required
      THEN generation_strategy IS streaming AND memory_efficient IS true (weight: 0.85)
```

#### Code Structure Rules
```fuzzy
R110: IF chart_complexity IS high
      THEN code_modularity IS high AND function_decomposition IS deep (weight: 0.8)

R111: IF maintainability_important IS true
      THEN code_documentation IS extensive AND naming_convention IS descriptive (weight: 0.7)
```

### 3. Prompt Engineering Rules (R201-R300)

Rules for optimizing prompts sent to language models.

#### Prompt Strategy Selection
```fuzzy
R201: IF query_ambiguity IS high
      THEN prompt_strategy IS clarification_first AND context_gathering IS extensive (weight: 0.9)

R202: IF domain_expertise_required IS high
      THEN prompt_strategy IS domain_expert AND technical_depth IS maximum (weight: 0.85)

R203: IF creative_visualization IS requested
      THEN prompt_strategy IS exploratory AND example_diversity IS high (weight: 0.8)
```

#### Context Enhancement Rules
```fuzzy
R210: IF data_schema IS complex
      THEN context_detail IS high AND schema_explanation IS included (weight: 0.9)

R211: IF user_constraints IS specific
      THEN constraint_emphasis IS high AND validation_strict IS true (weight: 0.8)
```

### 4. Visual Design Rules (R301-R400)

Rules governing visual design decisions and aesthetic choices.

#### Color Scheme Selection
```fuzzy
R301: IF data_type IS categorical AND categories IS many
      THEN color_scheme IS qualitative AND color_blind_safe IS true (weight: 0.9)

R302: IF data_represents IS temperature OR data_represents IS intensity
      THEN color_scheme IS sequential AND thermal_mapping IS true (weight: 0.85)

R303: IF comparison_emphasis IS required
      THEN color_scheme IS diverging AND contrast_high IS true (weight: 0.8)
```

#### Layout Optimization
```fuzzy
R310: IF data_density IS high
      THEN layout_spacing IS optimized AND overlap_prevention IS active (weight: 0.9)

R311: IF screen_size IS mobile
      THEN layout_responsive IS true AND touch_friendly IS enabled (weight: 0.95)
```

### 5. Interaction Design Rules (R401-R500)

Rules for interactive features and user experience design.

#### Interaction Type Selection
```fuzzy
R401: IF exploration_focused IS true
      THEN interactions IS zoom_pan AND filtering IS multi_dimensional (weight: 0.9)

R402: IF storytelling_mode IS active
      THEN interactions IS guided AND narrative_flow IS maintained (weight: 0.8)

R403: IF data_comparison IS primary_task
      THEN interactions IS brushing_linking AND highlighting IS coordinated (weight: 0.85)
```

#### Performance vs Feature Trade-offs
```fuzzy
R410: IF data_points IS very_large AND interactions IS complex
      THEN performance_optimization IS priority AND feature_reduction IS acceptable (weight: 0.9)

R411: IF response_time IS critical
      THEN lazy_loading IS enabled AND progressive_rendering IS true (weight: 0.85)
```

## Membership Functions

### Linguistic Variables

#### Query Complexity
- **Very Low**: [0, 0, 0.2, 0.3]
- **Low**: [0.2, 0.3, 0.4, 0.5]  
- **Medium**: [0.4, 0.5, 0.6, 0.7]
- **High**: [0.6, 0.7, 0.8, 0.9]
- **Very High**: [0.8, 0.9, 1.0, 1.0]

#### User Expertise
- **Novice**: [0, 0, 0.25, 0.4]
- **Intermediate**: [0.25, 0.4, 0.6, 0.75]
- **Expert**: [0.6, 0.75, 1.0, 1.0]

#### Data Size
- **Small**: [0, 100, 1000, 5000]
- **Medium**: [1000, 5000, 50000, 100000]
- **Large**: [50000, 100000, 1000000, 10000000]

## Rule Weights and Priorities

### Weight Categories
- **Critical (0.9-1.0)**: Safety, correctness, core functionality
- **Important (0.7-0.9)**: Performance, usability, aesthetics  
- **Moderate (0.5-0.7)**: Optimization, enhancement features
- **Low (0.3-0.5)**: Nice-to-have features, experimental

### Priority Resolution
When multiple rules fire simultaneously:
1. **Weighted Average**: Combine outputs using rule weights
2. **Maximum Activation**: Use rule with highest activation level
3. **Domain Hierarchy**: Domain-specific rules override general rules

## Dynamic Rule Learning

### Rule Discovery
```python
def discover_new_rule(successful_cases):
    patterns = extract_patterns(successful_cases)
    for pattern in patterns:
        if pattern.confidence > 0.8:
            candidate_rule = generate_rule(pattern)
            validate_rule(candidate_rule)
```

### Rule Validation Process
1. **Consistency Check**: Ensure no contradictions with existing rules
2. **Performance Test**: Validate on test cases
3. **Expert Review**: Human expert validation for critical rules
4. **A/B Testing**: Compare performance with/without new rule

### Rule Refinement
```python
def refine_rule_weights():
    for rule in rule_base:
        performance = evaluate_rule_performance(rule)
        if performance < threshold:
            adjust_weight(rule, performance_delta)
```

## Configuration

### Rule Base Loading
```yaml
# pretoria_config.yaml
rule_base:
  path: "rules/d3_generation.rules"
  format: "fuzzy_logic"
  auto_reload: true
  validation_strict: true

membership_functions:
  path: "config/membership_functions.json"
  custom_functions: "config/custom_mf.py"

inference_engine:
  type: "mamdani"
  defuzzification: "centroid"
  aggregation: "maximum"
```

### Performance Tuning
```yaml
optimization:
  rule_pruning: true
  parallel_inference: true
  cache_results: true
  max_rules_per_inference: 50
```

## Usage Examples

### Basic Rule Application
```python
from spectacular.pretoria import FuzzyEngine

engine = FuzzyEngine()
result = engine.infer({
    'query_complexity': 0.8,
    'user_expertise': 0.3,
    'data_size': 0.6
})

print(f"Generation strategy: {result['generation_strategy']}")
print(f"Code comments level: {result['code_comments']}")
```

### Custom Rule Addition
```python
engine.add_rule(
    antecedent="accessibility_required IS true AND color_usage IS primary",
    consequent="color_blind_testing IS mandatory AND alt_text IS comprehensive",
    weight=0.95
)
```

### Rule Performance Analysis
```python
stats = engine.get_rule_statistics()
for rule_id, perf in stats.items():
    if perf['activation_rate'] < 0.1:
        print(f"Consider removing rarely used rule: {rule_id}")
```

This fuzzy rule system enables Spectacular to make nuanced decisions about code generation strategies based on complex, often ambiguous input conditions. 
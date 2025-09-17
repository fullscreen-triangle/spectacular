"""
Markdown Report Generator

Creates comprehensive markdown reports showing every step of the
Bayesian Evidence Network reasoning process.
"""

from typing import Dict, Any, List
from datetime import datetime
import os

class MarkdownReportGenerator:
    """Generates detailed markdown reports of the reasoning process"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.output_config = config.get('output', {})
    
    def generate_report(self, results: Dict[str, Any]) -> str:
        """Generate complete markdown report"""
        
        report = []
        
        # Header
        report.extend(self._generate_header(results))
        
        # Table of Contents
        report.extend(self._generate_toc())
        
        # Executive Summary
        report.extend(self._generate_summary(results))
        
        # Query Analysis
        report.extend(self._generate_query_section(results))
        
        # Bayesian Network Processing
        report.extend(self._generate_network_section(results))
        
        # Step-by-Step Reasoning
        report.extend(self._generate_steps_section(results))
        
        # Visual Representations
        report.extend(self._generate_visuals_section(results))
        
        # Network State Analysis
        report.extend(self._generate_network_analysis(results))
        
        # Final Results
        report.extend(self._generate_results_section(results))
        
        # Technical Appendix
        report.extend(self._generate_appendix(results))
        
        return '\n'.join(report)
    
    def _generate_header(self, results: Dict[str, Any]) -> List[str]:
        """Generate report header"""
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        return [
            "# 🧠 Spectacular Bayesian Evidence Network",
            "## Complete Reasoning Trace Report",
            "",
            f"**Generated:** {timestamp}",
            f"**Query:** {results['query']}",
            f"**Network Coherence:** {results['network_coherence']:.3f}/1.0",
            f"**Processing Time:** {results['total_processing_time']:.2f} seconds",
            f"**Total Steps:** {len(results['processing_steps'])}",
            "",
            "---",
            ""
        ]
    
    def _generate_toc(self) -> List[str]:
        """Generate table of contents"""
        
        return [
            "## 📋 Table of Contents",
            "",
            "1. [Executive Summary](#executive-summary)",
            "2. [Query Analysis](#query-analysis)",
            "3. [Bayesian Network Processing](#bayesian-network-processing)",
            "4. [Step-by-Step Reasoning](#step-by-step-reasoning)",
            "5. [Visual Representations](#visual-representations)",
            "6. [Network State Analysis](#network-state-analysis)",
            "7. [Final Results](#final-results)",
            "8. [Technical Appendix](#technical-appendix)",
            "",
            "---",
            ""
        ]
    
    def _generate_summary(self, results: Dict[str, Any]) -> List[str]:
        """Generate executive summary"""
        
        nodes_state = results['nodes_final_state']
        converged_nodes = sum(1 for state in nodes_state.values() if state['state'] == 'converged')
        total_recursive = sum(state['recursive_count'] for state in nodes_state.values())
        
        return [
            "## 📊 Executive Summary",
            "",
            f"The Bayesian Evidence Network successfully processed the query **\"{results['query']}\"** ",
            f"through {len(results['processing_steps'])} reasoning steps over {results['total_processing_time']:.2f} seconds.",
            "",
            "### Key Metrics:",
            f"- **Network Coherence:** {results['network_coherence']:.3f}/1.0",
            f"- **Nodes Converged:** {converged_nodes}/{len(nodes_state)}",
            f"- **Recursive Loops:** {total_recursive} total across all nodes",
            f"- **Processing Efficiency:** {len(results['processing_steps'])/results['total_processing_time']:.1f} steps/second",
            "",
            "### Processing Quality:",
            "",
            self._get_quality_assessment(results['network_coherence']),
            "",
            "---",
            ""
        ]
    
    def _get_quality_assessment(self, coherence: float) -> str:
        """Get qualitative assessment based on coherence score"""
        
        if coherence >= 0.9:
            return "🟢 **Excellent** - Very high confidence in reasoning and results"
        elif coherence >= 0.8:
            return "🟡 **Good** - High confidence with minor uncertainties addressed"
        elif coherence >= 0.7:
            return "🟠 **Satisfactory** - Adequate confidence, some areas need more evidence"
        elif coherence >= 0.6:
            return "🟡 **Moderate** - Reasonable confidence but significant uncertainties remain"
        else:
            return "🔴 **Poor** - Low confidence, results should be used cautiously"
    
    def _generate_query_section(self, results: Dict[str, Any]) -> List[str]:
        """Generate query analysis section"""
        
        return [
            "## 🔍 Query Analysis",
            "",
            f"**Original Query:** `{results['query']}`",
            "",
            f"**Context Provided:**",
            *[f"- {ctx}" for ctx in results['context']],
            "",
            "### Query Complexity Assessment:",
            "",
            "The system analyzed this query for:",
            "- **Intent Recognition** - What the user really wants to know",
            "- **Knowledge Domain** - Which areas of knowledge are required", 
            "- **Complexity Level** - How sophisticated the explanation needs to be",
            "- **Visual Requirements** - What visualizations would help understanding",
            "",
            "---",
            ""
        ]
    
    def _generate_network_section(self, results: Dict[str, Any]) -> List[str]:
        """Generate Bayesian network overview"""
        
        nodes_state = results['nodes_final_state']
        
        section = [
            "## 🧠 Bayesian Network Processing",
            "",
            "The query was processed through a network of interconnected Bayesian nodes, ",
            "each specializing in different aspects of reasoning and validation.",
            "",
            "### Network Architecture:",
            ""
        ]
        
        # Add network diagram
        if self.output_config.get('include_ascii_art', True):
            section.extend([
                "```",
                "Query Input",
                "     │",
                "     ▼",
                "┌─────────────────┐",
                "│  Query Analysis │ ──┐",
                "└─────────────────┘   │",
                "     │                │",
                "     ▼                ▼",
                "┌─────────────────┐ ┌─────────────────┐",
                "│Knowledge Retriev│ │Reasoning Valid. │",
                "└─────────────────┘ └─────────────────┘",
                "     │                │",
                "     ▼                ▼",
                "┌─────────────────┐ ┌─────────────────┐",
                "│Visual Generation│ │Final Synthesis  │",
                "└─────────────────┘ └─────────────────┘",
                "     │                │",
                "     └────────┬───────┘",
                "              ▼",
                "         Final Result",
                "```",
                ""
            ])
        
        # Add node states
        section.extend([
            "### Node States:",
            ""
        ])
        
        for node_id, state in nodes_state.items():
            status_emoji = {
                'converged': '✅',
                'recursive_loop': '🔄',
                'needs_more_evidence': '❓',
                'failed': '❌'
            }.get(state['state'], '⚪')
            
            section.extend([
                f"**{node_id}** {status_emoji}",
                f"- Confidence: {state['confidence']:.3f}",
                f"- Uncertainty: {state['uncertainty']:.3f}",
                f"- Evidence Items: {state['evidence_count']}",
                f"- Recursive Loops: {state['recursive_count']}",
                f"- State: {state['state']}",
                ""
            ])
        
        section.extend([
            "---",
            ""
        ])
        
        return section
    
    def _generate_steps_section(self, results: Dict[str, Any]) -> List[str]:
        """Generate detailed step-by-step reasoning"""
        
        section = [
            "## 🔄 Step-by-Step Reasoning",
            "",
            "This section shows every single step the Bayesian network took to process your query. ",
            "Each step shows what happened, why it happened, and how it affected the system's confidence.",
            "",
        ]
        
        steps = results['processing_steps']
        
        for i, step in enumerate(steps, 1):
            confidence_change = step.confidence_after - step.confidence_before
            uncertainty_change = step.uncertainty_after - step.uncertainty_before
            
            # Determine step type icon
            step_icon = {
                'add_evidence': '📝',
                'check_convergence': '🎯', 
                'analyze_query': '🔍',
                'knowledge_retrieval': '📚',
                'reasoning_validation': '✅',
                'visual_generation': '🎨',
                'synthesis': '🎭'
            }.get(step.action, '⚙️')
            
            section.extend([
                f"### {step_icon} Step {i}: {step.action.replace('_', ' ').title()}",
                f"**Node:** {step.node_id} | **Duration:** {step.duration:.3f}s | **Time:** {step.timestamp.strftime('%H:%M:%S')}",
                "",
                f"**Action:** {step.action}",
                "",
                f"**Input:** {str(step.input_data)[:200]}{'...' if len(str(step.input_data)) > 200 else ''}",
                "",
                f"**Reasoning:** {step.reasoning}",
                "",
                "**Confidence Changes:**",
                f"- Before: {step.confidence_before:.3f}",
                f"- After: {step.confidence_after:.3f}",
                f"- Change: {confidence_change:+.3f} {'📈' if confidence_change > 0 else '📉' if confidence_change < 0 else '➡️'}",
                "",
                "**Uncertainty Changes:**",
                f"- Before: {step.uncertainty_before:.3f}",
                f"- After: {step.uncertainty_after:.3f}",
                f"- Change: {uncertainty_change:+.3f} {'📉' if uncertainty_change < 0 else '📈' if uncertainty_change > 0 else '➡️'}",
                "",
                f"**Output:** {str(step.output_data)[:300]}{'...' if len(str(step.output_data)) > 300 else ''}",
                "",
                "---",
                ""
            ])
        
        return section
    
    def _generate_visuals_section(self, results: Dict[str, Any]) -> List[str]:
        """Generate visual representations section"""
        
        section = [
            "## 🎨 Visual Representations",
            "",
            "Visual representations help validate understanding and make abstract concepts concrete.",
            "",
        ]
        
        # Look for visual generation steps
        visual_steps = [step for step in results['processing_steps'] 
                       if 'visual' in step.action.lower() or step.node_id == 'visual_generation']
        
        if visual_steps:
            section.extend([
                "### Generated Visualizations:",
                ""
            ])
            
            for step in visual_steps:
                if isinstance(step.output_data, str) and len(step.output_data) > 50:
                    section.extend([
                        "```",
                        str(step.output_data)[:1000] + ("..." if len(str(step.output_data)) > 1000 else ""),
                        "```",
                        ""
                    ])
        
        # Add confidence progression graph if enabled
        if self.output_config.get('include_confidence_graphs', True):
            section.extend(self._generate_confidence_graph(results))
        
        section.extend([
            "---",
            ""
        ])
        
        return section
    
    def _generate_confidence_graph(self, results: Dict[str, Any]) -> List[str]:
        """Generate ASCII confidence graph"""
        
        # Extract confidence values over time for each node
        node_confidences = {}
        
        for step in results['processing_steps']:
            node_id = step.node_id
            if node_id not in node_confidences:
                node_confidences[node_id] = []
            node_confidences[node_id].append(step.confidence_after)
        
        section = [
            "### Confidence Progression:",
            "",
            "Shows how confidence evolved for each node during processing:",
            ""
        ]
        
        # Create simple ASCII graph for each node
        for node_id, confidences in node_confidences.items():
            if len(confidences) > 1:
                section.append(f"**{node_id}:**")
                section.append("```")
                
                # Create simple bar chart
                for i, conf in enumerate(confidences):
                    bar_length = int(conf * 20)
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    section.append(f"Step {i+1:2d}: [{bar}] {conf:.3f}")
                
                section.extend([
                    "```",
                    ""
                ])
        
        return section
    
    def _generate_network_analysis(self, results: Dict[str, Any]) -> List[str]:
        """Generate network state analysis"""
        
        nodes_state = results['nodes_final_state']
        
        return [
            "## 🌐 Network State Analysis",
            "",
            "Analysis of the final state of the Bayesian Evidence Network:",
            "",
            f"### Overall Network Coherence: {results['network_coherence']:.3f}",
            "",
            self._get_coherence_explanation(results['network_coherence']),
            "",
            "### Node Performance Summary:",
            "",
            *self._generate_node_performance_table(nodes_state),
            "",
            "### Recursive Processing Analysis:",
            "",
            *self._generate_recursive_analysis(nodes_state),
            "",
            "---",
            ""
        ]
    
    def _get_coherence_explanation(self, coherence: float) -> str:
        """Explain what the coherence score means"""
        
        if coherence >= 0.9:
            return ("This very high coherence score indicates the network reached strong consensus "
                   "across all nodes with minimal uncertainty. The reasoning is highly reliable.")
        elif coherence >= 0.8:
            return ("This high coherence score shows good agreement between nodes with manageable "
                   "uncertainty levels. The reasoning is trustworthy.")
        elif coherence >= 0.7:
            return ("This moderate coherence score indicates reasonable agreement between nodes "
                   "but with some uncertainty remaining. Results should be considered solid but not definitive.")
        else:
            return ("This low coherence score suggests significant disagreement or uncertainty "
                   "between nodes. Results should be interpreted cautiously.")
    
    def _generate_node_performance_table(self, nodes_state: Dict[str, Any]) -> List[str]:
        """Generate performance table for nodes"""
        
        table = [
            "| Node | Confidence | Uncertainty | Evidence | Recursive | State |",
            "|------|------------|-------------|----------|-----------|-------|"
        ]
        
        for node_id, state in nodes_state.items():
            table.append(
                f"| {node_id} | {state['confidence']:.3f} | {state['uncertainty']:.3f} | "
                f"{state['evidence_count']} | {state['recursive_count']} | {state['state']} |"
            )
        
        return table
    
    def _generate_recursive_analysis(self, nodes_state: Dict[str, Any]) -> List[str]:
        """Analyze recursive processing patterns"""
        
        recursive_nodes = [(node_id, state) for node_id, state in nodes_state.items() 
                          if state['recursive_count'] > 0]
        
        if not recursive_nodes:
            return ["No recursive processing was needed - all nodes converged on first pass."]
        
        analysis = []
        total_recursive = sum(state['recursive_count'] for _, state in recursive_nodes)
        
        analysis.extend([
            f"Total recursive loops executed: {total_recursive}",
            "",
            "Recursive processing was triggered when nodes had high uncertainty or low confidence. ",
            "This allows the system to gather additional evidence and improve its reasoning.",
            "",
            "Nodes that used recursive processing:"
        ])
        
        for node_id, state in recursive_nodes:
            analysis.append(f"- **{node_id}**: {state['recursive_count']} loops")
        
        return analysis
    
    def _generate_results_section(self, results: Dict[str, Any]) -> List[str]:
        """Generate final results section"""
        
        return [
            "## 🎯 Final Results",
            "",
            "The Bayesian Evidence Network has completed processing and generated the following results:",
            "",
            "### Response:",
            "",
            str(results.get('final_response', 'No final response generated')),
            "",
            "### Validation Summary:",
            "",
            f"- **Network Coherence:** {results['network_coherence']:.3f}/1.0",
            f"- **Processing Time:** {results['total_processing_time']:.2f} seconds",
            f"- **Steps Completed:** {len(results['processing_steps'])}",
            f"- **Nodes Converged:** {sum(1 for state in results['nodes_final_state'].values() if state['state'] == 'converged')}/{len(results['nodes_final_state'])}",
            "",
            "### Confidence Assessment:",
            "",
            self._get_quality_assessment(results['network_coherence']),
            "",
            "---",
            ""
        ]
    
    def _generate_appendix(self, results: Dict[str, Any]) -> List[str]:
        """Generate technical appendix"""
        
        return [
            "## 📋 Technical Appendix",
            "",
            "### System Configuration:",
            "",
            "```yaml",
            f"Query: {results['query']}",
            f"Context: {results['context']}",
            f"Total Processing Time: {results['total_processing_time']:.3f}s",
            f"Network Coherence: {results['network_coherence']:.6f}",
            f"Total Steps: {len(results['processing_steps'])}",
            "```",
            "",
            "### Bayesian Network Theory:",
            "",
            "This system uses Bayesian inference to update beliefs based on evidence:",
            "",
            "- **Prior Belief**: Initial confidence before evidence",
            "- **Likelihood**: How well evidence supports the hypothesis", 
            "- **Posterior Belief**: Updated confidence after evidence",
            "- **Uncertainty**: Measure of how much we don't know",
            "",
            "The formula used: `P(H|E) = P(E|H) * P(H) / P(E)`",
            "",
            "Where:",
            "- `P(H|E)` = Probability of hypothesis given evidence",
            "- `P(E|H)` = Probability of evidence given hypothesis", 
            "- `P(H)` = Prior probability of hypothesis",
            "- `P(E)` = Probability of evidence",
            "",
            "### Fuzzy Logic Integration:",
            "",
            "Each node uses fuzzy logic to handle uncertainty:",
            "- **Confidence** (0-1): How sure we are about something",
            "- **Uncertainty** (0-1): How much we don't know",
            "- **Evidence Strength** = Confidence × (1 - Uncertainty)",
            "",
            "### Recursive Processing:",
            "",
            "When uncertainty is high or confidence is low, nodes can trigger recursive processing ",
            "to gather additional evidence and improve their beliefs.",
            "",
            "---",
            "",
            f"*Report generated by Spectacular Bayesian Evidence Network Demo*",
            f"*Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"
        ]
    
    def save_report(self, report_content: str, filename: Optional[str] = None) -> str:
        """Save report to file"""
        
        if not filename:
            filename = self.output_config.get('filename', 'spectacular_reasoning_trace.md')
        
        # Create output directory if it doesn't exist
        output_dir = 'output'
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return filepath

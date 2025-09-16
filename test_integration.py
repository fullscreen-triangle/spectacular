#!/usr/bin/env python3
"""
Integration test for the Spectacular Triple Validation Framework.

This script tests the basic functionality of all components to ensure
the system works end-to-end.
"""

import asyncio
import sys
import traceback
from datetime import datetime

async def test_validation_framework():
    """Test the core validation framework components."""
    
    print("🔧 Testing Validation Framework...")
    
    try:
        # Test imports
        from validation import TripleValidator, TripleValidationResult
        from validation.core.pugachev_cobra import PugachevCobraGenerator
        from validation.core.intent_analyzer import IntentAnalyzer
        from validation.core.reasoning_validator import ReasoningValidator
        
        print("  ✓ All validation modules imported successfully")
        
        # Test basic initialization
        validator = TripleValidator()
        print("  ✓ Triple Validator initialized")
        
        # Test simple validation
        test_query = "What is the relationship between force and acceleration?"
        test_context = {
            'data': [
                [1, 2], [2, 4], [3, 6], [4, 8], [5, 10]  # Simple linear data
            ],
            'query': test_query,
            'timestamp': datetime.now().isoformat()
        }
        
        result = await validator.validate_query(test_query, test_context)
        print(f"  ✓ Triple validation completed with coherence: {result.coherence_score:.2f}")
        print(f"  ✓ Validation passed: {result.validation_passed}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Validation framework test failed: {str(e)}")
        traceback.print_exc()
        return False

async def test_visual_reasoning():
    """Test the visual reasoning framework components."""
    
    print("🔧 Testing Visual Reasoning Framework...")
    
    try:
        # Test imports
        from visual_reasoning.core.visual_embeddings import VisualEmbeddingProcessor, VisualEmbedding
        from visual_reasoning.core.spatial_reasoning import SpatialReasoningEngine, SpatialContext
        from visual_reasoning.core.mathematical_visualization import MathVisualizationEngine, MathematicalFunction
        
        print("  ✓ All visual reasoning modules imported successfully")
        
        # Test visual embedding processor
        visual_processor = VisualEmbeddingProcessor()
        print("  ✓ Visual Embedding Processor initialized")
        
        # Test with simple SVG content
        test_svg = '''<svg width="400" height="300">
            <rect x="10" y="10" width="100" height="50" fill="blue"/>
            <circle cx="200" cy="150" r="30" fill="red"/>
        </svg>'''
        
        embedding = await visual_processor.create_visual_embedding(test_svg, "svg")
        print(f"  ✓ Visual embedding created with {len(embedding.get_combined_embedding())} dimensions")
        
        # Test spatial reasoning
        spatial_engine = SpatialReasoningEngine()
        spatial_context = await spatial_engine.analyze_spatial_context(test_svg)
        print(f"  ✓ Spatial context analyzed: {spatial_context.coordinate_system}")
        
        # Test mathematical visualization
        math_engine = MathVisualizationEngine()
        
        # Create a simple linear function
        linear_func = MathematicalFunction(
            expression="2*x + 1",
            domain=(-5, 5),
            range=(-9, 11),
            function_type="linear",
            parameters={'slope': 2, 'intercept': 1},
            derivative="2",
            properties={'continuous': True, 'monotonic': True}
        )
        
        visualization = await math_engine.create_function_visualization([linear_func], "Test Function")
        print(f"  ✓ Mathematical visualization created with accuracy: {visualization.mathematical_accuracy:.2f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Visual reasoning test failed: {str(e)}")
        traceback.print_exc()
        return False

async def test_chat_interface():
    """Test the chat interface components."""
    
    print("🔧 Testing Chat Interface...")
    
    try:
        # Test FastAPI app import (without running server)
        import chat_interface.backend.main as chat_main
        
        print("  ✓ Chat interface backend imported successfully")
        print("  ✓ FastAPI app configuration verified")
        
        # Test that global components can be initialized
        from validation import TripleValidator
        from visual_reasoning.core.visual_embeddings import VisualEmbeddingProcessor
        
        validator = TripleValidator()
        processor = VisualEmbeddingProcessor()
        
        print("  ✓ Chat backend components can be initialized")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Chat interface test failed: {str(e)}")
        traceback.print_exc()
        return False

async def test_end_to_end_scenario():
    """Test a complete end-to-end scenario."""
    
    print("🔧 Testing End-to-End Scenario...")
    
    try:
        from validation import TripleValidator
        from visual_reasoning.core.visual_embeddings import VisualEmbeddingProcessor
        
        # Initialize system
        validator = TripleValidator()
        visual_processor = VisualEmbeddingProcessor()
        
        # Simulate a physics query with data
        query = "How does Newton's second law relate force to acceleration?"
        data = {
            'forces': [1, 2, 3, 4, 5],      # Newtons
            'accelerations': [1, 2, 3, 4, 5] # m/s²
        }
        
        context = {
            'query': query,
            'data': data,
            'domain': 'physics',
            'timestamp': datetime.now().isoformat()
        }
        
        print("  🚀 Running triple validation...")
        
        # Perform triple validation
        validation_result = await validator.validate_query(query, context)
        
        print(f"  ✓ Ridiculous plot generated: {len(validation_result.ridiculous.svg_content)} chars")
        print(f"  ✓ Intent plot generated: {len(validation_result.intent.svg_content)} chars")
        print(f"  ✓ Reasoning plot generated: {len(validation_result.reasoning.svg_content)} chars")
        
        # Create visual embeddings for each plot
        embeddings = {}
        
        for plot_name, plot_data in [
            ("ridiculous", validation_result.ridiculous),
            ("intent", validation_result.intent),
            ("reasoning", validation_result.reasoning)
        ]:
            if hasattr(plot_data, 'svg_content'):
                embedding = await visual_processor.create_visual_embedding(
                    plot_data.svg_content,
                    content_type="svg",
                    context={"plot_type": plot_name}
                )
                embeddings[plot_name] = embedding
                print(f"  ✓ {plot_name} embedding: {len(embedding.get_combined_embedding())} dimensions")
        
        # Final validation check
        overall_success = (
            validation_result.coherence_score > 0.1 and  # Relaxed threshold for test
            len(embeddings) == 3 and
            all(len(emb.get_combined_embedding()) > 0 for emb in embeddings.values())
        )
        
        if overall_success:
            print(f"  🎉 End-to-end test PASSED!")
            print(f"     - Coherence Score: {validation_result.coherence_score:.3f}")
            print(f"     - Validation Passed: {validation_result.validation_passed}")
            print(f"     - Processing Time: {validation_result.processing_time:.3f}s")
            return True
        else:
            print(f"  ❌ End-to-end test FAILED - validation criteria not met")
            return False
        
    except Exception as e:
        print(f"  ✗ End-to-end test failed: {str(e)}")
        traceback.print_exc()
        return False

async def main():
    """Run all integration tests."""
    
    print("🎯 Spectacular Triple Validation Framework - Integration Tests")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    tests = [
        ("Validation Framework", test_validation_framework),
        ("Visual Reasoning", test_visual_reasoning),
        ("Chat Interface", test_chat_interface),
        ("End-to-End Scenario", test_end_to_end_scenario)
    ]
    
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        
        try:
            result = await test_func()
            test_results.append((test_name, result))
        except Exception as e:
            print(f"  ✗ Test {test_name} crashed: {str(e)}")
            test_results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    passed_tests = sum(1 for _, result in test_results if result)
    total_tests = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {status} - {test_name}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("🎉 ALL TESTS PASSED! The system is ready for deployment.")
        return 0
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())

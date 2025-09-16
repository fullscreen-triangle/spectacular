"""
Chat Interface Backend: FastAPI service for Triple Validation Framework.

This module provides the REST API endpoints for the AI chat service that
generates triple validation plots (Pugachev-Cobra, Intent Analysis, Reasoning Validation).
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Import our AI orchestration framework
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from spectacular.advanced_pipeline_orchestrator import AdvancedPipelineOrchestrator
from validation import TripleValidationResult
from visual_reasoning.core.visual_embeddings import VisualEmbedding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
advanced_pipeline_orchestrator: AdvancedPipelineOrchestrator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    global advanced_pipeline_orchestrator
    
    # Initialize Advanced 8-Stage Pipeline System
    logger.info("Initializing Advanced 8-Stage Pipeline System...")
    
    # Configuration for LLM models and sensor systems
    orchestrator_config = {
        'llm': {
            'openai_api_key': os.getenv('OPENAI_API_KEY', 'your-api-key-here'),
            'query_model': 'gpt-4-turbo-preview',
            'reasoning_model': 'gpt-4-turbo-preview',
            'viz_model': 'gpt-4-turbo-preview',
            'synthesis_model': 'gpt-4-turbo-preview'
        },
        'sensors': {
            'enable_audio': True,
            'enable_camera': True,
            'sensor_polling_rate': 1.0
        },
        'rag': {
            'knowledge_base_path': 'knowledge',
            'embedding_model': 'text-embedding-ada-002'
        }
    }
    
    advanced_pipeline_orchestrator = AdvancedPipelineOrchestrator(orchestrator_config)
    logger.info("Advanced 8-Stage Pipeline System initialized successfully")
    
    yield
    
    # Cleanup
    logger.info("Shutting down Triple Validation Framework...")

# Create FastAPI app
app = FastAPI(
    title="Spectacular Triple Validation Framework",
    description="AI Chat Service with Triple Plot Validation",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # React frontend ports
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., description="User message/query")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional context")
    data: Optional[Any] = Field(None, description="Data for analysis if provided")
    conversation_id: Optional[str] = Field(None, description="Conversation identifier")

class PlotData(BaseModel):
    """Individual plot data model."""
    svg_content: str
    title: str
    description: str
    confidence: float
    metadata: Dict[str, Any]

class ChatResponse(BaseModel):
    """Chat response model."""
    response_text: str
    plots: Dict[str, PlotData]  # ridiculous, intent, reasoning
    validation_passed: bool
    coherence_score: float
    processing_time: float
    conversation_id: str
    timestamp: str
    validation_details: Dict[str, Any]

class EmbeddingRequest(BaseModel):
    """Visual embedding request model."""
    visual_content: str
    content_type: str = "svg"
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)

class EmbeddingResponse(BaseModel):
    """Visual embedding response model."""
    embedding_id: str
    dimensionality: int
    confidence_scores: Dict[str, float]
    semantic_annotations: List[str]
    processing_time: float

class SystemStatus(BaseModel):
    """System status model."""
    status: str
    components: Dict[str, str]
    validation_metrics: Dict[str, Any]
    uptime: float

# Main Chat Endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def process_chat_message(
    request: ChatRequest,
    background_tasks: BackgroundTasks
) -> ChatResponse:
    """
    Process chat message with AI-orchestrated sophisticated reasoning pipeline.
    
    This uses the AI reasoning orchestrator to:
    1. Analyze query with LLM intelligence
    2. Coordinate validation components intelligently 
    3. Generate triple validation plots
    4. Synthesize results with AI reasoning
    """
    
    start_time = datetime.now()
    conversation_id = request.conversation_id or f"conv_{start_time.strftime('%Y%m%d_%H%M%S')}"
    
    logger.info("🚀 Starting AI-orchestrated processing for conversation: %s", conversation_id)
    
    try:
        # Prepare context for AI orchestration
        orchestration_context = {
            'query': request.message,
            'data': request.data,
            'timestamp': start_time.isoformat(),
            'conversation_id': conversation_id,
            'user_context': request.context
        }
        
        # Run Advanced 8-Stage Pipeline
        orchestration_result = await advanced_pipeline_orchestrator.execute_full_pipeline(
            request.message, orchestration_context
        )
        
        # Extract validation results from 8-stage pipeline
        validation_results = orchestration_result.get('validation_results', {})
        
        # Create visual embeddings for enhanced plots (background task)
        background_tasks.add_task(
            create_visual_embeddings_async,
            orchestration_result.get('visual_embeddings', {}),
            conversation_id
        )
        
        # Use AI-synthesized response from Stage 8
        response_text = orchestration_result.get('synthesized_response', 'Advanced pipeline processing completed')
        
        # Format plots with Advanced Pipeline insights
        ridiculous_data = validation_results.get('ridiculous', {})
        intent_data = validation_results.get('intent', {})
        reasoning_data = validation_results.get('reasoning', {})
        
        plots = {
            "ridiculous": PlotData(
                svg_content=ridiculous_data.get('svg_content', '<svg><text>No plot generated</text></svg>'),
                title="8-Stage Pipeline: Pugachev-Cobra Boundary Test",
                description=ridiculous_data.get('interpretation', 'Boundary validation analysis'),
                confidence=ridiculous_data.get('confidence', 0.5),
                metadata={
                    "ai_reasoning": "Environmental boundary validation through 8-stage pipeline",
                    "boundary_established": ridiculous_data.get('boundary_established', False),
                    "environmental_integration": True,
                    "pipeline_stage": "Stage 6: Validation Convergence"
                }
            ),
            "intent": PlotData(
                svg_content=intent_data.get('svg_content', '<svg><text>No plot generated</text></svg>'),
                title="8-Stage Pipeline: Environmental Intent Analysis",
                description=intent_data.get('inferred_intent', 'Cognitive mapping analysis'),
                confidence=intent_data.get('intent_confidence', 0.5),
                metadata={
                    "ai_reasoning": "12-dimensional environmental cognitive mapping",
                    "alternative_intents": intent_data.get('alternatives', []),
                    "environmental_factors": orchestration_result.get('environmental_snapshot', {}),
                    "pipeline_stage": "Stage 2: Cognitive Mapping"
                }
            ),
            "reasoning": PlotData(
                svg_content=reasoning_data.get('svg_content', '<svg><text>No plot generated</text></svg>'),
                title="8-Stage Pipeline: Environmental Reasoning Validation",
                description=reasoning_data.get('explanation', 'Environmental reasoning analysis'),
                confidence=orchestration_result.get('overall_coherence', 0.5),
                metadata={
                    "ai_reasoning": "Environmental information construction through sensor data",
                    "understanding_validated": reasoning_data.get('understanding_validated', False),
                    "patterns_identified": reasoning_data.get('patterns', []),
                    "environmental_coherence": orchestration_result.get('environmental_snapshot', {}).get('overall_coherence', 0.5),
                    "pipeline_stage": "Stage 7: Visual Coherence"
                }
            )
        }
        
        processing_time = orchestration_result.get('total_processing_time', 1.0)
        
        response = ChatResponse(
            response_text=response_text,
            plots=plots,
            validation_passed=validation_results.get('validation_passed', False),
            coherence_score=orchestration_result.get('overall_coherence', 0.5),
            processing_time=processing_time,
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat(),
            validation_details={
                'advanced_pipeline': True,
                'pipeline_stages_completed': orchestration_result.get('pipeline_metadata', {}).get('stages_completed', 0),
                'environmental_integration': orchestration_result.get('pipeline_metadata', {}).get('environmental_integration', False),
                'sensor_data_collected': orchestration_result.get('environmental_snapshot') is not None,
                'knowledge_items_synthesized': orchestration_result.get('pipeline_metadata', {}).get('knowledge_items_synthesized', 0),
                'visual_embeddings_created': orchestration_result.get('pipeline_metadata', {}).get('visual_embeddings_created', 0),
                'stage_timings': orchestration_result.get('stage_timings', {}),
                'confidence_progression': orchestration_result.get('confidence_progression', [])
            }
        )
        
        logger.info("✅ AI-orchestrated processing completed successfully in %.2fs", processing_time)
        return response
        
    except Exception as e:
        logger.error("❌ Error in AI-orchestrated processing: %s", str(e))
        raise HTTPException(status_code=500, detail=f"AI Processing error: {str(e)}")

# Visual Embedding Endpoint
@app.post("/api/embeddings", response_model=EmbeddingResponse)
async def create_visual_embedding(request: EmbeddingRequest) -> EmbeddingResponse:
    """Create AI-enhanced multi-dimensional visual embedding from visual content."""
    
    start_time = datetime.now()
    
    logger.info("Creating AI-enhanced visual embedding for %s content", request.content_type)
    
    try:
        # Use the advanced orchestrator's visual processor for consistency  
        visual_processor = advanced_pipeline_orchestrator.visual_processor
        
        embedding: VisualEmbedding = await visual_processor.create_visual_embedding(
            request.visual_content,
            request.content_type,
            request.context
        )
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = EmbeddingResponse(
            embedding_id=embedding.embedding_id,
            dimensionality=len(embedding.get_combined_embedding()),
            confidence_scores=embedding.confidence_scores,
            semantic_annotations=embedding.semantic_annotations,
            processing_time=processing_time
        )
        
        logger.info("✅ AI-enhanced visual embedding created in %.2fs", processing_time)
        return response
        
    except Exception as e:
        logger.error("❌ Error creating visual embedding: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Embedding creation error: {str(e)}")

# System Status Endpoint
@app.get("/api/status", response_model=SystemStatus)
async def get_system_status() -> SystemStatus:
    """Get current system status and AI orchestration metrics."""
    
    try:
        # Get Advanced Pipeline metrics
        pipeline_stats = advanced_pipeline_orchestrator.get_pipeline_statistics()
        orchestrator_metrics = {
            'pipeline_executions': pipeline_stats.get('total_executions', 0),
            'average_coherence': pipeline_stats.get('average_coherence', 0.5),
            'environmental_sensors': pipeline_stats.get('system_capabilities', {}),
            'llm_coordination': 'active',
            '8_stage_pipeline': 'operational',
            'sensor_integration': 'active'
        }
        
        # Calculate uptime (simplified)
        uptime = 3600.0  # Would be actual uptime in production
        
        status = SystemStatus(
            status="operational",
            components={
                "advanced_pipeline_orchestrator": "active",
                "environmental_sensor_system": "active",
                "8_stage_pipeline": "operational",
                "llm_coordinator": "active", 
                "rag_knowledge_retriever": "active",
                "triple_validator": "integrated",
                "visual_processor": "integrated",
                "math_visualizer": "integrated",
                "12d_sensor_array": "monitoring",
                "api_service": "running"
            },
            validation_metrics=orchestrator_metrics,
            uptime=uptime
        )
        
        return status
        
    except Exception as e:
        logger.error("Error getting system status: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Status error: {str(e)}")

# Health Check Endpoint
@app.get("/api/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

# Conversation History Endpoint (placeholder)
@app.get("/api/conversations/{conversation_id}")
async def get_conversation_history(conversation_id: str):
    """Get conversation history (placeholder for future implementation)."""
    return {
        "conversation_id": conversation_id,
        "messages": [],
        "note": "Conversation history not yet implemented"
    }

# Helper Functions
async def create_visual_embeddings_async(
    visual_embeddings: Dict[str, Any],
    conversation_id: str
):
    """Create visual embeddings for 8-stage pipeline plots (background task)."""
    
    try:
        # Extract the visual processor from the advanced orchestrator
        visual_processor = advanced_pipeline_orchestrator.visual_processor
        
        # The visual embeddings are already created by the pipeline
        # This function now serves as a storage/logging point
        
        embedding_count = len(visual_embeddings)
        total_dimensions = sum(
            emb.get('embedding_dimensions', 0) 
            for emb in visual_embeddings.values() 
            if isinstance(emb, dict)
        )
        
        # Store embeddings for future use (would be database in production)
        logger.info("✅ Processed %d visual embeddings (%d total dimensions) for conversation: %s", 
                   embedding_count, total_dimensions, conversation_id)
        
        # Additional processing could be done here, like:
        # - Storing to vector database
        # - Similarity comparisons
        # - Pattern analysis across conversations
        
    except Exception as e:
        logger.error("❌ Error processing visual embeddings: %s", str(e))

# Development Server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

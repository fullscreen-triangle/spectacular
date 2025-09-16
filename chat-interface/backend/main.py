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

from spectacular.reasoning_orchestrator import ReasoningOrchestrator
from validation import TripleValidationResult
from visual_reasoning.core.visual_embeddings import VisualEmbedding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
reasoning_orchestrator: ReasoningOrchestrator = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    global reasoning_orchestrator
    
    # Initialize AI-orchestrated reasoning system
    logger.info("Initializing AI-Orchestrated Reasoning System...")
    
    # Configuration for LLM models (would come from environment variables)
    orchestrator_config = {
        'openai_api_key': os.getenv('OPENAI_API_KEY', 'your-api-key-here'),
        'query_model': 'gpt-4-turbo-preview',
        'reasoning_model': 'gpt-4-turbo-preview',
        'viz_model': 'gpt-4-turbo-preview',
        'synthesis_model': 'gpt-4-turbo-preview'
    }
    
    reasoning_orchestrator = ReasoningOrchestrator(orchestrator_config)
    logger.info("AI Reasoning Orchestrator initialized successfully")
    
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
        
        # Run AI-orchestrated reasoning pipeline
        orchestration_result = await reasoning_orchestrator.process_query(
            request.message, orchestration_context
        )
        
        # Extract validation results from orchestration
        validation_results = orchestration_result['validation_results']
        
        # Create visual embeddings for enhanced plots (background task)
        background_tasks.add_task(
            create_visual_embeddings_async,
            orchestration_result['enhanced_visualizations'],
            conversation_id
        )
        
        # Use AI-synthesized response
        response_text = orchestration_result['synthesized_response']
        
        # Format plots with AI insights
        plots = {
            "ridiculous": PlotData(
                svg_content=validation_results['ridiculous']['svg_content'],
                title="AI-Generated Pugachev-Cobra Boundary Test",
                description=validation_results['ridiculous']['interpretation'],
                confidence=validation_results['ridiculous']['confidence'],
                metadata={
                    "ai_reasoning": "Boundary validation to test solution space limits",
                    "boundary_established": validation_results['ridiculous']['boundary_established'],
                    "ai_analysis": orchestration_result['ai_analysis']
                }
            ),
            "intent": PlotData(
                svg_content=validation_results['intent']['svg_content'],
                title="AI Intent Analysis & Recognition",
                description=validation_results['intent']['inferred_intent'],
                confidence=validation_results['intent']['intent_confidence'],
                metadata={
                    "ai_reasoning": "12-dimensional environmental intent analysis",
                    "alternative_intents": validation_results['intent']['alternatives'],
                    "llm_analysis": orchestration_result['ai_analysis']
                }
            ),
            "reasoning": PlotData(
                svg_content=validation_results['reasoning']['svg_content'],
                title="AI Reasoning Validation & Understanding",
                description=validation_results['reasoning']['explanation'],
                confidence=orchestration_result['reasoning_confidence'],
                metadata={
                    "ai_reasoning": "Environmental information construction validation",
                    "understanding_validated": validation_results['reasoning']['understanding_validated'],
                    "patterns_identified": validation_results['reasoning']['patterns'],
                    "ai_pipeline": orchestration_result['reasoning_pipeline']
                }
            )
        }
        
        processing_time = orchestration_result['processing_metadata']['processing_time']
        
        response = ChatResponse(
            response_text=response_text,
            plots=plots,
            validation_passed=validation_results['validation_passed'],
            coherence_score=validation_results['overall_coherence'],
            processing_time=processing_time,
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat(),
            validation_details={
                'ai_orchestration': True,
                'llm_models_used': orchestration_result['processing_metadata']['ai_models_used'],
                'pipeline_steps': orchestration_result['processing_metadata']['pipeline_steps'],
                'knowledge_items_retrieved': orchestration_result['processing_metadata']['knowledge_items_retrieved'],
                'reasoning_confidence': orchestration_result['reasoning_confidence']
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
        # Use the orchestrator's visual processor for consistency
        visual_processor = reasoning_orchestrator.visual_processor
        
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
        # Get AI orchestration metrics
        orchestrator_metrics = {
            'llm_models_available': len(reasoning_orchestrator.llm_coordinator.models),
            'rag_knowledge_sources': 'active',
            'validation_components': 'integrated',
            'visual_reasoning': 'active'
        }
        
        # Calculate uptime (simplified)
        uptime = 3600.0  # Would be actual uptime in production
        
        status = SystemStatus(
            status="operational",
            components={
                "ai_orchestrator": "active",
                "llm_coordinator": "active", 
                "rag_retriever": "active",
                "triple_validator": "integrated",
                "visual_processor": "integrated",
                "math_visualizer": "integrated",
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
    enhanced_visualizations: Dict[str, Any],
    conversation_id: str
):
    """Create visual embeddings for AI-enhanced validation plots (background task)."""
    
    try:
        # Extract the visual processor from the orchestrator
        visual_processor = reasoning_orchestrator.visual_processor
        
        # Create embeddings for each plot
        embeddings = {}
        
        triple_plots = enhanced_visualizations.get('triple_validation_plots', {})
        
        for plot_name, svg_content in triple_plots.items():
            if svg_content:
                embedding = await visual_processor.create_visual_embedding(
                    svg_content,
                    content_type="svg",
                    context={
                        "plot_type": plot_name,
                        "conversation_id": conversation_id,
                        "ai_enhanced": True
                    }
                )
                embeddings[plot_name] = embedding
        
        # Store embeddings for future use (would be database in production)
        logger.info("✅ Created AI-enhanced visual embeddings for conversation: %s", conversation_id)
        
    except Exception as e:
        logger.error("❌ Error creating visual embeddings: %s", str(e))

# Development Server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

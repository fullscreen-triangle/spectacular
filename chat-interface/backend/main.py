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

# Import our validation framework
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from validation import TripleValidator, TripleValidationResult
from visual_reasoning.core.visual_embeddings import VisualEmbeddingProcessor, VisualEmbedding

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
triple_validator: TripleValidator = None
visual_processor: VisualEmbeddingProcessor = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    global triple_validator, visual_processor
    
    # Initialize core components
    logger.info("Initializing Triple Validation Framework...")
    triple_validator = TripleValidator()
    visual_processor = VisualEmbeddingProcessor()
    logger.info("Framework initialized successfully")
    
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
    Process chat message with triple validation plots.
    
    This is the main endpoint that generates three validation plots:
    1. Ridiculous Solution Plot (Pugachev-Cobra)
    2. Intent Recognition Plot 
    3. Reasoning Validation Plot
    """
    
    start_time = datetime.now()
    conversation_id = request.conversation_id or f"conv_{start_time.strftime('%Y%m%d_%H%M%S')}"
    
    logger.info("Processing chat request for conversation: %s", conversation_id)
    
    try:
        # Prepare context for validation
        validation_context = {
            'query': request.message,
            'data': request.data,
            'timestamp': start_time.isoformat(),
            'conversation_id': conversation_id,
            **request.context
        }
        
        # Perform triple validation
        validation_result: TripleValidationResult = await triple_validator.validate_query(
            request.message, validation_context
        )
        
        # Create visual embeddings for each plot (background task for performance)
        background_tasks.add_task(
            create_visual_embeddings_async,
            validation_result,
            conversation_id
        )
        
        # Generate response text based on validation results
        response_text = await generate_response_text(request.message, validation_result)
        
        # Format plots for response
        plots = {
            "ridiculous": PlotData(
                svg_content=validation_result.ridiculous.svg_content,
                title="Pugachev-Cobra Boundary Test",
                description=validation_result.ridiculous.ridiculous_interpretation,
                confidence=validation_result.ridiculous.boundary_confidence,
                metadata={
                    "boundary_type": validation_result.ridiculous.boundary_type,
                    "inversion_strategy": validation_result.ridiculous.inversion_strategy,
                    "boundary_established": validation_result.ridiculous.boundary_established
                }
            ),
            "intent": PlotData(
                svg_content=validation_result.intent.svg_content,
                title="Intent Analysis",
                description=validation_result.intent.inferred_intent,
                confidence=validation_result.intent.intent_confidence,
                metadata={
                    "alternative_intents": validation_result.intent.alternative_intents,
                    "dimensional_analysis": validation_result.intent.dimensional_analysis,
                    "reasoning_chain": validation_result.intent.intent_reasoning_chain
                }
            ),
            "reasoning": PlotData(
                svg_content=validation_result.reasoning.svg_content,
                title="Reasoning Validation",
                description=validation_result.reasoning.reasoning_explanation,
                confidence=validation_result.reasoning.coherence_score,
                metadata={
                    "mathematical_relationships": validation_result.reasoning.mathematical_relationships,
                    "patterns_identified": validation_result.reasoning.data_patterns_identified,
                    "understanding_validated": validation_result.reasoning.understanding_validated,
                    "visualization_type": validation_result.reasoning.visualization_type
                }
            )
        }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        response = ChatResponse(
            response_text=response_text,
            plots=plots,
            validation_passed=validation_result.validation_passed,
            coherence_score=validation_result.coherence_score,
            processing_time=processing_time,
            conversation_id=conversation_id,
            timestamp=datetime.now().isoformat(),
            validation_details=validation_result.validation_details
        )
        
        logger.info("Chat request processed successfully in %.2fs", processing_time)
        return response
        
    except Exception as e:
        logger.error("Error processing chat request: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")

# Visual Embedding Endpoint
@app.post("/api/embeddings", response_model=EmbeddingResponse)
async def create_visual_embedding(request: EmbeddingRequest) -> EmbeddingResponse:
    """Create multi-dimensional visual embedding from visual content."""
    
    start_time = datetime.now()
    
    logger.info("Creating visual embedding for %s content", request.content_type)
    
    try:
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
        
        logger.info("Visual embedding created in %.2fs", processing_time)
        return response
        
    except Exception as e:
        logger.error("Error creating visual embedding: %s", str(e))
        raise HTTPException(status_code=500, detail=f"Embedding creation error: {str(e)}")

# System Status Endpoint
@app.get("/api/status", response_model=SystemStatus)
async def get_system_status() -> SystemStatus:
    """Get current system status and metrics."""
    
    try:
        # Get validation metrics from triple validator
        validation_metrics = triple_validator.get_validation_metrics()
        
        # Calculate uptime (simplified)
        uptime = 3600.0  # Would be actual uptime in production
        
        status = SystemStatus(
            status="operational",
            components={
                "triple_validator": "active",
                "visual_processor": "active",
                "api_service": "running"
            },
            validation_metrics=validation_metrics,
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
async def generate_response_text(query: str, validation_result: TripleValidationResult) -> str:
    """Generate response text based on validation results."""
    
    if validation_result.validation_passed:
        response_parts = [
            f"I've analyzed your query: '{query[:100]}...' using triple validation.",
            "",
            "✓ **Reasoning Validation**: " + (
                "Understanding confirmed through data visualization patterns" 
                if validation_result.reasoning.understanding_validated 
                else "Pattern recognition needs refinement"
            ),
            "",
            f"✓ **Intent Analysis**: {validation_result.intent.inferred_intent[:150]}...",
            "",
            f"✓ **Boundary Testing**: Pugachev-Cobra validation established solution space boundaries",
            "",
            f"**Coherence Score**: {validation_result.coherence_score:.2f}/1.0",
            "",
            "The visualizations below show my reasoning process and validate my understanding of your query."
        ]
    else:
        response_parts = [
            f"I've analyzed your query: '{query[:100]}...' but validation shows areas needing improvement.",
            "",
            "⚠ **Validation Concerns**:",
            f"- Coherence Score: {validation_result.coherence_score:.2f}/1.0 (threshold: 0.7)",
            f"- Intent Confidence: {validation_result.intent.intent_confidence:.2f}/1.0",
            f"- Understanding Validated: {'Yes' if validation_result.reasoning.understanding_validated else 'No'}",
            "",
            "The visualizations below show where my reasoning may be incomplete or incorrect."
        ]
    
    return "\n".join(response_parts)

async def create_visual_embeddings_async(
    validation_result: TripleValidationResult,
    conversation_id: str
):
    """Create visual embeddings for validation plots (background task)."""
    
    try:
        # Create embeddings for each plot
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
                    context={
                        "plot_type": plot_name,
                        "conversation_id": conversation_id
                    }
                )
                embeddings[plot_name] = embedding
        
        # Store embeddings for future use (would be database in production)
        logger.info("Created visual embeddings for conversation: %s", conversation_id)
        
    except Exception as e:
        logger.error("Error creating visual embeddings: %s", str(e))

# Development Server
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

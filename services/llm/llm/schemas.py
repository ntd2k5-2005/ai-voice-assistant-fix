"""
Pydantic schemas for request/response validation
Ensures type safety and automatic validation
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import List, Optional, Dict, Any
from datetime import datetime


class Message(BaseModel):
    """Single chat message"""
    role: str = Field(..., description="Message role: 'user', 'assistant', or 'system'")
    content: str = Field(..., min_length=1, description="Message content")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "role": "user",
                "content": "Hello, how are you?"
            }
        }
    )


class GenerateRequest(BaseModel):
    """Request for text generation"""
    prompt: str = Field(..., min_length=1, description="Input prompt")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0, description="Nucleus sampling")
    max_tokens: Optional[int] = Field(None, ge=1, le=4096, description="Maximum tokens to generate")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "prompt": "What is artificial intelligence?",
                "temperature": 0.7,
                "max_tokens": 200
            }
        }
    )


class GenerateResponse(BaseModel):
    """Response from text generation"""
    response: str = Field(..., description="Generated text")
    model: str = Field(..., description="Model used")
    metrics: Optional[Dict[str, Any]] = Field(None, description="Performance metrics if enabled")
    timestamp: str = Field(..., description="Response timestamp")


class ChatRequest(BaseModel):
    """Request for chat conversation"""
    messages: List[Message] = Field(..., min_length=1, description="Conversation messages")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, ge=1, le=4096)
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "messages": [
                    {"role": "user", "content": "What is machine learning?"}
                ],
                "temperature": 0.7,
                "max_tokens": 300
            }
        }
    )


class ChatResponse(BaseModel):
    """Response from chat conversation"""
    message: Message = Field(..., description="Assistant's response")
    model: str = Field(..., description="Model used")
    timestamp: str = Field(..., description="Response timestamp")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status: 'healthy' or 'unhealthy'")
    model: str = Field(..., description="Configured model name")
    ollama_available: bool = Field(..., description="Whether Ollama is reachable")
    model_loaded: bool = Field(..., description="Whether model is loaded")
    timestamp: str = Field(..., description="Check timestamp")


class MetricsData(BaseModel):
    """Performance metrics data"""
    ttft_ms: Optional[float] = Field(None, description="Time to first token (milliseconds)")
    avg_tbt_ms: Optional[float] = Field(None, description="Average time between tokens (milliseconds)")
    total_latency_ms: float = Field(..., description="Total generation latency (milliseconds)")
    tokens_per_second: Optional[float] = Field(None, description="Generation throughput")
    output_tokens: Optional[int] = Field(None, description="Number of output tokens")

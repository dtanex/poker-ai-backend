"""
FastAPI Backend for Poker AI
Production-ready REST API wrapper for the CFR poker AI
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import os

# Import the AI player - PROFESSIONAL VERSION
try:
    from aiplayer_pro import ProfessionalAIPlayer
    print("🎯 Loading PROFESSIONAL AI Player (Opus V2)")
    USE_PRO = True
except ImportError:
    from aiplayer import CFRAIPlayer
    print("⚠️  Falling back to basic AI player")
    USE_PRO = False

# Initialize FastAPI app
app = FastAPI(
    title="Poker AI API - Professional Edition",
    description="GTO poker strategy API powered by professional CFR solver (comparable to PioSolver)",
    version="2.0.0"
)

# CORS middleware (allow your Next.js frontend to call this API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",  # Local development
        "https://yourdomain.com",  # Your production domain
        "https://exploit-actual.vercel.app",  # Your frontend
        "*"  # Remove this in production, only for testing
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize AI player (loads strategies on startup)
try:
    if USE_PRO:
        ai_player = ProfessionalAIPlayer("gto_strategies.pkl")
        print("✅ Professional AI player loaded successfully")
        print("   📊 Features: Position-aware, Stack-depth adjusted, Realistic calling frequencies")
    else:
        ai_player = CFRAIPlayer("cfr_strategies.pkl")
        print("✅ Basic AI player loaded successfully")
except Exception as e:
    print(f"⚠️  Warning: Could not load AI player: {e}")
    ai_player = None


# Pydantic models for request/response validation
class PokerHand(BaseModel):
    """Request model for poker decision"""
    stage: str = Field(..., description="Game stage: preflop, flop, turn, or river")
    hole_cards: List[str] = Field(..., description="Your two hole cards, e.g. ['As', 'Kh']", min_items=2, max_items=2)
    board: List[str] = Field(default=[], description="Community cards, e.g. ['Qh', 'Jd', '7c']", max_items=5)
    
    class Config:
        schema_extra = {
            "example": {
                "stage": "flop",
                "hole_cards": ["As", "Kd"],
                "board": ["Ah", "Qh", "7c"]
            }
        }


class StrategyResponse(BaseModel):
    """Response model for strategy"""
    strategy: Dict[str, float] = Field(..., description="Probability distribution over actions")
    recommended_action: str = Field(..., description="Single action sampled from strategy")
    bucket: int = Field(..., description="Hand strength bucket (for debugging)")
    explanation: Optional[str] = Field(None, description="Human-readable explanation (requires Claude API)")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    ai_loaded: bool
    version: str


# API Routes
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint - API info"""
    return {
        "message": "Poker AI API - Professional Edition",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if ai_player else "degraded",
        "ai_loaded": ai_player is not None,
        "version": "2.0.0"
    }


@app.post("/strategy", response_model=StrategyResponse)
async def get_strategy(hand: PokerHand):
    """
    Get poker strategy for a given situation
    
    Returns both the full strategy (probability distribution)
    and a recommended action (sampled from the distribution)
    """
    if not ai_player:
        raise HTTPException(status_code=503, detail="AI player not loaded")
    
    try:
        # Validate stage
        valid_stages = ["preflop", "flop", "turn", "river"]
        if hand.stage.lower() not in valid_stages:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid stage. Must be one of: {valid_stages}"
            )
        
        # Validate board cards for stage
        stage = hand.stage.lower()
        if stage == "preflop" and len(hand.board) != 0:
            raise HTTPException(status_code=400, detail="Preflop should have no board cards")
        if stage == "flop" and len(hand.board) != 3:
            raise HTTPException(status_code=400, detail="Flop should have exactly 3 board cards")
        if stage == "turn" and len(hand.board) != 4:
            raise HTTPException(status_code=400, detail="Turn should have exactly 4 board cards")
        if stage == "river" and len(hand.board) != 5:
            raise HTTPException(status_code=400, detail="River should have exactly 5 board cards")
        
        # Get strategy from AI
        strategy = ai_player.get_strategy(stage, hand.hole_cards, hand.board)
        action = ai_player.get_action(stage, hand.hole_cards, hand.board)

        # Get bucket (preflop or postflop)
        if stage == "preflop":
            bucket = ai_player.get_preflop_bucket(hand.hole_cards)
        else:
            equity = ai_player.calculate_equity(hand.hole_cards, hand.board)
            bucket = ai_player.compute_postflop_bucket(equity)

        return {
            "strategy": strategy,
            "recommended_action": action,
            "bucket": bucket,
            "explanation": None  # TODO: Add Claude API integration for natural language explanation
        }
    
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/action")
async def get_action(hand: PokerHand):
    """
    Get a single recommended action (shorthand for /strategy)
    """
    if not ai_player:
        raise HTTPException(status_code=503, detail="AI player not loaded")
    
    try:
        action = ai_player.get_action(hand.stage, hand.hole_cards, hand.board)
        return {"action": action}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")


@app.post("/coaching")
async def get_coaching(hand: PokerHand):
    """
    Get GTO strategy with natural language coaching explanation
    
    TODO: Integrate Claude API for explanations
    """
    if not ai_player:
        raise HTTPException(status_code=503, detail="AI player not loaded")
    
    # Get GTO strategy
    strategy = ai_player.get_strategy(hand.stage, hand.hole_cards, hand.board)
    bucket, _ = ai_player.get_bucket_info(hand.stage, hand.hole_cards, hand.board)
    
    # TODO: Call Claude API to generate explanation
    # Example prompt:
    # """
    # I have {hole_cards} on the {stage}.
    # Board: {board}
    # 
    # The GTO solver recommends:
    # {strategy}
    # 
    # Explain why this is the optimal strategy in simple terms.
    # """
    
    return {
        "strategy": strategy,
        "bucket": bucket,
        "coaching": "Natural language coaching requires Claude API integration. See TODO in code.",
        "next_steps": "Integrate anthropic.Client to generate coaching text"
    }


# Development server
if __name__ == "__main__":
    import uvicorn
    
    # Check if strategies file exists
    if not os.path.exists("cfr_strategies.pkl"):
        print("\n" + "⚠️ "*20)
        print("WARNING: cfr_strategies.pkl not found!")
        print("Run 'python cfr_trainer.py' first to generate strategies.")
        print("⚠️ "*20 + "\n")
    
    # Run server
    print("\n🚀 Starting Poker AI API server...")
    print("📝 API docs: http://localhost:8000/docs")
    print("🏥 Health check: http://localhost:8000/health")
    print("\n")
    
    uvicorn.run(
        "fastapi_backend:app",
        host="0.0.0.0",
        port=8000,
        reload=True  # Auto-reload on code changes (disable in production)
    )

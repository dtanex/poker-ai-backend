"""
Master GTO Poker AI - FastAPI Backend v3.0
Integrated with Opus's enhanced CFR system with proper calling ranges
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging
import os
from pathlib import Path

# Import Opus's master GTO system
from master_gto_system import MasterGTOSystem

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Professional GTO Poker AI",
    version="3.0.0",
    description="World-class GTO poker coaching with proper calling ranges"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Master GTO System
logger.info("Initializing Master GTO System...")
system = MasterGTOSystem()

# Check for pre-trained strategies
STRATEGIES_PATH = Path(__file__).parent / "master_gto_strategies.pkl"

if STRATEGIES_PATH.exists():
    logger.info("Loading pre-trained strategies...")
    system.load_strategies(str(STRATEGIES_PATH))
    logger.info("✅ Pre-trained strategies loaded")
else:
    logger.info("No pre-trained strategies found. Training new strategies...")
    logger.info("This will take a few minutes on first run...")

    # Train with moderate iterations (increase for production)
    system.train_from_scratch(iterations=50000)

    # Save for future use
    system.save_strategies(str(STRATEGIES_PATH))
    logger.info("✅ Training complete and saved")


# Pydantic models for API
class StrategyRequest(BaseModel):
    """Request model matching your frontend"""
    stage: str  # "preflop", "flop", "turn", "river"
    hole_cards: List[str]  # ["As", "Kh"]
    board: List[str] = []  # ["Ks", "9s", "2h"]
    position: Optional[str] = "BTN"
    stack_depth: Optional[int] = 100
    action_facing: Optional[str] = "UNOPENED"
    pot_size: Optional[float] = None
    bet_size: Optional[float] = None
    villain_id: Optional[str] = None


class StrategyResponse(BaseModel):
    """Response model matching your frontend expectations"""
    recommended_action: str
    strategy: Dict[str, float]  # {"fold": 0.1, "call": 0.6, "raise": 0.3}
    bucket: str
    hand_analysis: Dict
    explanation: str
    version: str = "3.0.0"


@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "message": "Professional GTO Poker AI v3.0",
        "status": "ready",
        "features": [
            "Realistic calling frequencies",
            "Set mining with implied odds",
            "Suited connector defense",
            "Position-aware strategies",
            "Opponent exploitation",
            "Postflop equity calculations"
        ]
    }


@app.get("/version")
def get_version():
    """Get API version info"""
    return {
        "version": "3.0.0",
        "ai_type": "Enhanced CFR with Proper Calling Ranges",
        "features": {
            "calling_ranges": True,
            "set_mining": True,
            "opponent_modeling": True,
            "postflop_solver": True,
            "equity_calculator": True
        }
    }


@app.post("/strategy", response_model=StrategyResponse)
def get_strategy(request: StrategyRequest):
    """
    Get GTO strategy for given situation

    This endpoint is compatible with your existing frontend
    but now uses Opus's enhanced system with proper calling ranges!
    """
    try:
        # Map frontend request to master system format
        action_facing_map = {
            "UNOPENED": "UNOPENED",
            "FACING_RAISE": "FACING_RAISE",
            "facing_raise": "FACING_RAISE",
            "FACING_3BET": "FACING_3BET",
            "facing_3bet": "FACING_3BET",
            "FACING_4BET": "FACING_4BET"
        }

        # Default pot size based on action
        if request.pot_size is None:
            if "RAISE" in request.action_facing.upper():
                request.pot_size = 6.5  # 3BB raise + blinds
            elif "3BET" in request.action_facing.upper():
                request.pot_size = 15.0
            else:
                request.pot_size = 1.5  # Blinds only

        # Default bet size
        if request.bet_size is None:
            if "RAISE" in request.action_facing.upper():
                request.bet_size = 3.0
            else:
                request.bet_size = 0.0

        # Build game state for master system
        game_state = {
            'hole_cards': request.hole_cards,
            'board': request.board,
            'position': request.position or 'BTN',
            'action_facing': action_facing_map.get(request.action_facing, 'UNOPENED'),
            'stack_depth': request.stack_depth or 100,
            'pot_size': request.pot_size,
            'bet_size': request.bet_size,
            'stage': request.stage,
            'villain_id': request.villain_id
        }

        # Get decision from master system
        decision = system.make_decision(game_state)

        # Map probabilities to strategy format (frontend expects this)
        strategy = {
            'fold': decision['probabilities']['fold'],
            'call': decision['probabilities']['call'],
            'raise': decision['probabilities']['raise']
        }

        # Determine recommended action (highest probability)
        recommended_action = decision['action']

        # Create bucket description
        hand_analysis = decision['hand_analysis']
        bucket_parts = []

        if hand_analysis['strength'] >= 95:
            bucket_parts.append("Premium")
        elif hand_analysis['strength'] >= 80:
            bucket_parts.append("Strong")
        elif hand_analysis['strength'] >= 60:
            bucket_parts.append("Playable")
        else:
            bucket_parts.append("Marginal")

        if hand_analysis['set_mining']:
            bucket_parts.append("Set Mining")
        if hand_analysis['suited_connector']:
            bucket_parts.append("Suited Connector")

        bucket = " | ".join(bucket_parts) if bucket_parts else "Standard"

        # Build explanation
        explanation_parts = [decision['explanation']]

        # Add key insights
        if recommended_action == 'call' and strategy['call'] > 0.5:
            explanation_parts.append("Strong calling spot - proper pot odds")
        elif recommended_action == 'raise' and strategy['raise'] > 0.8:
            explanation_parts.append("Clear raising opportunity")

        explanation = " • ".join(explanation_parts)

        return StrategyResponse(
            recommended_action=recommended_action,
            strategy=strategy,
            bucket=bucket,
            hand_analysis=hand_analysis,
            explanation=explanation,
            version="3.0.0 - Enhanced CFR"
        )

    except Exception as e:
        logger.error(f"Error in get_strategy: {e}")
        raise HTTPException(status_code=500, detail=f"Strategy calculation failed: {str(e)}")


@app.post("/update_opponent")
def update_opponent(player_id: str, action: Dict):
    """
    Update opponent statistics for exploitation

    Example:
    {
        "player_id": "villain123",
        "action": {
            "position": "BTN",
            "action_type": "raise",
            "stage": "preflop",
            "facing_action": "unopened"
        }
    }
    """
    try:
        system.opponent_modeler.update_opponent(player_id, action)

        # Get updated stats
        if player_id in system.opponent_modeler.opponents:
            stats = system.opponent_modeler.opponents[player_id]
            player_type = stats.get_player_type()

            return {
                "success": True,
                "player_id": player_id,
                "hands_played": stats.hands_played,
                "player_type": player_type.name,
                "vpip": stats.vpip,
                "pfr": stats.pfr
            }

        return {"success": True, "player_id": player_id}

    except Exception as e:
        logger.error(f"Error updating opponent: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/opponent/{player_id}")
def get_opponent_stats(player_id: str):
    """Get opponent statistics and classification"""
    try:
        if player_id not in system.opponent_modeler.opponents:
            raise HTTPException(status_code=404, detail="Opponent not found")

        stats = system.opponent_modeler.opponents[player_id]
        player_type = stats.get_player_type()

        return {
            "player_id": player_id,
            "player_type": player_type.name,
            "hands_played": stats.hands_played,
            "vpip": stats.vpip,
            "pfr": stats.pfr,
            "three_bet": stats.three_bet,
            "aggression_factor": stats.aggression_factor,
            "fold_to_3bet": stats.fold_to_3bet
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting opponent stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    """Detailed health check"""
    strategies_loaded = hasattr(system.cfr_trainer, 'strategy_sum') and len(system.cfr_trainer.strategy_sum) > 0

    return {
        "status": "healthy",
        "version": "3.0.0",
        "strategies_loaded": strategies_loaded,
        "num_trained_states": len(system.cfr_trainer.strategy_sum) if strategies_loaded else 0,
        "opponents_tracked": len(system.opponent_modeler.opponents)
    }


# For local development
if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))

    logger.info(f"Starting Professional GTO Poker AI on port {port}")
    logger.info("Features: Proper calling ranges, set mining, opponent exploitation")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

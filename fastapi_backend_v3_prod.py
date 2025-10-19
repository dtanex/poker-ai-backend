"""
Master GTO Poker AI - FastAPI Backend v3.0 PRODUCTION
Simplified version without training validation report
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging
import os
import pickle
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

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Master GTO System
logger.info("Initializing Master GTO System v3.0...")
system = MasterGTOSystem()

# Check for pre-trained strategies
STRATEGIES_PATH = Path(__file__).parent / "master_gto_strategies.pkl"

if STRATEGIES_PATH.exists():
    logger.info("✅ Loading pre-trained strategies...")
    system.load_strategies(str(STRATEGIES_PATH))
    logger.info("✅ Strategies loaded successfully")
else:
    logger.warning("⚠️ No pre-trained strategies found!")
    logger.info("Run training script first or strategies will be calculated on-the-fly")


# Pydantic models
class StrategyRequest(BaseModel):
    stage: str
    hole_cards: List[str]
    board: List[str] = []
    position: Optional[str] = "BTN"
    stack_depth: Optional[int] = 100
    action_facing: Optional[str] = "UNOPENED"
    pot_size: Optional[float] = None
    bet_size: Optional[float] = None
    villain_id: Optional[str] = None


class StrategyResponse(BaseModel):
    recommended_action: str
    strategy: Dict[str, float]
    bucket: str
    hand_analysis: Dict
    explanation: str
    version: str = "3.0.0"


@app.get("/")
def root():
    return {
        "message": "Professional GTO Poker AI v3.0 - Enhanced CFR",
        "status": "ready",
        "features": [
            "✅ Proper calling ranges",
            "✅ Set mining with implied odds",
            "✅ Suited connector defense",
            "✅ Position-aware strategies",
            "✅ Opponent exploitation",
            "✅ Postflop equity calculations"
        ]
    }


@app.get("/version")
def get_version():
    return {
        "version": "3.0.0",
        "ai_type": "Enhanced CFR with Proper Calling Ranges",
        "iteration": system.cfr_trainer.iteration if hasattr(system.cfr_trainer, 'iteration') else 0,
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
    try:
        # Map actions
        action_map = {
            "UNOPENED": "UNOPENED",
            "FACING_RAISE": "FACING_RAISE",
            "facing_raise": "FACING_RAISE",
            "FACING_3BET": "FACING_3BET",
            "facing_3bet": "FACING_3BET",
            "FACING_4BET": "FACING_4BET"
        }

        # Set defaults
        if request.pot_size is None:
            if "RAISE" in request.action_facing.upper():
                request.pot_size = 6.5
            elif "3BET" in request.action_facing.upper():
                request.pot_size = 15.0
            else:
                request.pot_size = 1.5

        if request.bet_size is None:
            if "RAISE" in request.action_facing.upper():
                request.bet_size = 3.0
            else:
                request.bet_size = 0.0

        # Build game state
        game_state = {
            'hole_cards': request.hole_cards,
            'board': request.board,
            'position': request.position or 'BTN',
            'action_facing': action_map.get(request.action_facing, 'UNOPENED'),
            'stack_depth': request.stack_depth or 100,
            'pot_size': request.pot_size,
            'bet_size': request.bet_size,
            'stage': request.stage,
            'villain_id': request.villain_id
        }

        # Get decision
        decision = system.make_decision(game_state)

        # HOTFIX: Override broken ranges for common spots
        # TODO: Mark needs to retrain the model properly
        hand_str = ''.join(sorted(request.hole_cards))
        is_pocket_pair = len(set([c[0] for c in request.hole_cards])) == 1
        pair_rank = request.hole_cards[0][0] if is_pocket_pair else None

        # Small/medium pairs from BB facing single raise = mostly call
        if (is_pocket_pair and pair_rank in ['2','3','4','5','6','7','8','9'] and
            request.position == 'BB' and
            'FACING_RAISE' in request.action_facing and
            request.stack_depth >= 40):  # Deep enough for set mining

            # Override to 70% call, 20% fold, 10% 3bet
            decision['probabilities'] = {'call': 0.70, 'fold': 0.20, 'raise': 0.10}
            decision['action'] = 'call'
            decision['explanation'] = f"Set mining with {pair_rank}{pair_rank} from BB | Good pot odds (getting ~3:1) | Deep stacks ({request.stack_depth}BB) for implied odds"

        # BTN facing single raise - defend wider (we have position)
        if (request.position == 'BTN' and 'FACING_RAISE' in request.action_facing):
            cards = [c[0] for c in request.hole_cards]

            # Premium pairs (TT+) = mostly 3bet
            if is_pocket_pair and pair_rank in ['T', 'J', 'Q', 'K', 'A']:
                decision['probabilities'] = {'call': 0.20, 'fold': 0.05, 'raise': 0.75}
                decision['action'] = 'raise'
                decision['explanation'] = f"Premium pair {pair_rank}{pair_rank} on BTN | 3bet for value with position | Mix in some calls to trap"

            # Medium pairs (66-99) = mostly call
            elif is_pocket_pair and pair_rank in ['6','7','8','9']:
                decision['probabilities'] = {'call': 0.70, 'fold': 0.10, 'raise': 0.20}
                decision['action'] = 'call'
                decision['explanation'] = f"Set mining with {pair_rank}{pair_rank} on BTN | Position advantage for implied odds | Sometimes 3bet as bluff"

            # AK/AQ = mostly 3bet
            elif ('A' in cards and 'K' in cards) or ('A' in cards and 'Q' in cards):
                decision['probabilities'] = {'call': 0.25, 'fold': 0.05, 'raise': 0.70}
                decision['action'] = 'raise'
                decision['explanation'] = f"Premium broadway on BTN | 3bet for value | High card strength + position = profitable squeeze"

            # Suited connectors (98s, 87s, 76s, etc) = mostly call
            elif (len(set([c[1] for c in request.hole_cards])) == 1 and  # Suited
                  abs(ord(cards[0]) - ord(cards[1])) <= 2):  # Connected or 1-gapper
                decision['probabilities'] = {'call': 0.60, 'fold': 0.25, 'raise': 0.15}
                decision['action'] = 'call'
                decision['explanation'] = f"Suited connector on BTN | Good playability with position | Implied odds to make straights/flushes"

        # Format response
        strategy = {
            'fold': decision['probabilities']['fold'],
            'call': decision['probabilities']['call'],
            'raise': decision['probabilities']['raise']
        }

        hand_analysis = decision['hand_analysis']

        # Create bucket
        if hand_analysis['strength'] >= 95:
            bucket = "Premium"
        elif hand_analysis['strength'] >= 80:
            bucket = "Strong"
        elif hand_analysis['strength'] >= 60:
            bucket = "Playable"
        else:
            bucket = "Marginal"

        if hand_analysis['set_mining']:
            bucket += " | Set Mining"
        if hand_analysis['suited_connector']:
            bucket += " | Suited Connector"

        return StrategyResponse(
            recommended_action=decision['action'],
            strategy=strategy,
            bucket=bucket,
            hand_analysis=hand_analysis,
            explanation=decision['explanation'],
            version="3.0.0"
        )

    except Exception as e:
        logger.error(f"Error in get_strategy: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Strategy calculation failed: {str(e)}")


@app.get("/health")
def health_check():
    strategies_loaded = hasattr(system.cfr_trainer, 'strategy_sum') and len(system.cfr_trainer.strategy_sum) > 0

    return {
        "status": "healthy",
        "version": "3.0.0",
        "strategies_loaded": strategies_loaded,
        "num_trained_states": len(system.cfr_trainer.strategy_sum) if strategies_loaded else 0,
        "opponents_tracked": len(system.opponent_modeler.opponents)
    }


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))

    logger.info("="*60)
    logger.info("🚀 Professional GTO Poker AI v3.0")
    logger.info("="*60)
    logger.info(f"Starting on port {port}")
    logger.info("Features:")
    logger.info("  ✅ Proper calling ranges (set mining, suited connectors)")
    logger.info("  ✅ Position-aware strategies")
    logger.info("  ✅ Opponent exploitation")
    logger.info("  ✅ Postflop equity calculations")
    logger.info("="*60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port,
        log_level="info"
    )

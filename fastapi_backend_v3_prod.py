"""
Master GTO Poker AI - FastAPI Backend v3.0 PRODUCTION
Simplified version without training validation report
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
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


class RangeRequest(BaseModel):
    game_type: str  # "6max", "8max", "10max"
    format: str  # "cash" or "mtt"
    stack_depth: int  # BB depth
    hero_position: str  # Position of hero
    villain_position: Optional[str] = None  # Position of villain (None = unopened pot)
    action_type: str = "UNOPENED"  # "UNOPENED", "RAISE", "3BET", "4BET"


class RangeResponse(BaseModel):
    ranges: Dict[str, Dict[str, float]]  # hand -> {fold, call, raise} frequencies
    summary: Dict[str, Any]


@app.post("/ranges", response_model=RangeResponse)
def get_ranges(request: RangeRequest):
    """
    Generate complete range chart for given game conditions.
    Returns action frequencies for all 169 starting hands.
    """
    try:
        # All 169 starting hands
        ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        ranges = {}

        # Adjust action facing based on villain position
        if request.villain_position:
            # Determine action type based on positions
            villain_pos_order = {'EP': 0, 'MP': 1, 'CO': 2, 'BTN': 3, 'SB': 4, 'BB': 5}
            hero_pos_order = {'EP': 0, 'MP': 1, 'CO': 2, 'BTN': 3, 'SB': 4, 'BB': 5}

            if request.villain_position in villain_pos_order and request.hero_position in hero_pos_order:
                if villain_pos_order[request.villain_position] < hero_pos_order[request.hero_position]:
                    action_facing = "FACING_RAISE"
                else:
                    action_facing = request.action_type
            else:
                action_facing = "FACING_RAISE"
        else:
            action_facing = "UNOPENED"

        # Generate frequency for each hand
        for i, rank1 in enumerate(ranks):
            for j, rank2 in enumerate(ranks):
                if i == j:
                    # Pocket pair
                    hand = f"{rank1}{rank2}"
                    hole_cards = [f"{rank1}s", f"{rank2}h"]
                elif i < j:
                    # Suited
                    hand = f"{rank1}{rank2}s"
                    hole_cards = [f"{rank1}s", f"{rank2}s"]
                else:
                    # Offsuit
                    hand = f"{rank2}{rank1}o"
                    hole_cards = [f"{rank2}s", f"{rank1}h"]

                # Adjust stack depth based on game format and player count
                effective_stack = request.stack_depth

                # MTT adjustments (tighter at lower stacks)
                if request.format == "mtt" and request.stack_depth < 30:
                    # Push/fold mode in tournaments
                    effective_stack = request.stack_depth

                # Game type adjustments (10max plays tighter than 6max)
                position_adjustment = 1.0
                if request.game_type == "10max":
                    position_adjustment = 0.85  # Tighter ranges
                elif request.game_type == "8max":
                    position_adjustment = 0.92

                # Use simplified GTO ranges (trained system may not be accurate for all scenarios)
                # Calculate hand strength for GTO baseline
                rank_values = {'A': 12, 'K': 11, 'Q': 10, 'J': 9, 'T': 8, '9': 7, '8': 6, '7': 5, '6': 4, '5': 3, '4': 2, '3': 1, '2': 0}
                is_pair = (i == j)
                is_suited = (i < j)
                high_rank = max(rank_values[rank1], rank_values[rank2])
                low_rank = min(rank_values[rank1], rank_values[rank2])

                # Position-based tightness (higher = tighter)
                pos_tightness = {
                    'UTG': 0.90, 'UTG1': 0.87, 'UTG2': 0.84, 'EP': 0.87, 'EP2': 0.82,
                    'MP': 0.75, 'MP2': 0.70, 'CO': 0.60, 'BTN': 0.45, 'SB': 0.70, 'BB': 0.80
                }.get(request.hero_position, 0.75)

                # Apply game type adjustment
                pos_tightness *= position_adjustment

                # Hand strength score (0-100) - much more conservative
                gap = high_rank - low_rank

                if is_pair:
                    # Pairs: AA=100, KK=98, down to 22=76
                    strength = 76 + high_rank * 2
                elif is_suited:
                    # Suited hands - weight high cards but not too harshly
                    strength = 35 + high_rank * 3.5 + low_rank * 2.0

                    # Connectivity bonus
                    if gap == 0:  # Connector
                        strength += 5
                    elif gap == 1:  # One-gapper
                        strength += 3
                    elif gap == 2:  # Two-gapper
                        strength += 1

                    # Light penalty for very low cards only
                    if high_rank < 6:  # Below 8
                        strength *= 0.90
                    if low_rank < 3:  # Below 5
                        strength *= 0.92

                else:  # Offsuit
                    # Offsuit - need high cards
                    strength = 20 + high_rank * 3.2 + low_rank * 1.5

                    # Small connectivity bonus
                    if gap == 0:
                        strength += 4

                    # Penalty for low cards
                    if high_rank < 8:  # Below T
                        strength *= 0.85
                    if low_rank < 6:  # Below 8
                        strength *= 0.88

                # Stack depth adjustments
                if effective_stack < 20:  # Short stack
                    if is_pair or (high_rank >= 11 and low_rank >= 9):  # Pairs and broadway
                        strength *= 1.1
                    else:
                        strength *= 0.9
                elif effective_stack > 200:  # Deep stack
                    if is_suited or is_pair:
                        strength *= 1.05

                # Calculate thresholds (very strict for realistic ranges)
                # EP (0.87): raise=92, call=78  ->  ~10-12% VPIP
                # BTN (0.45): raise=84, call=69  ->  ~45% VPIP
                raise_threshold = 75 + pos_tightness * 20
                call_threshold = 60 + pos_tightness * 20  # Increased from 15 to 20

                # Generate frequencies
                if strength >= raise_threshold:
                    # Strong hands: mostly raise
                    raise_freq = 0.70 + min(0.25, (strength - raise_threshold) / 30)
                    call_freq = 0.05
                    fold_freq = 1.0 - raise_freq - call_freq
                elif strength >= call_threshold:
                    # Medium hands: mixed call/raise
                    diff = (strength - call_threshold) / (raise_threshold - call_threshold)
                    raise_freq = diff * 0.30
                    call_freq = 0.40 + diff * 0.25
                    fold_freq = 1.0 - call_freq - raise_freq
                elif strength >= (call_threshold - 10):
                    # Marginal hands: mostly call or fold
                    call_freq = 0.25 + (strength - (call_threshold - 10)) / 40
                    raise_freq = 0.05
                    fold_freq = 1.0 - call_freq - raise_freq
                else:
                    # Weak hands: mostly fold
                    fold_freq = 0.85 + min(0.13, (call_threshold - 10 - strength) / 40)
                    call_freq = max(0.01, (1.0 - fold_freq) * 0.7)
                    raise_freq = 1.0 - fold_freq - call_freq

                # Normalize
                total = fold_freq + call_freq + raise_freq
                ranges[hand] = {
                    'fold': fold_freq / total,
                    'call': call_freq / total,
                    'raise': raise_freq / total
                }

        # Calculate summary statistics (sum of frequencies, not count of hands)
        vpip = sum(r['call'] + r['raise'] for r in ranges.values()) / 169 * 100
        pfr = sum(r['raise'] for r in ranges.values()) / 169 * 100

        summary = {
            'vpip': round(vpip, 1),
            'pfr': round(pfr, 1),
            'game_type': request.game_type,
            'format': request.format,
            'stack_depth': request.stack_depth,
            'hero_position': request.hero_position,
            'villain_position': request.villain_position,
            'action_type': action_facing
        }

        return RangeResponse(
            ranges=ranges,
            summary=summary
        )

    except Exception as e:
        logger.error(f"Error in get_ranges: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Range calculation failed: {str(e)}")


@app.get("/health")
def health_check():
    strategies_loaded = hasattr(system.cfr_trainer, 'strategy_sum') and len(system.cfr_trainer.strategy_sum) > 0

    return {
        "status": "healthy",
        "version": "3.0.1",
        "strategies_loaded": strategies_loaded,
        "num_trained_states": len(system.cfr_trainer.strategy_sum) if strategies_loaded else 0,
        "opponents_tracked": len(system.opponent_modeler.opponents),
        "deployment": "render-auto-deploy-test"
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

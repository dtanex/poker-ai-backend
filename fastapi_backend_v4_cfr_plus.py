"""
FastAPI Backend v4 - CFR+ with Claude AI Integration
World-class GTO solver with personalized coaching
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging
import os
import pickle
import json
from pathlib import Path
import numpy as np

# Import CFR+ trainer and Claude AI integration
from claude_ai_integration import initialize_claude_coach, get_coach

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="Professional GTO Poker AI v4.0",
    version="4.0.0",
    description="CFR+ solver with Claude AI coaching"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class CFRPlusSystem:
    """
    CFR+ GTO System with strategy loading
    """
    def __init__(self):
        self.strategies = {}
        self.loaded = False

        # Load hand rankings
        rankings_file = Path(__file__).parent / "hand_rankings.json"
        with open(rankings_file, 'r') as f:
            self.hand_rankings = json.load(f)

        logger.info("CFR+ System initialized")

    def load_strategies(self, filename: str):
        """Load trained CFR+ strategies"""
        try:
            with open(filename, 'rb') as f:
                data = pickle.load(f)

            self.strategies = data.get('strategies', {})
            self.loaded = True

            logger.info(f"✅ Loaded CFR+ strategies from {filename}")
            logger.info(f"   Total info sets: {len(self.strategies)}")
            return True

        except Exception as e:
            logger.error(f"Could not load strategies: {e}")
            return False

    def get_strategy(
        self,
        hand: str,
        position: str,
        action_facing: str,
        stack_depth: int = 100
    ) -> Dict:
        """
        Get GTO strategy for a given situation
        """
        # Build info set key (matches CFR+ trainer format)
        info_set = f"{hand}|{position}|{action_facing}|{stack_depth}"

        # Get strategy from trained model
        if info_set in self.strategies:
            strategy = self.strategies[info_set]

            # Convert list to dict
            if isinstance(strategy, list):
                strategy = {
                    'fold': strategy[0],
                    'call': strategy[1],
                    'raise': strategy[2]
                }
        else:
            # Fallback strategy based on hand strength
            strategy = self._fallback_strategy(hand, position, action_facing, stack_depth)

        # Determine recommended action
        actions = ['fold', 'call', 'raise']
        probs = [strategy['fold'], strategy['call'], strategy['raise']]
        recommended_action = actions[np.argmax(probs)]

        return {
            'recommended_action': recommended_action,
            'strategy': strategy,
            'info_set': info_set,
            'from_trained_model': info_set in self.strategies
        }

    def _fallback_strategy(self, hand: str, position: str, action_facing: str, stack_depth: int) -> Dict:
        """Fallback strategy if no trained strategy available"""
        # Get hand strength
        rank = self.hand_rankings.get(hand, 169)
        hand_strength = 1.0 - (rank / 169.0)

        # Basic strategy based on hand strength
        if hand_strength > 0.9:  # Premium (top 10%)
            return {'fold': 0.02, 'call': 0.08, 'raise': 0.90}
        elif hand_strength > 0.7:  # Strong (top 30%)
            return {'fold': 0.15, 'call': 0.35, 'raise': 0.50}
        elif hand_strength > 0.5:  # Medium (top 50%)
            return {'fold': 0.40, 'call': 0.45, 'raise': 0.15}
        elif hand_strength > 0.3:  # Weak (top 70%)
            return {'fold': 0.70, 'call': 0.25, 'raise': 0.05}
        else:  # Trash
            return {'fold': 0.95, 'call': 0.04, 'raise': 0.01}


# Initialize system
logger.info("Initializing CFR+ GTO System v4.0...")
system = CFRPlusSystem()

# Load strategies
STRATEGIES_PATH = Path(__file__).parent / "master_gto_strategies.pkl"
if STRATEGIES_PATH.exists():
    logger.info("✅ Loading pre-trained CFR+ strategies...")
    system.load_strategies(str(STRATEGIES_PATH))
else:
    logger.warning("⚠️ No pre-trained strategies found!")
    logger.info("Strategies will use fallback model based on hand strength")

# Initialize Claude AI coach
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if ANTHROPIC_API_KEY:
    initialize_claude_coach(ANTHROPIC_API_KEY)
    logger.info("✅ Claude AI coach initialized")
else:
    logger.warning("⚠️ ANTHROPIC_API_KEY not set - coaching will be basic")


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
    user_action: Optional[str] = None  # For reviewing decisions


class StrategyResponse(BaseModel):
    success: bool
    gto: Dict
    coaching: str
    hand: Dict
    version: str = "4.0.0"


@app.get("/")
def root():
    return {
        "message": "Professional GTO Poker AI v4.0 - CFR+ with Claude AI",
        "status": "ready",
        "features": [
            "✅ CFR+ algorithm with 1M iterations",
            "✅ Monte Carlo chance sampling",
            "✅ Claude AI personalized coaching",
            "✅ Pattern detection and insights",
            "✅ All 169 starting hands covered",
            "✅ Position-aware strategies"
        ]
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "version": "4.0.0",
        "strategies_loaded": system.loaded,
        "num_trained_states": len(system.strategies),
        "claude_ai_enabled": get_coach() is not None
    }


@app.get("/version")
def get_version():
    return {
        "version": "4.0.0",
        "ai_type": "CFR+ with Monte Carlo Sampling + Claude AI",
        "strategies_loaded": system.loaded,
        "features": {
            "cfr_plus": True,
            "monte_carlo_sampling": True,
            "claude_ai_coaching": get_coach() is not None,
            "pattern_detection": True,
            "all_169_hands": True
        }
    }


@app.post("/analyze-hand", response_model=StrategyResponse)
def analyze_hand(request: StrategyRequest):
    """
    Analyze a poker hand with CFR+ GTO strategy and Claude AI coaching
    """
    try:
        # Parse hand
        hand = ''.join(request.hole_cards)

        # Convert to standard notation (e.g., "AsKs" -> "AKs")
        hand_notation = _convert_to_hand_notation(request.hole_cards)

        # Get GTO strategy from CFR+
        gto_strategy = system.get_strategy(
            hand=hand_notation,
            position=request.position,
            action_facing=request.action_facing,
            stack_depth=request.stack_depth
        )

        # Classify hand
        hand_analysis = _classify_hand(
            hand_notation,
            request.position,
            request.action_facing,
            request.stack_depth
        )

        # Generate Claude AI coaching
        coach = get_coach()
        if coach:
            coaching = coach.generate_coaching(
                hand_analysis=hand_analysis,
                gto_strategy=gto_strategy,
                user_action=request.user_action
            )
        else:
            # Fallback coaching
            coaching = _basic_coaching(hand_analysis, gto_strategy)

        return StrategyResponse(
            success=True,
            gto={
                "strategy": gto_strategy['strategy'],
                "recommended_action": gto_strategy['recommended_action'],
                "bucket": hand_analysis['category'],
                "from_trained_model": gto_strategy['from_trained_model']
            },
            coaching=coaching,
            hand=hand_analysis
        )

    except Exception as e:
        logger.error(f"Error analyzing hand: {e}")
        raise HTTPException(status_code=500, detail=str(e))


def _convert_to_hand_notation(cards: List[str]) -> str:
    """Convert ['As', 'Ks'] to 'AKs' notation"""
    if len(cards) != 2:
        return 'XX'

    ranks = []
    suits = []

    for card in cards:
        if len(card) < 2:
            continue
        rank = card[0]
        suit = card[1]
        ranks.append(rank)
        suits.append(suit)

    # Sort by rank value
    rank_order = {'A': 14, 'K': 13, 'Q': 12, 'J': 11, 'T': 10, '9': 9, '8': 8, '7': 7, '6': 6, '5': 5, '4': 4, '3': 3, '2': 2}

    if rank_order.get(ranks[0], 0) >= rank_order.get(ranks[1], 0):
        high_rank, low_rank = ranks[0], ranks[1]
    else:
        high_rank, low_rank = ranks[1], ranks[0]

    # Check if suited or paired
    if ranks[0] == ranks[1]:
        return f"{high_rank}{low_rank}"  # Pair (e.g., "AA")
    elif suits[0] == suits[1]:
        return f"{high_rank}{low_rank}s"  # Suited (e.g., "AKs")
    else:
        return f"{high_rank}{low_rank}o"  # Offsuit (e.g., "AKo")


def _classify_hand(hand: str, position: str, action_facing: str, stack_depth: int) -> Dict:
    """Classify hand with properties"""
    rank = system.hand_rankings.get(hand, 169)
    percentile = (1 - rank / 169) * 100

    # Category
    if rank <= 20:
        category = "PREMIUM"
    elif rank <= 50:
        category = "STRONG"
    elif rank <= 100:
        category = "MEDIUM"
    elif rank <= 140:
        category = "WEAK"
    else:
        category = "TRASH"

    # Properties
    is_suited = 's' in hand
    is_pair = len(hand) == 2

    return {
        "cards": [hand[0] + 's', hand[1] + 's'] if is_suited else [hand[0] + 'o', hand[1] + 'o'],
        "position": position,
        "action_facing": action_facing,
        "stackSize": str(stack_depth),
        "category": category,
        "properties": {
            "strength_percentile": percentile,
            "blocker_value": 1.0 if hand[0] == 'A' else (0.7 if hand[0] == 'K' else 0.3),
            "playability_score": 90 if is_suited else 70,
            "set_mining_viable": is_pair and rank > 50,
            "position_advantage": 1.0 if position == "BTN" else 0.5
        }
    }


def _basic_coaching(hand_analysis: Dict, gto_strategy: Dict) -> str:
    """Basic coaching if Claude AI is not available"""
    action = gto_strategy['recommended_action'].upper()
    category = hand_analysis['category']

    return f"""## GTO Analysis

**Recommended Action:** {action}

**Hand Category:** {category}

**Strategy Frequencies:**
- Fold: {gto_strategy['strategy']['fold']:.1%}
- Call: {gto_strategy['strategy']['call']:.1%}
- Raise: {gto_strategy['strategy']['raise']:.1%}

**Situation:**
- Position: {hand_analysis['position']}
- Action Facing: {hand_analysis['action_facing']}
- Stack: {hand_analysis['stackSize']}BB

Play this hand according to the recommended action for optimal GTO strategy.
"""


if __name__ == "__main__":
    import uvicorn

    logger.info("=" * 60)
    logger.info("🚀 Professional GTO Poker AI v4.0")
    logger.info("=" * 60)
    logger.info("Starting on port 8000")
    logger.info("Features:")
    logger.info("  ✅ CFR+ with Monte Carlo sampling")
    logger.info("  ✅ Claude AI personalized coaching")
    logger.info("  ✅ 1M iterations training")
    logger.info("=" * 60)

    uvicorn.run(app, host="0.0.0.0", port=8000)

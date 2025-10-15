"""
Professional Poker Taxonomy System
Complete classification framework for poker hands, players, and game situations
"""
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import numpy as np

# ============================================================================
# HAND TAXONOMY - Granular hand classification
# ============================================================================

class HandCategory(Enum):
    """Top-level hand categories"""
    PREMIUM_PAIR = auto()      # AA, KK
    HIGH_PAIR = auto()         # QQ, JJ
    MEDIUM_PAIR = auto()       # TT-77
    LOW_PAIR = auto()          # 66-22
    PREMIUM_BROADWAY = auto()  # AK, AQ
    BROADWAY_SUITED = auto()   # KQs, QJs, etc
    BROADWAY_OFFSUIT = auto()  # KQo, QJo, etc
    SUITED_CONNECTOR = auto()  # T9s-54s
    SUITED_GAPPER = auto()     # J9s, 97s, etc
    SUITED_ACE = auto()        # A9s-A2s
    OFFSUIT_ACE = auto()       # AJo-A2o
    SUITED_KING = auto()       # K9s-K2s
    TRASH = auto()             # 72o, 83o, etc

class HandStrength(Enum):
    """Detailed hand strength levels (0-100 scale)"""
    MONSTER = (90, 100)        # Top 10% of hands
    PREMIUM = (80, 90)         # Top 10-20%
    STRONG = (70, 80)          # Top 20-30%
    GOOD = (60, 70)            # Top 30-40%
    PLAYABLE = (50, 60)        # Top 40-50%
    MARGINAL = (40, 50)        # Top 50-60%
    WEAK = (30, 40)            # Top 60-70%
    POOR = (20, 30)            # Top 70-80%
    TRASH = (10, 20)           # Top 80-90%
    GARBAGE = (0, 10)          # Bottom 10%

@dataclass
class HandProfile:
    """Complete profile of a poker hand"""
    cards: List[str]                    # ['As', 'Kh']
    category: HandCategory              # PREMIUM_BROADWAY
    strength_percentile: float          # 95.0 (top 5%)
    playability_score: float           # 0-100 based on multiple factors
    suited: bool
    connected: bool
    gap: int                           # 0 for connected, 1 for one-gap, etc
    
    # Position-specific recommendations
    ep_vpip: float                     # % of time to play from early position
    mp_vpip: float                     # Middle position
    co_vpip: float                     # Cutoff
    btn_vpip: float                    # Button
    sb_vpip: float                     # Small blind
    bb_vpip: float                     # Big blind defend
    
    # Action recommendations
    rfi_range: bool                    # In raise-first-in range?
    calling_range: bool                # In calling range?
    three_bet_range: bool              # In 3-bet range?
    four_bet_range: bool               # In 4-bet range?
    
    # Special properties
    set_mining_viable: bool            # Good for set mining?
    suited_connector_value: float      # 0-1 for multiway value
    blocker_value: float               # 0-1 for blocking premium hands

# ============================================================================
# PLAYER TAXONOMY - Player type classification
# ============================================================================

class PlayerType(Enum):
    """Primary player archetypes"""
    SHARK = "shark"                    # Professional, winning player
    REG = "reg"                        # Regular, break-even to small winner
    TAG = "tag"                        # Tight-aggressive
    LAG = "lag"                        # Loose-aggressive
    ROCK = "rock"                      # Super tight, passive
    NIT = "nit"                        # Extremely tight
    CALLING_STATION = "station"        # Calls too much
    MANIAC = "maniac"                  # Super aggressive, loose
    FISH = "fish"                      # Recreational, losing player
    WHALE = "whale"                    # Rich recreational, big losses

@dataclass
class PlayerStats:
    """Statistical profile of a player"""
    vpip: float                        # Voluntarily put in pot %
    pfr: float                         # Preflop raise %
    three_bet: float                   # 3-bet %
    fold_to_three_bet: float          # Fold to 3-bet %
    c_bet: float                       # Continuation bet %
    fold_to_c_bet: float              # Fold to c-bet %
    aggression_factor: float           # (Bet + Raise) / Call
    went_to_showdown: float            # WTSD %
    won_at_showdown: float             # W$SD %
    
    # Advanced stats
    four_bet_range: float
    squeeze_frequency: float
    donk_bet_frequency: float
    check_raise_frequency: float
    river_bluff_frequency: float

@dataclass
class PlayerProfile:
    """Complete player profile with all characteristics"""
    player_id: str
    player_type: PlayerType
    stats: PlayerStats
    
    # Skill metrics
    technical_skill: float             # 0-100
    psychological_skill: float         # 0-100
    mathematical_skill: float          # 0-100
    adaptability: float               # 0-100
    tilt_resistance: float            # 0-100
    
    # Bankroll info
    bankroll: float
    average_buyin: float
    risk_tolerance: float              # 0-1 scale
    
    # Session patterns
    average_session_hours: float
    sessions_per_week: int
    preferred_stakes: str
    preferred_games: List[str]        # ['NLH', 'PLO', etc]
    
    # Tax profile (linking to taxation system)
    tax_jurisdiction: str
    tax_rate: float
    professional_status: bool
    loss_deductible: bool

# ============================================================================
# SITUATION TAXONOMY - Game situation classification
# ============================================================================

class GameStage(Enum):
    """Stage of the game"""
    PREFLOP = "preflop"
    FLOP = "flop"
    TURN = "turn"
    RIVER = "river"

class PotType(Enum):
    """Type of pot"""
    HEADS_UP = "headsup"               # 2 players
    MULTIWAY = "multiway"              # 3+ players
    FAMILY = "family"                  # 4+ players

class BoardTexture(Enum):
    """Flop/board texture classification"""
    DRY = "dry"                        # K72 rainbow
    WET = "wet"                        # JT9 two-tone
    MONOTONE = "monotone"              # Three of same suit
    PAIRED = "paired"                  # Contains a pair
    CONNECTED = "connected"            # Straight possible
    BROADWAY = "broadway"              # High cards
    LOW = "low"                        # All cards 8 or lower

class ActionSequence(Enum):
    """Common action sequences"""
    UNOPENED = "unopened"
    SINGLE_RAISE = "single_raise"
    THREE_BET = "3bet"
    FOUR_BET = "4bet"
    FIVE_BET = "5bet"
    LIMP_POT = "limp_pot"
    CHECK_RAISE = "check_raise"
    DONK_BET = "donk_bet"
    PROBE_BET = "probe_bet"

@dataclass
class SituationProfile:
    """Complete situation profile"""
    game_stage: GameStage
    pot_type: PotType
    position: str                      # EP, MP, CO, BTN, SB, BB
    stack_depth: float                 # In BB
    pot_size: float                    # In BB
    
    # Action context
    action_sequence: ActionSequence
    num_players: int
    players_to_act: int
    
    # Board analysis (postflop)
    board_texture: Optional[BoardTexture]
    flush_possible: bool
    straight_possible: bool
    paired_board: bool
    
    # Betting context
    bet_sizing: float                  # As % of pot
    pot_odds: float                    # Required equity
    implied_odds: float                # Stack-to-pot ratio
    
    # ICM/Tournament
    tournament_stage: Optional[str]    # Early, middle, late, final table
    icm_pressure: float                # 0-1 scale
    bubble_factor: float               # Multiplier for bubble play

# ============================================================================
# TAXONOMY ENGINE - Main classification system
# ============================================================================

class PokerTaxonomyEngine:
    """
    Complete poker taxonomy engine for classification and analysis
    """
    
    def __init__(self):
        self.hand_rankings = self._initialize_hand_rankings()
        self.player_classifications = {}
        self.situation_patterns = {}
        
    def _initialize_hand_rankings(self) -> Dict[str, HandProfile]:
        """Initialize all 169 unique starting hands with profiles"""
        rankings = {}
        
        # Premium pairs
        rankings['AA'] = HandProfile(
            cards=['A', 'A'],
            category=HandCategory.PREMIUM_PAIR,
            strength_percentile=99.9,
            playability_score=100,
            suited=False, connected=False, gap=0,
            ep_vpip=100, mp_vpip=100, co_vpip=100,
            btn_vpip=100, sb_vpip=100, bb_vpip=100,
            rfi_range=True, calling_range=True,
            three_bet_range=True, four_bet_range=True,
            set_mining_viable=False,
            suited_connector_value=0,
            blocker_value=1.0
        )
        
        rankings['KK'] = HandProfile(
            cards=['K', 'K'],
            category=HandCategory.PREMIUM_PAIR,
            strength_percentile=99.5,
            playability_score=98,
            suited=False, connected=False, gap=0,
            ep_vpip=100, mp_vpip=100, co_vpip=100,
            btn_vpip=100, sb_vpip=100, bb_vpip=100,
            rfi_range=True, calling_range=True,
            three_bet_range=True, four_bet_range=True,
            set_mining_viable=False,
            suited_connector_value=0,
            blocker_value=0.9
        )
        
        # Add all other hands...
        # (In production, this would include all 169 starting hands)
        
        return rankings
    
    def classify_hand(self, hole_cards: List[str]) -> HandProfile:
        """
        Classify a poker hand and return its complete profile
        """
        # Parse cards
        rank1 = self._card_rank(hole_cards[0][0])
        rank2 = self._card_rank(hole_cards[1][0])
        suit1 = hole_cards[0][1] if len(hole_cards[0]) > 1 else 's'
        suit2 = hole_cards[1][1] if len(hole_cards[1]) > 1 else 's'
        
        suited = (suit1 == suit2)
        high = max(rank1, rank2)
        low = min(rank1, rank2)
        gap = high - low - 1
        paired = (rank1 == rank2)
        
        # Determine category
        if paired:
            if high >= 13:  # AA, KK
                category = HandCategory.PREMIUM_PAIR
            elif high >= 11:  # QQ, JJ
                category = HandCategory.HIGH_PAIR
            elif high >= 7:  # TT-77
                category = HandCategory.MEDIUM_PAIR
            else:  # 66-22
                category = HandCategory.LOW_PAIR
        elif high == 14:  # Ace high
            if low >= 12:  # AK, AQ
                category = HandCategory.PREMIUM_BROADWAY
            elif suited:
                category = HandCategory.SUITED_ACE
            else:
                category = HandCategory.OFFSUIT_ACE
        elif suited and gap <= 1 and high >= 5:
            category = HandCategory.SUITED_CONNECTOR
        elif suited and gap <= 2:
            category = HandCategory.SUITED_GAPPER
        elif high >= 10 and low >= 10:
            if suited:
                category = HandCategory.BROADWAY_SUITED
            else:
                category = HandCategory.BROADWAY_OFFSUIT
        elif high == 13 and suited:
            category = HandCategory.SUITED_KING
        else:
            category = HandCategory.TRASH
        
        # Calculate strength percentile
        strength_percentile = self._calculate_hand_strength(
            high, low, suited, paired, gap
        )
        
        # Calculate playability
        playability = self._calculate_playability(
            category, suited, gap, high, low
        )
        
        # Position-specific VPIP
        ep_vpip = 10 if strength_percentile > 85 else 0
        mp_vpip = 15 if strength_percentile > 80 else 0
        co_vpip = 25 if strength_percentile > 70 else 0
        btn_vpip = 40 if strength_percentile > 60 else 0
        sb_vpip = 35 if strength_percentile > 65 else 0
        bb_vpip = 50 if strength_percentile > 50 else 0
        
        # Action ranges
        rfi_range = strength_percentile > 70
        calling_range = strength_percentile > 60 or (paired and high >= 5)
        three_bet_range = strength_percentile > 85
        four_bet_range = strength_percentile > 95
        
        # Special properties
        set_mining_viable = paired and 5 <= high <= 10
        suited_connector_value = 1.0 if category == HandCategory.SUITED_CONNECTOR else 0
        blocker_value = (high / 14) * 0.5  # Higher cards block more
        
        return HandProfile(
            cards=hole_cards,
            category=category,
            strength_percentile=strength_percentile,
            playability_score=playability,
            suited=suited,
            connected=(gap == 0),
            gap=gap,
            ep_vpip=ep_vpip,
            mp_vpip=mp_vpip,
            co_vpip=co_vpip,
            btn_vpip=btn_vpip,
            sb_vpip=sb_vpip,
            bb_vpip=bb_vpip,
            rfi_range=rfi_range,
            calling_range=calling_range,
            three_bet_range=three_bet_range,
            four_bet_range=four_bet_range,
            set_mining_viable=set_mining_viable,
            suited_connector_value=suited_connector_value,
            blocker_value=blocker_value
        )
    
    def classify_player(self, stats: PlayerStats) -> PlayerType:
        """
        Classify a player based on their statistics
        """
        # VPIP/PFR based classification
        if stats.vpip < 15 and stats.pfr < 10:
            return PlayerType.ROCK
        elif stats.vpip < 20 and stats.pfr < 15:
            return PlayerType.NIT
        elif stats.vpip < 28 and stats.pfr > stats.vpip * 0.7:
            return PlayerType.TAG
        elif stats.vpip > 35 and stats.pfr > 25:
            return PlayerType.LAG
        elif stats.vpip > 50 and stats.pfr > 35:
            return PlayerType.MANIAC
        elif stats.vpip > 35 and stats.pfr < 15:
            return PlayerType.CALLING_STATION
        elif stats.vpip > 40 and stats.went_to_showdown < 20:
            return PlayerType.FISH
        elif stats.aggression_factor > 3:
            return PlayerType.SHARK
        else:
            return PlayerType.REG
    
    def classify_situation(self,
                          game_stage: GameStage,
                          position: str,
                          stack_depth: float,
                          pot_size: float,
                          num_players: int,
                          board: Optional[List[str]] = None) -> SituationProfile:
        """
        Classify a game situation
        """
        # Determine pot type
        if num_players == 2:
            pot_type = PotType.HEADS_UP
        elif num_players >= 4:
            pot_type = PotType.FAMILY
        else:
            pot_type = PotType.MULTIWAY
        
        # Analyze board texture (if postflop)
        board_texture = None
        flush_possible = False
        straight_possible = False
        paired_board = False
        
        if board and len(board) >= 3:
            board_texture = self._analyze_board_texture(board)
            flush_possible = self._check_flush_possible(board)
            straight_possible = self._check_straight_possible(board)
            paired_board = self._check_paired_board(board)
        
        # Calculate pot odds
        pot_odds = 0.33  # Default 2:1
        if pot_size > 0:
            pot_odds = 1 / (pot_size + 1)
        
        # Calculate implied odds
        implied_odds = stack_depth / pot_size if pot_size > 0 else stack_depth
        
        return SituationProfile(
            game_stage=game_stage,
            pot_type=pot_type,
            position=position,
            stack_depth=stack_depth,
            pot_size=pot_size,
            action_sequence=ActionSequence.UNOPENED,
            num_players=num_players,
            players_to_act=num_players - 1,
            board_texture=board_texture,
            flush_possible=flush_possible,
            straight_possible=straight_possible,
            paired_board=paired_board,
            bet_sizing=1.0,
            pot_odds=pot_odds,
            implied_odds=implied_odds,
            tournament_stage=None,
            icm_pressure=0,
            bubble_factor=1.0
        )
    
    def _card_rank(self, card: str) -> int:
        """Convert card to numeric rank"""
        ranks = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7,
                '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        return ranks.get(card, 0)
    
    def _calculate_hand_strength(self, high: int, low: int, 
                                suited: bool, paired: bool, gap: int) -> float:
        """Calculate hand strength percentile (0-100)"""
        # Simplified Sklansky-Chubukov rankings
        base_strength = (high + low) / 28 * 50  # Base on card ranks
        
        if paired:
            base_strength += high * 3  # Pairs are strong
        if suited:
            base_strength += 10  # Suited adds value
        if gap == 0:
            base_strength += 5  # Connected adds value
        
        # Normalize to 0-100
        return min(100, max(0, base_strength))
    
    def _calculate_playability(self, category: HandCategory,
                              suited: bool, gap: int, 
                              high: int, low: int) -> float:
        """Calculate playability score (0-100)"""
        score = 50  # Base score
        
        # Category bonuses
        category_scores = {
            HandCategory.PREMIUM_PAIR: 40,
            HandCategory.HIGH_PAIR: 30,
            HandCategory.PREMIUM_BROADWAY: 35,
            HandCategory.SUITED_CONNECTOR: 25,
            HandCategory.MEDIUM_PAIR: 20,
            HandCategory.SUITED_ACE: 15,
            HandCategory.TRASH: -20
        }
        
        score += category_scores.get(category, 0)
        
        # Modifiers
        if suited:
            score += 10
        if gap == 0:
            score += 5
        if high >= 10:  # Broadway card
            score += 5
            
        return min(100, max(0, score))
    
    def _analyze_board_texture(self, board: List[str]) -> BoardTexture:
        """Analyze and classify board texture"""
        ranks = [self._card_rank(card[0]) for card in board]
        suits = [card[1] if len(card) > 1 else 's' for card in board]
        
        # Check for various textures
        if len(set(suits)) == 1:
            return BoardTexture.MONOTONE
        if len(ranks) != len(set(ranks)):
            return BoardTexture.PAIRED
        if max(ranks) - min(ranks) <= 4 and len(ranks) >= 3:
            return BoardTexture.CONNECTED
        if min(ranks) >= 10:
            return BoardTexture.BROADWAY
        if max(ranks) <= 8:
            return BoardTexture.LOW
        if len(set(suits)) >= 3:
            return BoardTexture.DRY
        
        return BoardTexture.WET
    
    def _check_flush_possible(self, board: List[str]) -> bool:
        """Check if flush is possible on board"""
        suits = [card[1] if len(card) > 1 else 's' for card in board]
        return any(suits.count(suit) >= 3 for suit in set(suits))
    
    def _check_straight_possible(self, board: List[str]) -> bool:
        """Check if straight is possible on board"""
        ranks = sorted([self._card_rank(card[0]) for card in board])
        
        # Check for 3+ cards within 5 rank span
        for i in range(len(ranks) - 2):
            if ranks[i+2] - ranks[i] <= 4:
                return True
        return False
    
    def _check_paired_board(self, board: List[str]) -> bool:
        """Check if board is paired"""
        ranks = [self._card_rank(card[0]) for card in board]
        return len(ranks) != len(set(ranks))
    
    def get_comprehensive_analysis(self,
                                  hole_cards: List[str],
                                  position: str,
                                  stack_depth: float,
                                  facing_action: str,
                                  board: Optional[List[str]] = None) -> Dict:
        """
        Get complete analysis combining hand, situation, and recommendations
        """
        # Classify hand
        hand_profile = self.classify_hand(hole_cards)
        
        # Determine game stage
        game_stage = GameStage.PREFLOP if not board else GameStage.FLOP
        if board and len(board) == 4:
            game_stage = GameStage.TURN
        elif board and len(board) == 5:
            game_stage = GameStage.RIVER
        
        # Classify situation
        situation = self.classify_situation(
            game_stage=game_stage,
            position=position,
            stack_depth=stack_depth,
            pot_size=3.0 if facing_action else 1.5,
            num_players=2,
            board=board
        )
        
        # Generate recommendations based on taxonomy
        recommendations = self._generate_recommendations(
            hand_profile, situation, facing_action
        )
        
        return {
            'hand': {
                'cards': hole_cards,
                'category': hand_profile.category.name,
                'strength_percentile': hand_profile.strength_percentile,
                'playability': hand_profile.playability_score,
                'suited': hand_profile.suited,
                'set_mining_viable': hand_profile.set_mining_viable
            },
            'situation': {
                'stage': situation.game_stage.value,
                'position': position,
                'stack_depth': stack_depth,
                'pot_type': situation.pot_type.value,
                'pot_odds': situation.pot_odds,
                'implied_odds': situation.implied_odds
            },
            'recommendations': recommendations,
            'advanced_metrics': {
                'blocker_value': hand_profile.blocker_value,
                'suited_connector_value': hand_profile.suited_connector_value,
                'position_vpip': getattr(hand_profile, f'{position.lower()}_vpip', 0)
            }
        }
    
    def _generate_recommendations(self,
                                 hand: HandProfile,
                                 situation: SituationProfile,
                                 facing_action: str) -> Dict[str, float]:
        """
        Generate action recommendations based on taxonomy
        """
        if facing_action == "raise":
            # Facing a raise
            if hand.set_mining_viable and situation.implied_odds > 7.5:
                return {'fold': 0.1, 'call': 0.7, 'raise': 0.2}
            elif hand.three_bet_range:
                return {'fold': 0.0, 'call': 0.3, 'raise': 0.7}
            elif hand.calling_range:
                return {'fold': 0.2, 'call': 0.7, 'raise': 0.1}
            else:
                return {'fold': 0.9, 'call': 0.1, 'raise': 0.0}
        else:
            # Unopened pot
            if hand.rfi_range:
                return {'fold': 0.0, 'call': 0.05, 'raise': 0.95}
            else:
                return {'fold': 0.95, 'call': 0.05, 'raise': 0.0}


# Example usage
if __name__ == "__main__":
    print("="*60)
    print("POKER TAXONOMY ENGINE DEMO")
    print("="*60)
    
    engine = PokerTaxonomyEngine()
    
    # Test hand classification
    test_hands = [
        ['As', 'Ah'],  # Pocket aces
        ['7s', '7h'],  # Pocket sevens
        ['8s', '9s'],  # Suited connector
        ['As', 'Kh'],  # Ace king offsuit
        ['7s', '2h'],  # Trash
    ]
    
    for cards in test_hands:
        profile = engine.classify_hand(cards)
        print(f"\n{cards[0]}{cards[1]}:")
        print(f"  Category: {profile.category.name}")
        print(f"  Strength: {profile.strength_percentile:.1f} percentile")
        print(f"  Playability: {profile.playability_score:.1f}/100")
        print(f"  Set mining: {profile.set_mining_viable}")
        print(f"  BTN VPIP: {profile.btn_vpip}%")
        print(f"  BB VPIP: {profile.bb_vpip}%")
    
    # Test comprehensive analysis
    print("\n" + "="*60)
    print("COMPREHENSIVE ANALYSIS: 77 in BB facing BTN raise")
    print("="*60)
    
    analysis = engine.get_comprehensive_analysis(
        hole_cards=['7s', '7h'],
        position='BB',
        stack_depth=100,
        facing_action='raise'
    )
    
    print(f"\nHand Analysis:")
    print(f"  Category: {analysis['hand']['category']}")
    print(f"  Strength: {analysis['hand']['strength_percentile']:.1f} percentile")
    print(f"  Set mining viable: {analysis['hand']['set_mining_viable']}")
    
    print(f"\nSituation:")
    print(f"  Position: {analysis['situation']['position']}")
    print(f"  Stack depth: {analysis['situation']['stack_depth']}BB")
    print(f"  Pot odds: {analysis['situation']['pot_odds']:.2f}")
    print(f"  Implied odds: {analysis['situation']['implied_odds']:.1f}")
    
    print(f"\nRecommendations:")
    for action, freq in analysis['recommendations'].items():
        print(f"  {action}: {freq*100:.0f}%")

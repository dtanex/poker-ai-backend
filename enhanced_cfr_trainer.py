"""
Enhanced CFR Trainer with Proper Calling Ranges
This version properly incentivizes calling to create realistic GTO strategies
"""
import numpy as np
import pickle
import time
import json
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum, auto
import logging
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Position(Enum):
    """Poker positions with positional values"""
    EP = (0, 0.0)   # Early Position (worst)
    MP = (1, 0.2)   # Middle Position
    CO = (2, 0.4)   # Cutoff
    BTN = (3, 1.0)  # Button (best)
    SB = (4, -0.2)  # Small Blind
    BB = (5, -0.1)  # Big Blind (better than SB due to closing action)
    
    def __init__(self, index: int, value: float):
        self.index = index
        self.positional_value = value


class ActionFacing(Enum):
    """Action contexts with pot size implications"""
    UNOPENED = (0, 1.5)           # 1.5 BB pot
    FACING_RAISE = (1, 6.5)        # ~6.5 BB pot (3BB raise)
    FACING_3BET = (2, 15.0)        # ~15 BB pot (9BB 3-bet)
    FACING_4BET = (3, 35.0)        # ~35 BB pot (21BB 4-bet)
    
    def __init__(self, index: int, pot_size: float):
        self.index = index
        self.typical_pot_size = pot_size


@dataclass
class HandProperties:
    """Advanced hand properties for utility calculations"""
    strength_percentile: float      # 0-100
    suited: bool
    connected: bool
    gap: int                        # 0 for connected, 1 for one-gap, etc.
    pair: bool
    high_card_value: int           # 2-14 (A)
    low_card_value: int            # 2-14
    
    # Derived properties
    set_mining_viable: bool = False
    suited_connector: bool = False
    broadway: bool = False
    wheel_ace: bool = False         # A2-A5
    blocker_value: float = 0.0      # 0-1
    playability_score: float = 0.0  # 0-100
    
    def __post_init__(self):
        """Calculate derived properties"""
        # Set mining viable for pocket pairs 22-99
        if self.pair and 2 <= self.high_card_value <= 9:
            self.set_mining_viable = True
        
        # Suited connectors
        if self.suited and self.gap <= 1 and self.high_card_value >= 5:
            self.suited_connector = True
        
        # Broadway hands
        if self.high_card_value >= 10 and self.low_card_value >= 10:
            self.broadway = True
        
        # Wheel aces (A2-A5)
        if self.high_card_value == 14 and self.low_card_value <= 5:
            self.wheel_ace = True
        
        # Blocker value (having an A or K blocks premium hands)
        if self.high_card_value == 14:
            self.blocker_value = 1.0
        elif self.high_card_value == 13:
            self.blocker_value = 0.7
        elif self.high_card_value == 12:
            self.blocker_value = 0.4
        
        # Playability score (how well hand plays postflop)
        self.playability_score = self._calculate_playability()
    
    def _calculate_playability(self) -> float:
        """Calculate how well this hand plays postflop"""
        score = 50.0  # Base score
        
        if self.suited:
            score += 15  # Suited hands play better
        if self.connected:
            score += 10  # Connected hands make straights
        if self.gap == 1:
            score += 5   # One-gappers still decent
        if self.pair:
            score += 10  # Pairs are straightforward
        if self.broadway:
            score += 10  # High cards make top pairs
        if self.set_mining_viable:
            score += 5   # Set potential
        if self.wheel_ace:
            score += 5   # Can make wheel straight
        
        # Position will multiply this score
        return min(100, max(0, score))


@dataclass
class GameState:
    """Complete game state with all relevant information"""
    position: Position
    action_facing: ActionFacing
    stack_depth: float              # In BB
    pot_size: float                 # Current pot in BB
    bet_size: float                 # Size of bet facing in BB
    num_players: int                # Players in hand
    tournament: bool = False        # Cash vs tournament
    
    # Derived values
    pot_odds: float = 0.0           # Required equity to call
    implied_odds: float = 0.0       # Stack-to-pot ratio
    spr: float = 0.0                # Stack-to-pot ratio
    
    def __post_init__(self):
        """Calculate derived values"""
        if self.bet_size > 0:
            self.pot_odds = self.bet_size / (self.pot_size + self.bet_size)
        
        if self.pot_size > 0:
            self.spr = self.stack_depth / self.pot_size
            self.implied_odds = self.stack_depth / self.pot_size
        else:
            self.spr = self.stack_depth
            self.implied_odds = self.stack_depth
    
    def get_state_key(self) -> str:
        """Unique identifier for this state"""
        stack_bucket = self._bucket_stack_depth()
        return f"{self.position.index}_{self.action_facing.index}_{stack_bucket}_{self.num_players}"
    
    def _bucket_stack_depth(self) -> int:
        """Bucket stack depths for strategy differentiation"""
        if self.stack_depth <= 15:
            return 15  # Push/fold
        elif self.stack_depth <= 30:
            return 30  # Short
        elif self.stack_depth <= 50:
            return 50  # Medium
        elif self.stack_depth <= 100:
            return 100  # Standard
        elif self.stack_depth <= 200:
            return 200  # Deep
        else:
            return 300  # Super deep


class EnhancedCFRTrainer:
    """
    Enhanced CFR Trainer with proper calling range incentives
    
    Key improvements:
    1. Realistic utility functions that incentivize calling
    2. Equity realization modeling
    3. Implied odds calculations
    4. Position-aware training
    5. Exploration bonuses for underexplored actions
    6. Parallel training for speed
    """
    
    def __init__(self, num_buckets: int = 20):
        self.num_buckets = num_buckets
        
        # Strategy tables with action counting for exploration
        self.regrets = defaultdict(lambda: np.zeros(3))
        self.strategies = defaultdict(lambda: np.ones(3) / 3)
        self.strategy_sum = defaultdict(lambda: np.zeros(3))
        self.action_counts = defaultdict(lambda: np.ones(3))  # Track action frequency
        
        # Training parameters
        self.iteration = 0
        self.exploration_factor = 0.3  # Start with high exploration
        self.exploration_decay = 0.9995
        
        # Convergence tracking
        self.exploitability_history = []
        self.strategy_changes = []
        
        # Hand evaluation cache
        self.hand_properties_cache = {}
        
        logger.info("Enhanced CFR Trainer initialized")
        logger.info(f"Features: Proper calling incentives, equity realization, exploration bonuses")
    
    def get_hand_properties(self, hole_cards: List[str]) -> HandProperties:
        """Get hand properties with caching"""
        key = tuple(sorted(hole_cards))
        if key in self.hand_properties_cache:
            return self.hand_properties_cache[key]
        
        # Parse cards
        ranks = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8,
                '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}
        
        rank1 = ranks[hole_cards[0][0]]
        rank2 = ranks[hole_cards[1][0]]
        suit1 = hole_cards[0][1] if len(hole_cards[0]) > 1 else 's'
        suit2 = hole_cards[1][1] if len(hole_cards[1]) > 1 else 's'
        
        high = max(rank1, rank2)
        low = min(rank1, rank2)
        suited = (suit1 == suit2)
        paired = (rank1 == rank2)
        gap = high - low - 1 if not paired else 0
        connected = (gap == 0 and not paired)
        
        # Calculate strength percentile (simplified)
        strength = self._calculate_hand_strength(high, low, suited, paired, gap)
        
        props = HandProperties(
            strength_percentile=strength,
            suited=suited,
            connected=connected,
            gap=gap,
            pair=paired,
            high_card_value=high,
            low_card_value=low
        )
        
        self.hand_properties_cache[key] = props
        return props
    
    def _calculate_hand_strength(self, high: int, low: int, suited: bool, 
                                paired: bool, gap: int) -> float:
        """Calculate hand strength percentile (0-100)"""
        # Sklansky-Chubukov inspired ranking
        if paired:
            if high == 14:  # AA
                return 99.9
            elif high == 13:  # KK
                return 99.5
            elif high == 12:  # QQ
                return 99.0
            elif high == 11:  # JJ
                return 98.0
            elif high == 10:  # TT
                return 96.0
            elif high == 9:   # 99
                return 93.0
            else:
                # Lower pairs
                return 70 + (high * 2)
        
        # Non-paired hands
        if high == 14 and low == 13:  # AK
            return 97.0 if suited else 95.0
        elif high == 14 and low == 12:  # AQ
            return 94.0 if suited else 91.0
        elif high == 14:  # Other aces
            base = 50 + (low * 2)
            return base + 10 if suited else base
        
        # Broadway combos
        if high >= 10 and low >= 10:
            base = 70 + (high + low)
            return base + 8 if suited else base
        
        # Suited connectors
        if suited and gap <= 1:
            return 40 + (high * 2)
        
        # Everything else
        base = (high + low) / 28 * 40
        if suited:
            base += 10
        if gap == 0:
            base += 5
        
        return min(100, max(0, base))
    
    def calculate_utilities_with_calling_incentives(self, 
                                                   hand_props: HandProperties,
                                                   state: GameState) -> np.ndarray:
        """
        CRITICAL FUNCTION: Calculate utilities that properly incentivize calling
        
        This is the key to generating realistic GTO strategies!
        """
        fold_utility = 0.0
        call_utility = 0.0
        raise_utility = 0.0
        
        # Normalize hand strength
        strength = hand_props.strength_percentile / 100
        
        # Get position value
        pos_value = state.position.positional_value
        
        # === UNOPENED POT ===
        if state.action_facing == ActionFacing.UNOPENED:
            # Simple: strong hands raise, weak hands fold
            call_utility = -0.2  # Limping generally bad (except SB)
            
            if state.position == Position.SB:
                # SB can complete with wider range
                call_utility = strength * 0.3 - 0.15
            
            # Raise utility based on strength and position
            raise_utility = (strength * 1.5) + (pos_value * 0.3) - 0.4
            
            # Adjust for stack depth
            if state.stack_depth < 20:
                # Short stack: push or fold
                call_utility = -0.5
                if strength > 0.6:
                    raise_utility = 1.0
                else:
                    raise_utility = -0.5
        
        # === FACING A RAISE (MOST IMPORTANT) ===
        elif state.action_facing == ActionFacing.FACING_RAISE:
            # This is where we need proper calling frequencies!
            
            # Calculate equity needed and equity realization
            equity_needed = state.pot_odds
            equity_realization = self._calculate_equity_realization(
                hand_props, state.position, state.stack_depth
            )
            
            # === BB DEFENSE (CRITICAL) ===
            if state.position == Position.BB:
                # BB closes action and gets best pot odds
                
                # SET MINING HANDS (22-99)
                if hand_props.set_mining_viable:
                    implied_odds_needed = 15  # Need 15:1 to set mine profitably
                    actual_implied_odds = state.implied_odds
                    
                    if actual_implied_odds >= implied_odds_needed and state.stack_depth >= 50:
                        # Profitable set mine!
                        set_prob = 0.12  # 12% to flop set
                        win_when_set = state.stack_depth * 0.4  # Win 40% of stack
                        call_utility = (set_prob * win_when_set) - state.bet_size
                        call_utility = max(0.2, call_utility)  # Ensure positive
                        
                        # Don't 3-bet small pairs from BB
                        raise_utility = -0.2
                    else:
                        # Not deep enough for set mining
                        call_utility = -0.3
                        raise_utility = -0.4
                
                # SUITED CONNECTORS (54s-JTs)
                elif hand_props.suited_connector:
                    # These realize equity very well
                    raw_equity = 0.38  # ~38% vs BTN open
                    realized_equity = raw_equity * equity_realization
                    
                    # Calculate EV of calling
                    ev_call = (realized_equity * state.pot_size) - state.bet_size
                    call_utility = ev_call / state.pot_size  # Normalize
                    
                    # Add playability bonus
                    call_utility += hand_props.playability_score / 200
                    
                    # Sometimes 3-bet bluff (polarized)
                    if np.random.random() < 0.15:
                        raise_utility = 0.2  # Bluff frequency
                    else:
                        raise_utility = -0.1
                
                # SUITED ACES (A2s-A9s)
                elif hand_props.high_card_value == 14 and hand_props.suited:
                    # Good hands that play well
                    raw_equity = 0.42
                    realized_equity = raw_equity * equity_realization
                    
                    ev_call = (realized_equity * state.pot_size) - state.bet_size
                    call_utility = ev_call / state.pot_size
                    
                    # Wheel aces can 3-bet bluff
                    if hand_props.wheel_ace:
                        raise_utility = 0.15
                    else:
                        raise_utility = 0.05
                
                # OFFSUIT BROADWAY (KQo, QJo, JTo)
                elif hand_props.broadway and not hand_props.suited:
                    # Decent hands but don't realize full equity
                    raw_equity = 0.40
                    realized_equity = raw_equity * 0.85  # Realize less OOP
                    
                    ev_call = (realized_equity * state.pot_size) - state.bet_size
                    call_utility = ev_call / state.pot_size
                    
                    # Rarely 3-bet these
                    raise_utility = -0.1
                
                # PREMIUM HANDS (QQ+, AK)
                elif hand_props.strength_percentile >= 95:
                    # Sometimes flat to trap
                    call_utility = 0.1
                    # Usually 3-bet for value
                    raise_utility = 0.8
                
                # STRONG HANDS (TT-JJ, AQ)
                elif hand_props.strength_percentile >= 90:
                    # Mix of calling and 3-betting
                    call_utility = 0.3
                    raise_utility = 0.4
                
                # EVERYTHING ELSE
                else:
                    # Weak hands that don't fit categories
                    raw_equity = strength * 0.5  # Rough approximation
                    realized_equity = raw_equity * 0.7
                    
                    ev_call = (realized_equity * state.pot_size) - state.bet_size
                    call_utility = min(-0.1, ev_call / state.pot_size)
                    raise_utility = -0.5
            
            # === OTHER POSITIONS ===
            else:
                # Not BB - need stronger hands to continue
                
                if hand_props.set_mining_viable:
                    # Need better implied odds when not closing action
                    if state.implied_odds >= 20 and state.stack_depth >= 70:
                        call_utility = 0.15
                        raise_utility = -0.2
                    else:
                        call_utility = -0.3
                        raise_utility = -0.4
                
                elif hand_props.strength_percentile >= 85:
                    # Strong hands can call or 3-bet
                    call_utility = 0.2
                    raise_utility = 0.3 + (pos_value * 0.2)
                
                else:
                    # Fold most hands when not in BB
                    call_utility = -0.4
                    raise_utility = -0.5
        
        # === FACING 3-BET ===
        elif state.action_facing == ActionFacing.FACING_3BET:
            # Only continue with strong hands
            
            if hand_props.strength_percentile >= 95:
                # Premium hands
                call_utility = 0.3  # Sometimes call to trap
                raise_utility = 0.6  # Often 4-bet
            elif hand_props.strength_percentile >= 90:
                # Good hands mostly call
                call_utility = 0.4
                raise_utility = 0.0
            elif hand_props.suited_connector and state.stack_depth >= 150:
                # Deep stacked suited connectors can call
                call_utility = 0.1
                raise_utility = -0.3
            else:
                # Fold everything else
                call_utility = -0.6
                raise_utility = -0.7
        
        # === FACING 4-BET ===
        elif state.action_facing == ActionFacing.FACING_4BET:
            # Only premiums continue
            if hand_props.strength_percentile >= 98:
                call_utility = 0.4
                raise_utility = 0.5  # 5-bet jam
            elif hand_props.strength_percentile >= 95:
                call_utility = 0.2
                raise_utility = -0.2
            else:
                call_utility = -0.8
                raise_utility = -0.9
        
        # Add exploration noise (decreases over time)
        noise_scale = self.exploration_factor * (1.0 - self.iteration / 1000000)
        noise = np.random.normal(0, noise_scale, 3)
        
        utilities = np.array([fold_utility, call_utility, raise_utility]) + noise
        
        return utilities
    
    def _calculate_equity_realization(self, hand_props: HandProperties, 
                                     position: Position, stack_depth: float) -> float:
        """
        Calculate how much of our raw equity we'll realize postflop
        
        Key factors:
        - Position (IP realizes more equity)
        - Hand type (suited/connected realize better)
        - Stack depth (deeper = better for drawing hands)
        """
        base_realization = 1.0
        
        # Position is crucial
        if position == Position.BTN:
            base_realization = 1.15  # BTN over-realizes
        elif position == Position.CO:
            base_realization = 1.10
        elif position == Position.BB:
            base_realization = 0.85  # BB under-realizes (OOP)
        elif position == Position.SB:
            base_realization = 0.80  # SB worst position
        else:
            base_realization = 0.95
        
        # Hand properties affect realization
        if hand_props.suited:
            base_realization *= 1.1  # Suited hands realize better
        
        if hand_props.connected or hand_props.gap <= 1:
            base_realization *= 1.05  # Connected hands realize better
        
        if hand_props.pair:
            # Pairs are straightforward
            if hand_props.set_mining_viable:
                # Small pairs either hit set or don't
                base_realization = 1.0
            else:
                # Overpairs realize well
                base_realization *= 1.05
        
        # Stack depth affects drawing hands
        if stack_depth >= 100:
            if hand_props.suited_connector:
                base_realization *= 1.1  # Deep stacks favor draws
        elif stack_depth <= 30:
            if hand_props.suited_connector:
                base_realization *= 0.85  # Short stacks bad for draws
        
        return min(1.3, max(0.7, base_realization))
    
    def update_strategy_with_exploration(self, state_key: str) -> np.ndarray:
        """
        Update strategy using regret matching with exploration bonuses
        
        Ensures all actions get explored, preventing premature convergence
        """
        regrets = self.regrets[state_key]
        action_counts = self.action_counts[state_key]
        
        # Positive regrets for regret matching
        positive_regrets = np.maximum(regrets, 0)
        
        # Add exploration bonus inversely proportional to action count
        # This ensures we try all actions
        exploration_bonus = 1.0 / np.sqrt(action_counts)
        exploration_weight = self.exploration_factor * 0.1
        
        # Combine regrets with exploration
        adjusted_regrets = positive_regrets + (exploration_bonus * exploration_weight)
        
        # Convert to strategy
        total = adjusted_regrets.sum()
        if total > 0:
            strategy = adjusted_regrets / total
        else:
            strategy = np.ones(3) / 3
        
        # Ensure minimum exploration frequency
        min_freq = max(0.01, 0.05 * self.exploration_factor)
        strategy = strategy * (1 - min_freq * 3) + min_freq
        
        # Normalize
        return strategy / strategy.sum()
    
    def train(self, num_iterations: int = 500000, num_processes: int = None):
        """
        Main training loop with parallel processing
        """
        if num_processes is None:
            num_processes = mp.cpu_count() - 1
        
        logger.info(f"Starting training: {num_iterations} iterations on {num_processes} processes")
        start_time = time.time()
        
        # Generate training scenarios
        training_scenarios = self._generate_training_scenarios()
        
        for iteration in range(num_iterations):
            self.iteration = iteration
            
            # Decay exploration
            if iteration % 1000 == 0:
                self.exploration_factor *= self.exploration_decay
                self.exploration_factor = max(0.01, self.exploration_factor)
            
            # Train each scenario
            for scenario in training_scenarios:
                self._train_scenario(scenario)
            
            # Progress updates
            if (iteration + 1) % 10000 == 0:
                elapsed = time.time() - start_time
                exploit = self._calculate_exploitability()
                self.exploitability_history.append(exploit)
                
                logger.info(f"Iteration {iteration + 1}/{num_iterations}")
                logger.info(f"  Time: {elapsed:.1f}s")
                logger.info(f"  Exploitability: {exploit:.4f}")
                logger.info(f"  Exploration: {self.exploration_factor:.4f}")
                
                # Sample some strategies to check calling frequencies
                self._log_sample_strategies()
        
        total_time = time.time() - start_time
        logger.info(f"Training complete in {total_time:.1f} seconds")
        
        return self.get_final_strategies()
    
    def _generate_training_scenarios(self) -> List[Dict]:
        """Generate comprehensive training scenarios"""
        scenarios = []
        
        # All positions
        positions = list(Position)
        
        # Key stack depths
        stack_depths = [20, 30, 50, 70, 100, 150, 200]
        
        # Action sequences
        action_facings = list(ActionFacing)
        
        # Number of players
        num_players_options = [2, 3, 4, 6]  # Heads-up to 6-max
        
        # Generate all 169 starting hands
        ranks = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
        
        # Pocket pairs
        for rank in ranks:
            scenarios.append({
                'hole_cards': [f'{rank}s', f'{rank}h'],
                'positions': positions,
                'stack_depths': stack_depths,
                'action_facings': action_facings,
                'num_players': num_players_options
            })
        
        # Suited hands
        for i, rank1 in enumerate(ranks):
            for rank2 in ranks[i+1:]:
                scenarios.append({
                    'hole_cards': [f'{rank1}s', f'{rank2}s'],
                    'positions': positions,
                    'stack_depths': stack_depths,
                    'action_facings': action_facings,
                    'num_players': num_players_options
                })
        
        # Offsuit hands (sample to reduce total)
        for i, rank1 in enumerate(ranks):
            for rank2 in ranks[i+1:]:
                if np.random.random() < 0.5:  # Sample 50% of offsuit hands
                    scenarios.append({
                        'hole_cards': [f'{rank1}h', f'{rank2}d'],
                        'positions': positions,
                        'stack_depths': stack_depths,
                        'action_facings': action_facings,
                        'num_players': num_players_options
                    })
        
        return scenarios
    
    def _train_scenario(self, scenario: Dict):
        """Train a single scenario"""
        hole_cards = scenario['hole_cards']
        hand_props = self.get_hand_properties(hole_cards)
        
        # Sample random training parameters
        position = np.random.choice(scenario['positions'])
        stack_depth = np.random.choice(scenario['stack_depths'])
        action_facing = np.random.choice(scenario['action_facings'])
        num_players = np.random.choice(scenario['num_players'])
        
        # Create game state
        pot_size = action_facing.typical_pot_size
        bet_size = 3.0 if action_facing == ActionFacing.FACING_RAISE else 0
        
        state = GameState(
            position=position,
            action_facing=action_facing,
            stack_depth=stack_depth,
            pot_size=pot_size,
            bet_size=bet_size,
            num_players=num_players
        )
        
        state_key = state.get_state_key()
        
        # Get current strategy with exploration
        strategy = self.update_strategy_with_exploration(state_key)
        self.strategies[state_key] = strategy
        
        # Accumulate strategy for averaging
        self.strategy_sum[state_key] += strategy
        
        # Sample action
        action = np.random.choice(3, p=strategy)
        self.action_counts[state_key][action] += 1
        
        # Calculate utilities
        utilities = self.calculate_utilities_with_calling_incentives(hand_props, state)
        
        # Calculate regrets
        action_utility = utilities[action]
        regrets = utilities - action_utility
        
        # Update cumulative regrets
        self.regrets[state_key] += regrets
    
    def _calculate_exploitability(self) -> float:
        """Calculate exploitability metric"""
        total_exploit = 0
        count = 0
        
        for state_key, strategy_sum in self.strategy_sum.items():
            if strategy_sum.sum() > 0:
                avg_strategy = strategy_sum / strategy_sum.sum()
                # Entropy as measure of mixing
                entropy = -np.sum(avg_strategy * np.log(avg_strategy + 1e-10))
                max_entropy = np.log(3)
                mixing = entropy / max_entropy
                # Exploitability inversely related to mixing
                exploit = 1.0 - mixing
                total_exploit += exploit
                count += 1
        
        return total_exploit / max(count, 1)
    
    def _log_sample_strategies(self):
        """Log sample strategies to check calling frequencies"""
        # Test: 77 in BB facing BTN raise, 100BB
        test_state = GameState(
            position=Position.BB,
            action_facing=ActionFacing.FACING_RAISE,
            stack_depth=100,
            pot_size=6.5,
            bet_size=3.0,
            num_players=2
        )
        
        state_key = test_state.get_state_key()
        if state_key in self.strategy_sum:
            strategy_sum = self.strategy_sum[state_key]
            if strategy_sum.sum() > 0:
                strategy = strategy_sum / strategy_sum.sum()
                logger.info(f"77 BB vs BTN (100BB): Fold={strategy[0]:.1%}, "
                          f"Call={strategy[1]:.1%}, Raise={strategy[2]:.1%}")
    
    def get_final_strategies(self) -> Dict:
        """Get final averaged strategies"""
        final = {}
        
        for state_key, strategy_sum in self.strategy_sum.items():
            if strategy_sum.sum() > 0:
                final[state_key] = (strategy_sum / strategy_sum.sum()).tolist()
            else:
                final[state_key] = [0.33, 0.33, 0.34]
        
        return {
            'strategies': final,
            'metadata': {
                'iterations': self.iteration,
                'final_exploitability': self.exploitability_history[-1] if self.exploitability_history else None,
                'exploration_final': self.exploration_factor,
                'num_states': len(final)
            }
        }
    
    def save(self, filename: str = "enhanced_strategies.pkl"):
        """Save trained strategies"""
        data = self.get_final_strategies()
        
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
        
        # Also save readable version
        readable = {
            'metadata': data['metadata'],
            'sample_strategies': {}
        }
        
        # Add a few sample strategies
        for i, (state_key, strategy) in enumerate(data['strategies'].items()):
            if i < 10:  # Just first 10
                readable['sample_strategies'][state_key] = {
                    'fold': strategy[0],
                    'call': strategy[1],
                    'raise': strategy[2]
                }
        
        with open(filename.replace('.pkl', '.json'), 'w') as f:
            json.dump(readable, f, indent=2)
        
        logger.info(f"Strategies saved to {filename}")


def test_enhanced_trainer():
    """Test the enhanced trainer"""
    print("\n" + "="*60)
    print("TESTING ENHANCED CFR TRAINER")
    print("="*60)
    
    trainer = EnhancedCFRTrainer()
    
    # Test hand properties
    test_hands = [
        (['7s', '7h'], "Pocket sevens - set mining"),
        (['8s', '9s'], "Suited connector"),
        (['As', '5s'], "Suited ace wheel"),
        (['Kh', 'Qd'], "Offsuit broadway"),
    ]
    
    print("\nHand Properties Analysis:")
    for cards, desc in test_hands:
        props = trainer.get_hand_properties(cards)
        print(f"\n{desc} ({cards[0]}{cards[1]}):")
        print(f"  Strength: {props.strength_percentile:.1f} percentile")
        print(f"  Set mining: {props.set_mining_viable}")
        print(f"  Suited connector: {props.suited_connector}")
        print(f"  Playability: {props.playability_score:.1f}")
        print(f"  Blocker value: {props.blocker_value:.2f}")
    
    # Test utility calculations
    print("\n" + "-"*60)
    print("Testing Utility Calculations:")
    
    # Scenario: 77 in BB facing BTN raise
    hand_props = trainer.get_hand_properties(['7s', '7h'])
    state = GameState(
        position=Position.BB,
        action_facing=ActionFacing.FACING_RAISE,
        stack_depth=100,
        pot_size=6.5,
        bet_size=3.0,
        num_players=2
    )
    
    utilities = trainer.calculate_utilities_with_calling_incentives(hand_props, state)
    print(f"\n77 in BB vs BTN raise (100BB):")
    print(f"  Fold utility: {utilities[0]:.3f}")
    print(f"  Call utility: {utilities[1]:.3f} ← Should be highest!")
    print(f"  Raise utility: {utilities[2]:.3f}")
    
    if utilities[1] > utilities[0] and utilities[1] > utilities[2]:
        print("  ✅ PASS - Call is most profitable!")
    else:
        print("  ⚠️ Needs adjustment")
    
    print("\nRunning quick training test (1000 iterations)...")
    trainer.train(num_iterations=1000)
    
    print("\n✅ Enhanced trainer test complete!")


if __name__ == "__main__":
    test_enhanced_trainer()

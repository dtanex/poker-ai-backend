"""
Advanced Opponent Modeling and Exploitation System
Adapts strategies based on opponent tendencies for maximum EV
"""
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import deque
import json
import time


class PlayerType(Enum):
    """Opponent archetypes with exploitation strategies"""
    UNKNOWN = auto()      # No data yet
    NIT = auto()         # Extremely tight (VPIP < 15%)
    TAG = auto()         # Tight-aggressive (optimal)
    LAG = auto()         # Loose-aggressive
    ROCK = auto()        # Tight-passive
    STATION = auto()     # Loose-passive (calls too much)
    MANIAC = auto()      # Hyper-aggressive
    FISH = auto()        # Recreational player
    REG = auto()         # Regular/competent
    SHARK = auto()       # Highly skilled


@dataclass
class OpponentStats:
    """Comprehensive opponent statistics"""
    # Basic stats
    hands_played: int = 0
    vpip: float = 0.0           # Voluntarily put in pot %
    pfr: float = 0.0            # Preflop raise %
    three_bet: float = 0.0      # 3-bet %
    fold_to_3bet: float = 0.0   # Fold to 3-bet %
    
    # Postflop stats
    c_bet: float = 0.0          # Continuation bet %
    fold_to_c_bet: float = 0.0  # Fold to c-bet %
    aggression_factor: float = 0.0  # (Bet + Raise) / Call
    went_to_showdown: float = 0.0   # WTSD %
    won_at_showdown: float = 0.0    # W$SD %
    
    # Advanced stats
    four_bet_range: float = 0.0
    fold_to_4bet: float = 0.0
    squeeze_frequency: float = 0.0
    steal_attempt: float = 0.0   # Steal from late position
    fold_to_steal: float = 0.0   # Fold BB to steal
    
    # Position-specific stats
    btn_vpip: float = 0.0
    btn_pfr: float = 0.0
    bb_defend: float = 0.0       # BB call/3bet vs raise
    sb_defend: float = 0.0
    
    # Timing tells
    avg_decision_time: float = 0.0
    quick_fold_frequency: float = 0.0
    tank_frequency: float = 0.0
    
    # Bet sizing patterns
    avg_raise_size: float = 0.0
    avg_3bet_size: float = 0.0
    overbet_frequency: float = 0.0
    min_bet_frequency: float = 0.0
    
    # Showdown hands (for range construction)
    shown_hands: List[Tuple[str, str]] = field(default_factory=list)
    
    def get_player_type(self) -> PlayerType:
        """Classify player based on stats"""
        if self.hands_played < 30:
            return PlayerType.UNKNOWN
        
        # Classification logic
        if self.vpip < 15 and self.pfr < 12:
            return PlayerType.NIT
        elif self.vpip < 25 and self.pfr > self.vpip * 0.7:
            return PlayerType.TAG
        elif self.vpip > 30 and self.pfr > 25:
            return PlayerType.LAG
        elif self.vpip < 20 and self.aggression_factor < 1.0:
            return PlayerType.ROCK
        elif self.vpip > 35 and self.aggression_factor < 1.5:
            return PlayerType.STATION
        elif self.vpip > 40 and self.pfr > 30:
            return PlayerType.MANIAC
        elif self.went_to_showdown > 30 and self.won_at_showdown < 45:
            return PlayerType.FISH
        elif self.aggression_factor > 3.0 and self.won_at_showdown > 55:
            return PlayerType.SHARK
        else:
            return PlayerType.REG


@dataclass
class ExploitationStrategy:
    """Specific adjustments to exploit opponent tendencies"""
    # Preflop adjustments
    widen_value_range: float = 0.0      # Multiplier for value hands
    increase_bluff_frequency: float = 0.0
    tighten_calling_range: float = 0.0
    increase_3bet_frequency: float = 0.0
    
    # Postflop adjustments
    increase_c_bet: float = 0.0
    reduce_c_bet: float = 0.0
    call_down_lighter: float = 0.0
    reduce_bluff_frequency: float = 0.0
    
    # Sizing adjustments
    increase_bet_sizing: float = 0.0
    reduce_bet_sizing: float = 0.0
    use_exploitative_sizing: bool = False
    
    # Specific exploits
    attack_weakness: List[str] = field(default_factory=list)
    avoid_strength: List[str] = field(default_factory=list)


class OpponentModeling:
    """
    Advanced opponent modeling and exploitation engine
    """
    
    def __init__(self, hero_name: str = "Hero"):
        self.hero_name = hero_name
        self.opponents = {}  # player_id -> OpponentStats
        self.hand_histories = {}  # player_id -> deque of recent hands
        self.exploitation_strategies = {}  # player_id -> ExploitationStrategy
        self.population_tendencies = self._init_population_tendencies()
        
        # Dynamic adjustment parameters
        self.adjustment_aggression = 1.0  # How aggressively to exploit
        self.sample_size_threshold = 100  # Hands needed for reliable stats
        
        print(f"Opponent Modeling System initialized for {hero_name}")
    
    def _init_population_tendencies(self) -> Dict[str, Dict]:
        """Initialize population tendencies for different stakes/sites"""
        return {
            'micro': {  # $0.01/$0.02 - $0.10/$0.25
                'avg_vpip': 28,
                'avg_pfr': 20,
                'avg_3bet': 7,
                'fish_percentage': 40,
            },
            'small': {  # $0.25/$0.50 - $1/$2
                'avg_vpip': 24,
                'avg_pfr': 19,
                'avg_3bet': 8,
                'fish_percentage': 25,
            },
            'mid': {  # $2/$5 - $5/$10
                'avg_vpip': 22,
                'avg_pfr': 18,
                'avg_3bet': 9,
                'fish_percentage': 15,
            },
            'high': {  # $10/$20+
                'avg_vpip': 21,
                'avg_pfr': 17,
                'avg_3bet': 10,
                'fish_percentage': 5,
            }
        }
    
    def update_opponent(self, player_id: str, action: Dict):
        """
        Update opponent stats based on observed action
        
        Args:
            player_id: Unique player identifier
            action: {
                'position': 'BTN',
                'action_type': 'raise',
                'amount': 3.0,
                'stage': 'preflop',
                'facing_action': 'unopened',
                'hole_cards': ['As', 'Kh'],  # If shown
                'decision_time': 2.5  # seconds
            }
        """
        if player_id not in self.opponents:
            self.opponents[player_id] = OpponentStats()
            self.hand_histories[player_id] = deque(maxlen=1000)
        
        stats = self.opponents[player_id]
        stats.hands_played += 1
        
        # Update preflop stats
        if action['stage'] == 'preflop':
            if action['action_type'] in ['call', 'raise']:
                # VPIP - voluntarily put money in pot
                stats.vpip = self._update_frequency(stats.vpip, 1.0, stats.hands_played)
                
                if action['action_type'] == 'raise':
                    # PFR - preflop raise
                    stats.pfr = self._update_frequency(stats.pfr, 1.0, stats.hands_played)
                    
                    # Position-specific stats
                    if action['position'] == 'BTN':
                        stats.btn_pfr = self._update_frequency(
                            stats.btn_pfr, 1.0, stats.hands_played
                        )
                    
                    # 3-bet detection
                    if action['facing_action'] == 'facing_raise':
                        stats.three_bet = self._update_frequency(
                            stats.three_bet, 1.0, stats.hands_played
                        )
                    
                    # 4-bet detection
                    if action['facing_action'] == 'facing_3bet':
                        stats.four_bet_range = self._update_frequency(
                            stats.four_bet_range, 1.0, stats.hands_played
                        )
                    
                    # Steal attempt
                    if action['position'] in ['CO', 'BTN', 'SB'] and \
                       action['facing_action'] == 'unopened':
                        stats.steal_attempt = self._update_frequency(
                            stats.steal_attempt, 1.0, stats.hands_played
                        )
                
                # BB defense
                if action['position'] == 'BB' and action['facing_action'] == 'facing_raise':
                    stats.bb_defend = self._update_frequency(
                        stats.bb_defend, 1.0, stats.hands_played
                    )
            
            elif action['action_type'] == 'fold':
                # Fold to 3-bet
                if action['facing_action'] == 'facing_3bet':
                    stats.fold_to_3bet = self._update_frequency(
                        stats.fold_to_3bet, 1.0, stats.hands_played
                    )
                
                # Fold to 4-bet
                if action['facing_action'] == 'facing_4bet':
                    stats.fold_to_4bet = self._update_frequency(
                        stats.fold_to_4bet, 1.0, stats.hands_played
                    )
                
                # Fold to steal
                if action['position'] == 'BB' and action.get('vs_steal', False):
                    stats.fold_to_steal = self._update_frequency(
                        stats.fold_to_steal, 1.0, stats.hands_played
                    )
        
        # Update postflop stats
        elif action['stage'] in ['flop', 'turn', 'river']:
            if action['action_type'] == 'bet':
                # C-bet detection
                if action.get('is_aggressor', False) and action['stage'] == 'flop':
                    stats.c_bet = self._update_frequency(stats.c_bet, 1.0, stats.hands_played)
            
            elif action['action_type'] == 'fold':
                # Fold to c-bet
                if action.get('facing_c_bet', False):
                    stats.fold_to_c_bet = self._update_frequency(
                        stats.fold_to_c_bet, 1.0, stats.hands_played
                    )
        
        # Update timing tells
        if 'decision_time' in action:
            stats.avg_decision_time = self._update_average(
                stats.avg_decision_time, action['decision_time'], stats.hands_played
            )
            
            if action['decision_time'] < 1.0:
                stats.quick_fold_frequency = self._update_frequency(
                    stats.quick_fold_frequency, 1.0, stats.hands_played
                )
            elif action['decision_time'] > 10.0:
                stats.tank_frequency = self._update_frequency(
                    stats.tank_frequency, 1.0, stats.hands_played
                )
        
        # Update bet sizing patterns
        if action['action_type'] in ['raise', 'bet'] and 'amount' in action:
            stats.avg_raise_size = self._update_average(
                stats.avg_raise_size, action['amount'], stats.hands_played
            )
            
            if action['amount'] > action.get('pot_size', 1) * 1.5:
                stats.overbet_frequency = self._update_frequency(
                    stats.overbet_frequency, 1.0, stats.hands_played
                )
            elif action['amount'] < action.get('pot_size', 1) * 0.4:
                stats.min_bet_frequency = self._update_frequency(
                    stats.min_bet_frequency, 1.0, stats.hands_played
                )
        
        # Store shown hands for range construction
        if 'hole_cards' in action and action.get('shown', False):
            stats.shown_hands.append((action['hole_cards'], action['position']))
        
        # Store hand history
        self.hand_histories[player_id].append(action)
        
        # Update exploitation strategy
        self._update_exploitation_strategy(player_id)
    
    def _update_frequency(self, old_freq: float, observed: float, count: int) -> float:
        """Update frequency stat with incremental average"""
        return ((old_freq * (count - 1)) + observed) / count
    
    def _update_average(self, old_avg: float, new_value: float, count: int) -> float:
        """Update average with incremental calculation"""
        return ((old_avg * (count - 1)) + new_value) / count
    
    def _update_exploitation_strategy(self, player_id: str):
        """
        Generate exploitation strategy based on opponent stats
        
        This is where the magic happens - converting stats into adjustments!
        """
        stats = self.opponents[player_id]
        player_type = stats.get_player_type()
        
        # Start with neutral strategy
        exploit = ExploitationStrategy()
        
        # Not enough data - use population tendencies
        if stats.hands_played < 30:
            return
        
        # === EXPLOIT NITS ===
        if player_type == PlayerType.NIT:
            exploit.increase_bluff_frequency = 0.5  # Bluff more - they fold
            exploit.tighten_calling_range = 0.3     # Their raises are strong
            exploit.increase_3bet_frequency = 0.4   # They fold to 3-bets
            exploit.attack_weakness = ['steal_blinds', 'c_bet_bluff', 'overbet_bluff']
            exploit.avoid_strength = ['call_3bet', 'call_4bet']
        
        # === EXPLOIT CALLING STATIONS ===
        elif player_type == PlayerType.STATION:
            exploit.widen_value_range = 0.3         # Value bet thinner
            exploit.reduce_bluff_frequency = 0.6    # Don't bluff - they call
            exploit.increase_bet_sizing = 0.2       # Bigger value bets
            exploit.attack_weakness = ['value_bet_thin', 'triple_barrel_value']
            exploit.avoid_strength = ['bluff', 'semi_bluff']
        
        # === EXPLOIT MANIACS ===
        elif player_type == PlayerType.MANIAC:
            exploit.tighten_calling_range = -0.2    # Call down lighter
            exploit.call_down_lighter = 0.4         # They bluff too much
            exploit.reduce_bluff_frequency = 0.3    # Let them hang themselves
            exploit.attack_weakness = ['call_down', 'trap', 'check_raise']
            exploit.avoid_strength = ['fold_to_aggression']
        
        # === EXPLOIT LAGs ===
        elif player_type == PlayerType.LAG:
            # Competent but exploitable if too aggressive
            exploit.increase_3bet_frequency = 0.2   # 3-bet them more
            exploit.call_down_lighter = 0.2         # They barrel too much
            exploit.attack_weakness = ['4bet_bluff', 'float_flop']
        
        # === EXPLOIT ROCKS ===
        elif player_type == PlayerType.ROCK:
            # Super tight and passive
            exploit.increase_bluff_frequency = 0.4  # They fold too much
            exploit.increase_c_bet = 0.3           # C-bet relentlessly
            exploit.attack_weakness = ['steal_blinds', 'barrel_scare_cards']
            exploit.avoid_strength = ['call_raises']
        
        # === EXPLOIT FISH ===
        elif player_type == PlayerType.FISH:
            # Recreational - play straightforward
            exploit.widen_value_range = 0.4        # Value bet wide
            exploit.reduce_bluff_frequency = 0.5   # Don't get fancy
            exploit.use_exploitative_sizing = True  # Use non-standard sizes
            exploit.attack_weakness = ['isolate', 'value_bet', 'pot_control']
        
        # === SPECIFIC STAT-BASED EXPLOITS ===
        
        # Fold to 3-bet exploit
        if stats.fold_to_3bet > 70:
            exploit.increase_3bet_frequency += 0.3  # 3-bet bluff more
        elif stats.fold_to_3bet < 40:
            exploit.increase_3bet_frequency -= 0.2  # 3-bet tighter for value
        
        # C-bet exploit
        if stats.fold_to_c_bet > 60:
            exploit.increase_c_bet += 0.3          # C-bet bluff more
        elif stats.fold_to_c_bet < 40:
            exploit.reduce_c_bet += 0.3            # C-bet less, more value
        
        # WTSD exploit
        if stats.went_to_showdown > 30:
            exploit.widen_value_range += 0.2       # Value bet thin
            exploit.reduce_bluff_frequency += 0.3  # Don't bluff stations
        elif stats.went_to_showdown < 20:
            exploit.increase_bluff_frequency += 0.2  # Bluff more
        
        # Timing tell exploits
        if stats.quick_fold_frequency > 0.7:
            # Folds quickly when weak
            exploit.attack_weakness.append('time_pressure')
        if stats.tank_frequency > 0.2:
            # Tanks with tough decisions
            exploit.attack_weakness.append('put_to_decision')
        
        # Store exploitation strategy
        self.exploitation_strategies[player_id] = exploit
    
    def get_adjusted_strategy(self, base_strategy: Dict[str, float], 
                             opponent_id: str,
                             situation: str = "general") -> Dict[str, float]:
        """
        Adjust base GTO strategy based on opponent tendencies
        
        Args:
            base_strategy: {'fold': 0.3, 'call': 0.5, 'raise': 0.2}
            opponent_id: Player to exploit
            situation: Context for adjustment
        
        Returns:
            Exploitatively adjusted strategy
        """
        if opponent_id not in self.exploitation_strategies:
            return base_strategy  # No exploitation available
        
        exploit = self.exploitation_strategies[opponent_id]
        adjusted = base_strategy.copy()
        
        # Apply exploitative adjustments
        if situation == "facing_raise":
            # Adjust calling range
            adjusted['call'] *= (1 - exploit.tighten_calling_range)
            # Adjust 3-bet frequency
            adjusted['raise'] *= (1 + exploit.increase_3bet_frequency)
        
        elif situation == "unopened":
            # Adjust opening range
            if exploit.widen_value_range > 0:
                adjusted['raise'] *= (1 + exploit.widen_value_range)
        
        elif situation == "facing_3bet":
            # Against aggressive players, call more
            if exploit.call_down_lighter > 0:
                adjusted['call'] *= (1 + exploit.call_down_lighter)
                adjusted['fold'] *= (1 - exploit.call_down_lighter * 0.5)
        
        # Normalize probabilities
        total = sum(adjusted.values())
        if total > 0:
            for action in adjusted:
                adjusted[action] /= total
        
        return adjusted
    
    def get_opponent_range_estimate(self, player_id: str, 
                                   position: str,
                                   action: str) -> List[str]:
        """
        Estimate opponent's range based on stats and history
        
        Returns list of hands in estimated range
        """
        if player_id not in self.opponents:
            return self._get_default_range(position, action)
        
        stats = self.opponents[player_id]
        player_type = stats.get_player_type()
        
        # Build range based on VPIP/PFR and player type
        if action == "open":
            if position == "EP":
                if player_type == PlayerType.NIT:
                    # Nit EP open: QQ+, AK
                    return ['AA', 'KK', 'QQ', 'AKs', 'AKo']
                elif player_type == PlayerType.LAG:
                    # LAG EP open: 77+, A9s+, ATo+, KJs+, QJs+
                    return self._build_range_from_percentage(15)
                else:
                    # Standard EP: TT+, AQ+, KQs
                    return self._build_range_from_percentage(8)
            
            elif position == "BTN":
                # Use BTN-specific stats if available
                if stats.btn_pfr > 0:
                    return self._build_range_from_percentage(stats.btn_pfr)
                else:
                    return self._build_range_from_percentage(stats.pfr * 1.5)
        
        elif action == "3bet":
            # Use 3-bet stat
            return self._build_range_from_percentage(stats.three_bet)
        
        return self._get_default_range(position, action)
    
    def _build_range_from_percentage(self, percentage: float) -> List[str]:
        """Build hand range from top X% of hands"""
        # Simplified - in production use full hand ranking
        all_hands = [
            'AA', 'KK', 'QQ', 'AKs', 'JJ', 'AKo', 'TT', 'AQs', 
            'KQs', '99', 'AJs', 'AQo', '88', 'KJs', 'QJs', 'ATs',
            'KQo', '77', 'JTs', 'A9s', 'KTs', 'AJo', 'QTs', '66',
            'K9s', 'T9s', 'A8s', 'J9s', 'Q9s', 'KJo', '55', 'QJo',
            'A7s', 'A5s', '98s', 'ATo', 'JTo', '44', '87s', 'K8s',
            'A6s', 'T8s', '33', '76s', 'A4s', 'K7s', '97s', '22'
        ]
        
        num_hands = int(len(all_hands) * percentage / 100)
        return all_hands[:max(1, num_hands)]
    
    def _get_default_range(self, position: str, action: str) -> List[str]:
        """Get default range when no stats available"""
        defaults = {
            ('EP', 'open'): ['AA', 'KK', 'QQ', 'JJ', 'TT', 'AKs', 'AKo', 'AQs'],
            ('MP', 'open'): ['AA', 'KK', 'QQ', 'JJ', 'TT', '99', 'AKs', 'AKo', 'AQs', 'AQo'],
            ('CO', 'open'): self._build_range_from_percentage(25),
            ('BTN', 'open'): self._build_range_from_percentage(40),
            ('SB', 'open'): self._build_range_from_percentage(35),
            ('BB', '3bet'): ['AA', 'KK', 'QQ', 'AKs', 'AKo'],
        }
        
        return defaults.get((position, action), ['AA', 'KK'])
    
    def generate_report(self, player_id: str) -> Dict:
        """Generate comprehensive opponent report"""
        if player_id not in self.opponents:
            return {"error": "No data for player"}
        
        stats = self.opponents[player_id]
        player_type = stats.get_player_type()
        exploit = self.exploitation_strategies.get(player_id, ExploitationStrategy())
        
        return {
            'player_id': player_id,
            'hands_played': stats.hands_played,
            'player_type': player_type.name,
            'key_stats': {
                'VPIP': f"{stats.vpip:.1%}",
                'PFR': f"{stats.pfr:.1%}",
                '3-Bet': f"{stats.three_bet:.1%}",
                'Fold to 3-Bet': f"{stats.fold_to_3bet:.1%}",
                'C-Bet': f"{stats.c_bet:.1%}",
                'WTSD': f"{stats.went_to_showdown:.1%}",
                'W$SD': f"{stats.won_at_showdown:.1%}",
            },
            'weaknesses': exploit.attack_weakness,
            'strengths': exploit.avoid_strength,
            'exploitation_adjustments': {
                'Bluff frequency': f"{exploit.increase_bluff_frequency:+.1%}",
                'Value range': f"{exploit.widen_value_range:+.1%}",
                '3-bet frequency': f"{exploit.increase_3bet_frequency:+.1%}",
                'Calling range': f"{exploit.tighten_calling_range:+.1%}",
            },
            'recent_hands': list(self.hand_histories[player_id])[-5:] if player_id in self.hand_histories else []
        }
    
    def save_database(self, filename: str = "opponent_database.json"):
        """Save opponent database to file"""
        data = {
            'opponents': {},
            'exploitation_strategies': {},
            'metadata': {
                'total_players': len(self.opponents),
                'timestamp': time.time()
            }
        }
        
        # Convert dataclasses to dicts
        for player_id, stats in self.opponents.items():
            data['opponents'][player_id] = {
                'hands_played': stats.hands_played,
                'vpip': stats.vpip,
                'pfr': stats.pfr,
                'three_bet': stats.three_bet,
                'player_type': stats.get_player_type().name
            }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Opponent database saved: {len(self.opponents)} players")


def demo_opponent_modeling():
    """Demonstrate opponent modeling system"""
    print("\n" + "="*60)
    print("OPPONENT MODELING AND EXPLOITATION DEMO")
    print("="*60)
    
    # Create modeling system
    modeler = OpponentModeling("Hero")
    
    # Simulate some opponents
    print("\nSimulating opponent actions...")
    
    # Simulate a NIT
    for _ in range(50):
        modeler.update_opponent("NitPlayer", {
            'position': np.random.choice(['EP', 'MP', 'CO', 'BTN', 'SB', 'BB']),
            'action_type': np.random.choice(['fold', 'raise'], p=[0.85, 0.15]),
            'stage': 'preflop',
            'facing_action': 'unopened',
            'decision_time': np.random.uniform(1, 3)
        })
    
    # Simulate a MANIAC
    for _ in range(50):
        modeler.update_opponent("ManiacPlayer", {
            'position': np.random.choice(['EP', 'MP', 'CO', 'BTN', 'SB', 'BB']),
            'action_type': np.random.choice(['fold', 'call', 'raise'], p=[0.3, 0.2, 0.5]),
            'stage': 'preflop',
            'facing_action': 'unopened',
            'decision_time': np.random.uniform(0.5, 2)
        })
    
    # Simulate a CALLING STATION
    for _ in range(50):
        modeler.update_opponent("StationPlayer", {
            'position': np.random.choice(['EP', 'MP', 'CO', 'BTN', 'SB', 'BB']),
            'action_type': np.random.choice(['fold', 'call', 'raise'], p=[0.2, 0.65, 0.15]),
            'stage': 'preflop',
            'facing_action': 'facing_raise',
            'decision_time': np.random.uniform(2, 5)
        })
    
    # Generate reports
    print("\n" + "-"*60)
    print("OPPONENT REPORTS:")
    
    for player_id in ["NitPlayer", "ManiacPlayer", "StationPlayer"]:
        report = modeler.generate_report(player_id)
        print(f"\n{player_id} ({report['player_type']}):")
        print(f"  Hands: {report['hands_played']}")
        print(f"  Key stats: {report['key_stats']}")
        print(f"  Weaknesses: {', '.join(report['weaknesses'])}")
        print(f"  Exploitation adjustments:")
        for adj, value in report['exploitation_adjustments'].items():
            if value != "+0.0%":
                print(f"    {adj}: {value}")
    
    # Test strategy adjustment
    print("\n" + "-"*60)
    print("STRATEGY ADJUSTMENTS:")
    
    base_strategy = {'fold': 0.3, 'call': 0.5, 'raise': 0.2}
    print(f"\nBase GTO strategy: {base_strategy}")
    
    for player_id in ["NitPlayer", "ManiacPlayer", "StationPlayer"]:
        adjusted = modeler.get_adjusted_strategy(base_strategy, player_id, "facing_raise")
        print(f"\nvs {player_id}:")
        print(f"  Adjusted: {adjusted}")
    
    # Save database
    modeler.save_database("opponent_database.json")
    
    print("\n✅ Opponent modeling demo complete!")


if __name__ == "__main__":
    demo_opponent_modeling()

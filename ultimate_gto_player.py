"""
Ultimate Integrated GTO AI Player
Combines CFR strategies, poker taxonomy, and taxation for complete decision making
"""
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import sys
sys.path.append('/mnt/user-data/outputs')

from poker_taxonomy import (
    PokerTaxonomyEngine, HandProfile, HandCategory,
    PlayerType, PlayerStats, SituationProfile, GameStage
)
from poker_taxation import (
    TaxationSystem, TaxJurisdiction, PlayerStatus, TaxProfile
)

@dataclass
class IntegratedGameState:
    """Complete game state with all context"""
    hole_cards: List[str]
    board: List[str]
    position: str
    stack_depth: float
    pot_size: float
    bet_size: float
    num_players: int
    
    # Taxonomy context
    hand_profile: HandProfile
    situation_profile: SituationProfile
    
    # Tax context
    tax_profile: TaxProfile
    after_tax_pot_odds: float
    
    # Opponent context
    villain_type: Optional[PlayerType] = None
    villain_stats: Optional[PlayerStats] = None
    
    # History context
    session_profit: float = 0
    hands_played: int = 0
    current_image: str = "unknown"  # tight, loose, aggressive, etc

class UltimateGTOPlayer:
    """
    The most advanced poker AI player ever created
    Integrates CFR, taxonomy, and taxation for optimal decisions
    """
    
    def __init__(self, 
                 strategy_file: str = "gto_strategies.pkl",
                 tax_jurisdiction: TaxJurisdiction = TaxJurisdiction.USA_PROFESSIONAL,
                 player_status: PlayerStatus = PlayerStatus.PROFESSIONAL):
        """
        Initialize the ultimate player with all systems
        """
        # Load CFR strategies
        self.strategies = self._load_strategies(strategy_file)
        
        # Initialize taxonomy engine
        self.taxonomy = PokerTaxonomyEngine()
        
        # Initialize taxation system
        self.tax_system = TaxationSystem()
        self.tax_profile = self.tax_system.tax_profiles[(tax_jurisdiction, player_status)]
        
        # Session tracking
        self.session_history = []
        self.session_profit = 0
        self.hands_played = 0
        
        # Player profiling
        self.opponent_profiles = {}
        
        print("🎰 Ultimate GTO Player Initialized")
        print(f"   Tax Jurisdiction: {tax_jurisdiction.value}")
        print(f"   Tax Rate: {self.tax_profile.marginal_rate*100:.0f}%")
        print(f"   Taxonomy Engine: Active")
        print(f"   CFR Strategies: Loaded")
        print("   Ready for integrated decision making!")
    
    def _load_strategies(self, strategy_file: str) -> Dict:
        """Load pre-trained CFR strategies"""
        try:
            with open(strategy_file, 'rb') as f:
                return pickle.load(f)
        except:
            print("⚠️  No strategy file found, using defaults")
            return {'preflop': {}, 'postflop': {}}
    
    def make_decision(self, game_state: Dict) -> Dict:
        """
        Make a complete decision considering all factors
        
        This is the MASTER FUNCTION that integrates everything!
        """
        # Parse input
        hole_cards = game_state['hole_cards']
        board = game_state.get('board', [])
        position = game_state['position']
        stack_depth = game_state['stack_depth']
        pot_size = game_state.get('pot_size', 1.5)
        bet_size = game_state.get('bet_size', 0)
        facing_action = game_state.get('facing_action', 'unopened')
        
        # Step 1: Taxonomic Analysis
        hand_profile = self.taxonomy.classify_hand(hole_cards)
        
        game_stage = GameStage.PREFLOP if not board else GameStage.FLOP
        if len(board) == 4:
            game_stage = GameStage.TURN
        elif len(board) == 5:
            game_stage = GameStage.RIVER
        
        situation_profile = self.taxonomy.classify_situation(
            game_stage=game_stage,
            position=position,
            stack_depth=stack_depth,
            pot_size=pot_size,
            num_players=game_state.get('num_players', 2),
            board=board
        )
        
        # Step 2: Tax-Adjusted Pot Odds
        base_pot_odds = bet_size / (pot_size + bet_size) if bet_size > 0 else 0
        after_tax_pot_odds = self.tax_system.adjust_calling_threshold(
            base_pot_odds, self.tax_profile
        )
        
        # Step 3: Get Base CFR Strategy
        base_strategy = self._get_cfr_strategy(
            hand_profile, situation_profile, facing_action
        )
        
        # Step 4: Apply Tax Adjustments
        tax_adjusted_strategy = self._apply_tax_adjustments(
            base_strategy, after_tax_pot_odds, base_pot_odds
        )
        
        # Step 5: Apply Taxonomic Adjustments
        final_strategy = self._apply_taxonomic_adjustments(
            tax_adjusted_strategy, hand_profile, situation_profile
        )
        
        # Step 6: Consider Opponent Profile (if available)
        if 'villain_id' in game_state and game_state['villain_id'] in self.opponent_profiles:
            villain_profile = self.opponent_profiles[game_state['villain_id']]
            final_strategy = self._apply_opponent_adjustments(
                final_strategy, villain_profile
            )
        
        # Step 7: Generate Complete Decision Package
        decision = self._create_decision_package(
            final_strategy, hand_profile, situation_profile,
            base_pot_odds, after_tax_pot_odds
        )
        
        # Update session tracking
        self.hands_played += 1
        
        return decision
    
    def _get_cfr_strategy(self, 
                         hand_profile: HandProfile,
                         situation_profile: SituationProfile,
                         facing_action: str) -> Dict[str, float]:
        """
        Get base CFR strategy for the situation
        """
        if situation_profile.game_stage == GameStage.PREFLOP:
            # Map hand to CFR bucket
            bucket = self._hand_to_bucket(hand_profile)
            
            # Get position and action keys
            pos_key = self._position_to_key(situation_profile.position)
            action_key = self._action_to_key(facing_action)
            
            # Look up strategy
            state_key = f"{pos_key}_{action_key}_{int(situation_profile.stack_depth)}"
            
            if bucket in self.strategies.get('preflop', {}) and \
               state_key in self.strategies['preflop'][bucket]:
                strategy_array = self.strategies['preflop'][bucket][state_key]
                return {'fold': strategy_array[0], 
                       'call': strategy_array[1], 
                       'raise': strategy_array[2]}
        
        # Default strategy based on hand strength
        if hand_profile.strength_percentile > 80:
            return {'fold': 0.0, 'call': 0.2, 'raise': 0.8}
        elif hand_profile.strength_percentile > 60:
            return {'fold': 0.1, 'call': 0.6, 'raise': 0.3}
        elif hand_profile.strength_percentile > 40:
            return {'fold': 0.3, 'call': 0.5, 'raise': 0.2}
        else:
            return {'fold': 0.7, 'call': 0.2, 'raise': 0.1}
    
    def _apply_tax_adjustments(self,
                              strategy: Dict[str, float],
                              after_tax_pot_odds: float,
                              base_pot_odds: float) -> Dict[str, float]:
        """
        Adjust strategy based on taxation
        """
        if self.tax_profile.marginal_rate == 0:
            return strategy  # No tax, no adjustment
        
        # Get tax modifiers
        modifiers = self.tax_system.get_tax_adjusted_strategy_modifiers(self.tax_profile)
        
        # Apply modifiers
        adjusted = strategy.copy()
        
        # Reduce aggression
        adjusted['raise'] *= modifiers['aggression_factor']
        
        # Adjust calling based on pot odds change
        if after_tax_pot_odds > base_pot_odds * 1.2:
            # Need much better odds after tax
            adjusted['call'] *= 0.7
            adjusted['fold'] += adjusted['call'] * 0.3
        
        # Reduce bluffing
        if 'bluff_frequency' in modifiers:
            # Assume some raises are bluffs
            bluff_portion = adjusted['raise'] * 0.3
            adjusted['raise'] -= bluff_portion * (1 - modifiers['bluff_frequency'])
            adjusted['fold'] += bluff_portion * (1 - modifiers['bluff_frequency'])
        
        # Normalize
        total = sum(adjusted.values())
        if total > 0:
            for action in adjusted:
                adjusted[action] /= total
        
        return adjusted
    
    def _apply_taxonomic_adjustments(self,
                                    strategy: Dict[str, float],
                                    hand_profile: HandProfile,
                                    situation_profile: SituationProfile) -> Dict[str, float]:
        """
        Fine-tune strategy based on hand taxonomy
        """
        adjusted = strategy.copy()
        
        # Set mining adjustment
        if hand_profile.set_mining_viable and situation_profile.implied_odds > 7.5:
            # Increase calling for set mining
            adjusted['call'] = min(0.8, adjusted['call'] * 1.5)
            adjusted['raise'] *= 0.5
            adjusted['fold'] = 1 - adjusted['call'] - adjusted['raise']
        
        # Suited connector adjustment
        if hand_profile.suited_connector_value > 0.7:
            # Good for multiway pots
            if situation_profile.pot_type.value == "multiway":
                adjusted['call'] *= 1.3
                adjusted['raise'] *= 0.7
        
        # Blocker adjustment
        if hand_profile.blocker_value > 0.8:
            # Strong blockers favor 3betting
            adjusted['raise'] *= 1.2
            adjusted['call'] *= 0.8
        
        # Position adjustment
        if situation_profile.position == "BTN":
            adjusted['raise'] *= 1.1  # More aggressive on button
        elif situation_profile.position == "EP":
            adjusted['raise'] *= 0.9  # Tighter early position
        
        # Normalize
        total = sum(adjusted.values())
        if total > 0:
            for action in adjusted:
                adjusted[action] /= total
        
        return adjusted
    
    def _apply_opponent_adjustments(self,
                                   strategy: Dict[str, float],
                                   villain_profile: Dict) -> Dict[str, float]:
        """
        Exploit opponent tendencies
        """
        adjusted = strategy.copy()
        villain_type = villain_profile.get('type', PlayerType.REG)
        
        if villain_type == PlayerType.NIT:
            # Nits fold too much
            adjusted['raise'] *= 1.3  # Bluff more
            adjusted['call'] *= 0.7  # Call less (they have it)
        elif villain_type == PlayerType.CALLING_STATION:
            # Stations call too much
            adjusted['raise'] *= 0.7  # Bluff less
            adjusted['call'] *= 1.2  # Value bet more
        elif villain_type == PlayerType.MANIAC:
            # Maniacs are too aggressive
            adjusted['call'] *= 1.3  # Call down lighter
            adjusted['fold'] *= 0.7  # Don't fold as much
        elif villain_type == PlayerType.FISH:
            # Fish make mistakes
            adjusted['raise'] *= 1.2  # Value bet more
            # Keep it simple, no fancy plays
        
        # Normalize
        total = sum(adjusted.values())
        if total > 0:
            for action in adjusted:
                adjusted[action] /= total
        
        return adjusted
    
    def _create_decision_package(self,
                                strategy: Dict[str, float],
                                hand_profile: HandProfile,
                                situation_profile: SituationProfile,
                                base_pot_odds: float,
                                after_tax_pot_odds: float) -> Dict:
        """
        Create comprehensive decision package with explanations
        """
        # Select action based on strategy
        actions = list(strategy.keys())
        probabilities = list(strategy.values())
        recommended_action = np.random.choice(actions, p=probabilities)
        
        # Generate explanation
        explanation = self._generate_explanation(
            recommended_action, strategy, hand_profile, 
            situation_profile, base_pot_odds, after_tax_pot_odds
        )
        
        return {
            'action': recommended_action,
            'strategy': strategy,
            'confidence': max(strategy.values()),
            'hand_analysis': {
                'category': hand_profile.category.name,
                'strength_percentile': hand_profile.strength_percentile,
                'playability': hand_profile.playability_score,
                'set_mining_viable': hand_profile.set_mining_viable
            },
            'tax_impact': {
                'base_pot_odds': base_pot_odds,
                'after_tax_pot_odds': after_tax_pot_odds,
                'tax_rate': self.tax_profile.marginal_rate,
                'requires_tighter_play': after_tax_pot_odds > base_pot_odds * 1.1
            },
            'explanation': explanation,
            'advanced_factors': {
                'blocker_value': hand_profile.blocker_value,
                'position_value': self._get_position_value(situation_profile.position),
                'stack_depth_factor': self._get_stack_depth_factor(situation_profile.stack_depth),
                'pot_type': situation_profile.pot_type.value
            }
        }
    
    def _generate_explanation(self, action: str, strategy: Dict[str, float],
                            hand_profile: HandProfile, situation_profile: SituationProfile,
                            base_pot_odds: float, after_tax_pot_odds: float) -> str:
        """
        Generate natural language explanation for the decision
        """
        explanations = []
        
        # Hand strength explanation
        if hand_profile.strength_percentile > 80:
            explanations.append(f"Strong hand ({hand_profile.category.name})")
        elif hand_profile.strength_percentile > 50:
            explanations.append(f"Playable hand ({hand_profile.category.name})")
        else:
            explanations.append(f"Marginal hand ({hand_profile.category.name})")
        
        # Tax impact explanation
        if self.tax_profile.marginal_rate > 0:
            tax_impact = (after_tax_pot_odds - base_pot_odds) / base_pot_odds * 100 if base_pot_odds > 0 else 0
            if tax_impact > 20:
                explanations.append(f"Tax significantly impacts decision (+{tax_impact:.0f}% equity needed)")
            elif tax_impact > 10:
                explanations.append(f"Tax moderately impacts decision (+{tax_impact:.0f}% equity needed)")
        
        # Special situations
        if hand_profile.set_mining_viable and situation_profile.implied_odds > 7.5:
            explanations.append("Good set mining opportunity")
        
        if hand_profile.blocker_value > 0.8:
            explanations.append("Strong blockers favor aggression")
        
        # Action-specific explanation
        if action == "fold":
            explanations.append(f"Not profitable after tax (need {after_tax_pot_odds*100:.0f}% equity)")
        elif action == "call":
            if hand_profile.set_mining_viable:
                explanations.append("Calling to set mine")
            else:
                explanations.append("Pot odds justify calling")
        elif action == "raise":
            if hand_profile.strength_percentile > 80:
                explanations.append("Raising for value")
            else:
                explanations.append("Balanced raising range")
        
        return " | ".join(explanations)
    
    def _hand_to_bucket(self, hand_profile: HandProfile) -> int:
        """Map hand profile to CFR bucket"""
        # Simplified mapping - in production would be more sophisticated
        return min(14, int(hand_profile.strength_percentile / 7))
    
    def _position_to_key(self, position: str) -> int:
        """Convert position to numeric key"""
        positions = {'EP': 0, 'MP': 1, 'CO': 2, 'BTN': 3, 'SB': 4, 'BB': 5}
        return positions.get(position, 5)
    
    def _action_to_key(self, action: str) -> int:
        """Convert action to numeric key"""
        actions = {'unopened': 0, 'facing_raise': 1, 'facing_3bet': 2, 'facing_4bet': 3}
        return actions.get(action, 0)
    
    def _get_position_value(self, position: str) -> float:
        """Get position value (0-1)"""
        values = {'EP': 0.2, 'MP': 0.4, 'CO': 0.6, 'BTN': 1.0, 'SB': 0.3, 'BB': 0.5}
        return values.get(position, 0.5)
    
    def _get_stack_depth_factor(self, stack_depth: float) -> float:
        """Get stack depth factor for decision making"""
        if stack_depth < 20:
            return 0.3  # Push/fold
        elif stack_depth < 50:
            return 0.6  # Limited post-flop
        elif stack_depth < 100:
            return 0.8  # Standard
        else:
            return 1.0  # Deep stack
    
    def update_opponent_profile(self, villain_id: str, stats: PlayerStats):
        """Update opponent profile for exploitation"""
        villain_type = self.taxonomy.classify_player(stats)
        self.opponent_profiles[villain_id] = {
            'type': villain_type,
            'stats': stats
        }
        print(f"Updated profile for {villain_id}: {villain_type.value}")
    
    def get_session_report(self) -> Dict:
        """Get comprehensive session report"""
        tax_report = self.tax_system.generate_tax_report(
            self.session_history, self.tax_profile
        )
        
        return {
            'hands_played': self.hands_played,
            'session_profit': self.session_profit,
            'tax_impact': tax_report,
            'effective_hourly': self.session_profit * (1 - self.tax_profile.marginal_rate),
            'recommendation': self._get_session_recommendation()
        }
    
    def _get_session_recommendation(self) -> str:
        """Generate session recommendation"""
        if self.session_profit > 0:
            after_tax = self.session_profit * (1 - self.tax_profile.marginal_rate)
            return f"Book win! After-tax profit: ${after_tax:.2f}. Consider tax withholding."
        else:
            if self.tax_profile.loss_deductible:
                return f"Loss of ${abs(self.session_profit):.2f} is tax deductible"
            else:
                return f"Loss of ${abs(self.session_profit):.2f} - not deductible (recreational player)"


# Demo function
def demo_ultimate_player():
    """Demonstrate the ultimate integrated player"""
    print("\n" + "🎰"*30)
    print("ULTIMATE INTEGRATED GTO PLAYER DEMO")
    print("CFR + Taxonomy + Taxation = Optimal Decisions")
    print("🎰"*30)
    
    # Initialize players with different tax profiles
    players = {
        'UK Pro': UltimateGTOPlayer(
            tax_jurisdiction=TaxJurisdiction.UK,
            player_status=PlayerStatus.PROFESSIONAL
        ),
        'US Pro': UltimateGTOPlayer(
            tax_jurisdiction=TaxJurisdiction.USA_PROFESSIONAL,
            player_status=PlayerStatus.PROFESSIONAL
        ),
        'US Rec': UltimateGTOPlayer(
            tax_jurisdiction=TaxJurisdiction.USA_RECREATIONAL,
            player_status=PlayerStatus.RECREATIONAL
        )
    }
    
    # Test scenario: 77 in BB facing BTN raise
    print("\n" + "="*60)
    print("SCENARIO: 77 in BB facing 3BB raise from BTN, 100BB deep")
    print("="*60)
    
    game_state = {
        'hole_cards': ['7s', '7h'],
        'position': 'BB',
        'stack_depth': 100,
        'pot_size': 6.5,
        'bet_size': 3.0,
        'facing_action': 'facing_raise',
        'num_players': 2
    }
    
    for player_name, player in players.items():
        decision = player.make_decision(game_state)
        
        print(f"\n{player_name} (Tax: {player.tax_profile.marginal_rate*100:.0f}%):")
        print(f"  Action: {decision['action'].upper()}")
        print(f"  Strategy: ", end="")
        for action, prob in decision['strategy'].items():
            print(f"{action}={prob*100:.0f}% ", end="")
        print()
        print(f"  Confidence: {decision['confidence']*100:.0f}%")
        print(f"  Tax Impact: ", end="")
        if decision['tax_impact']['requires_tighter_play']:
            print(f"Need {decision['tax_impact']['after_tax_pot_odds']*100:.0f}% equity (vs {decision['tax_impact']['base_pot_odds']*100:.0f}% base)")
        else:
            print("No significant impact")
        print(f"  Explanation: {decision['explanation']}")
    
    # Test scenario 2: Flush draw
    print("\n" + "="*60)
    print("SCENARIO: Flush draw on flop facing pot bet")
    print("="*60)
    
    game_state = {
        'hole_cards': ['As', '5s'],
        'board': ['Ks', '9s', '2h'],
        'position': 'BB',
        'stack_depth': 100,
        'pot_size': 12.0,
        'bet_size': 12.0,
        'facing_action': 'facing_raise',
        'num_players': 2
    }
    
    for player_name, player in players.items():
        decision = player.make_decision(game_state)
        
        print(f"\n{player_name}:")
        print(f"  Action: {decision['action'].upper()}")
        print(f"  Tax-adjusted pot odds: {decision['tax_impact']['after_tax_pot_odds']*100:.0f}%")
        print(f"  Decision: {decision['explanation']}")
    
    # Test with opponent profile
    print("\n" + "="*60)
    print("SCENARIO: AK vs NIT player")
    print("="*60)
    
    # Add a NIT opponent
    nit_stats = PlayerStats(
        vpip=12, pfr=10, three_bet=3, fold_to_three_bet=85,
        c_bet=45, fold_to_c_bet=75, aggression_factor=1.5,
        went_to_showdown=18, won_at_showdown=65,
        four_bet_range=2, squeeze_frequency=2,
        donk_bet_frequency=5, check_raise_frequency=8,
        river_bluff_frequency=10
    )
    
    uk_player = players['UK Pro']
    uk_player.update_opponent_profile('villain1', nit_stats)
    
    game_state = {
        'hole_cards': ['As', 'Kh'],
        'position': 'BTN',
        'stack_depth': 100,
        'pot_size': 7.5,
        'bet_size': 7.5,
        'facing_action': 'facing_3bet',
        'num_players': 2,
        'villain_id': 'villain1'
    }
    
    decision = uk_player.make_decision(game_state)
    
    print(f"\nUK Pro vs NIT:")
    print(f"  Hand: AKo facing 3bet from NIT")
    print(f"  Action: {decision['action'].upper()}")
    print(f"  Explanation: {decision['explanation']}")
    print(f"  Note: Strategy adjusted for tight opponent")


if __name__ == "__main__":
    demo_ultimate_player()

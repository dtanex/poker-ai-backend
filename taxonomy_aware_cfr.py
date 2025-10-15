"""
Taxonomy-Aware CFR Trainer
Integrates poker taxonomy deeply into the CFR training algorithm
This produces strategies based on hand properties, not just strength
"""
import numpy as np
import pickle
import time
from typing import List, Dict, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import json

from poker_taxonomy import (
    PokerTaxonomyEngine, 
    HandProfile, 
    HandCategory,
    SituationProfile,
    BoardTexture,
    PotType,
    GameStage
)

@dataclass
class TaxonomicState:
    """
    Represents a game state with full taxonomic context
    This replaces simple bucket-based state representation
    """
    hand_category: HandCategory
    playability_score: float
    set_mining_viable: bool
    blocker_value: float
    suited_connector_value: float
    position: str
    board_texture: Optional[BoardTexture]
    pot_type: PotType
    stack_depth: float
    action_sequence: List[str]
    
    def get_state_id(self) -> str:
        """Generate unique ID for this taxonomic state"""
        # Create a comprehensive state identifier that captures taxonomy
        components = [
            self.hand_category.name,
            f"play{int(self.playability_score/20)*20}",  # Round to nearest 20
            f"set{self.set_mining_viable}",
            f"block{int(self.blocker_value*10)}",
            f"sc{int(self.suited_connector_value*10)}",
            self.position,
            self.board_texture.value if self.board_texture else "preflop",
            self.pot_type.value,
            f"stack{int(self.stack_depth/25)*25}",  # Round to nearest 25BB
            "_".join(self.action_sequence[-3:])  # Last 3 actions
        ]
        return "|".join(components)

class TaxonomyAwareCFRTrainer:
    """
    CFR Trainer that uses poker taxonomy at every level of training
    Produces strategies based on hand properties, not just strength
    """
    
    def __init__(self):
        # Initialize taxonomy engine
        self.taxonomy_engine = PokerTaxonomyEngine()
        
        # Taxonomic strategy tables
        # Key: TaxonomicState.get_state_id() -> Value: strategy/regrets
        self.taxonomic_regrets = defaultdict(lambda: np.zeros(3))  # fold, call, raise
        self.taxonomic_strategies = defaultdict(lambda: np.ones(3) / 3)
        self.taxonomic_strategy_sum = defaultdict(lambda: np.zeros(3))
        
        # Coaching metadata for each state
        self.coaching_notes = defaultdict(list)
        
        # Training parameters
        self.iteration = 0
        self.exploration_factor = 0.1
        
        print("🧬 Taxonomy-Aware CFR Trainer Initialized")
        print("   Uses hand categories, not strength buckets")
        print("   Incorporates playability and special properties")
        print("   Produces coaching-friendly strategies")
    
    def train_with_taxonomy(self, num_iterations: int = 100000):
        """
        Train strategies using full taxonomic understanding
        """
        print(f"\n🧬 Training Taxonomy-Aware Strategies ({num_iterations} iterations)")
        print("   This produces strategies based on hand properties, not just strength")
        
        start_time = time.time()
        
        # Generate training scenarios
        training_hands = self._generate_training_hands()
        training_situations = self._generate_training_situations()
        
        for iteration in range(num_iterations):
            self.iteration = iteration
            
            # Reduce exploration over time
            self.exploration_factor = max(0.01, 0.2 * (1 - iteration / num_iterations))
            
            # Train each hand in each situation
            for hole_cards in training_hands:
                hand_profile = self.taxonomy_engine.classify_hand(hole_cards)
                
                for situation in training_situations:
                    # Create taxonomic state
                    state = self._create_taxonomic_state(hand_profile, situation)
                    
                    # Get position-aware starting strategy
                    initial_strategy = self._get_position_aware_initial_strategy(
                        hand_profile, state.position
                    )
                    
                    # Update strategy based on regrets
                    strategy = self._update_strategy_from_regrets(
                        state, initial_strategy
                    )
                    
                    # Calculate taxonomically-adjusted utilities
                    utilities = self._calculate_taxonomic_utilities(state, hand_profile)
                    
                    # Sample action and calculate regret
                    action_probs = strategy + self.exploration_factor * np.random.dirichlet([1,1,1])
                    action_probs /= action_probs.sum()
                    action = np.random.choice(3, p=action_probs)
                    
                    # Calculate regrets with taxonomic adjustments
                    action_utility = utilities[action]
                    regrets = utilities - action_utility
                    
                    # Apply taxonomic weighting to regrets
                    weighted_regrets = self._apply_taxonomic_weighting(
                        regrets, state, hand_profile
                    )
                    
                    # Update cumulative regrets
                    state_id = state.get_state_id()
                    self.taxonomic_regrets[state_id] += weighted_regrets
                    self.taxonomic_strategy_sum[state_id] += strategy
                    self.taxonomic_strategies[state_id] = strategy
                    
                    # Generate coaching notes
                    if iteration % 10000 == 0:
                        self._generate_coaching_notes(state, hand_profile, strategy)
            
            # Progress updates
            if (iteration + 1) % 10000 == 0:
                elapsed = time.time() - start_time
                print(f"   Progress: {iteration + 1}/{num_iterations} ({elapsed:.1f}s)")
                self._print_sample_strategies()
        
        print(f"\n✅ Taxonomy-aware training complete in {time.time() - start_time:.1f}s")
    
    def _generate_training_hands(self) -> List[List[str]]:
        """Generate representative training hands from each category"""
        training_hands = []
        
        # Premium pairs
        training_hands.extend([
            ['As', 'Ah'], ['Ks', 'Kh'], ['Qs', 'Qh']
        ])
        
        # Medium pairs (set mining candidates)
        training_hands.extend([
            ['7s', '7h'], ['8s', '8h'], ['9s', '9h']
        ])
        
        # Small pairs
        training_hands.extend([
            ['2s', '2h'], ['3s', '3h'], ['4s', '4h']
        ])
        
        # Premium broadway
        training_hands.extend([
            ['As', 'Kh'], ['As', 'Ks'], ['As', 'Qh']
        ])
        
        # Suited connectors
        training_hands.extend([
            ['8s', '9s'], ['7s', '8s'], ['9s', 'Ts'], ['Js', 'Ts']
        ])
        
        # Suited gappers
        training_hands.extend([
            ['Js', '9s'], ['Ts', '8s'], ['9s', '7s']
        ])
        
        # Suited aces
        training_hands.extend([
            ['As', '5s'], ['As', '4s'], ['As', '3s'], ['As', '2s']
        ])
        
        # Offsuit broadways
        training_hands.extend([
            ['Kh', 'Qd'], ['Qh', 'Jd'], ['Jh', 'Td']
        ])
        
        # Trash hands
        training_hands.extend([
            ['7s', '2h'], ['8s', '3h'], ['9s', '4h']
        ])
        
        return training_hands
    
    def _generate_training_situations(self) -> List[Dict]:
        """Generate diverse training situations"""
        situations = []
        
        positions = ['EP', 'MP', 'CO', 'BTN', 'SB', 'BB']
        stack_depths = [30, 50, 100, 200]
        action_sequences = [
            [],  # Unopened
            ['open'],  # Facing open
            ['open', '3bet'],  # Facing 3bet
            ['open', '3bet', '4bet'],  # Facing 4bet
        ]
        pot_types = [PotType.HEADS_UP, PotType.MULTIWAY]
        
        for position in positions:
            for stack_depth in stack_depths:
                for action_seq in action_sequences:
                    for pot_type in pot_types:
                        situations.append({
                            'position': position,
                            'stack_depth': stack_depth,
                            'action_sequence': action_seq,
                            'pot_type': pot_type
                        })
        
        return situations
    
    def _create_taxonomic_state(self, hand_profile: HandProfile, 
                                situation: Dict) -> TaxonomicState:
        """Create a complete taxonomic state"""
        return TaxonomicState(
            hand_category=hand_profile.category,
            playability_score=hand_profile.playability_score,
            set_mining_viable=hand_profile.set_mining_viable,
            blocker_value=hand_profile.blocker_value,
            suited_connector_value=hand_profile.suited_connector_value,
            position=situation['position'],
            board_texture=None,  # Preflop for now
            pot_type=situation['pot_type'],
            stack_depth=situation['stack_depth'],
            action_sequence=situation['action_sequence']
        )
    
    def _get_position_aware_initial_strategy(self, hand_profile: HandProfile, 
                                            position: str) -> np.ndarray:
        """
        Get position-aware starting strategy based on taxonomy
        This seeds CFR with reasonable starting points
        """
        # Get position-specific VPIP from taxonomy
        vpip = getattr(hand_profile, f'{position.lower()}_vpip', 25) / 100
        
        # Convert VPIP to initial strategy
        if len(hand_profile.cards) == 0:  # No action sequence
            # RFI (raise first in) scenario
            if hand_profile.rfi_range:
                # Strong hand in RFI range
                return np.array([0.0, 0.05, 0.95])  # Mostly raise
            else:
                # Weak hand not in RFI range
                return np.array([0.95, 0.05, 0.0])  # Mostly fold
        else:
            # Facing action
            if hand_profile.three_bet_range:
                return np.array([0.1, 0.3, 0.6])  # Mix of call and 3bet
            elif hand_profile.calling_range:
                return np.array([0.2, 0.7, 0.1])  # Mostly call
            else:
                return np.array([0.8, 0.15, 0.05])  # Mostly fold
    
    def _calculate_taxonomic_utilities(self, state: TaxonomicState, 
                                      hand_profile: HandProfile) -> np.ndarray:
        """
        Calculate utilities with full taxonomic adjustments
        This is the CORE of taxonomy-aware training
        """
        # Base utilities
        fold_utility = 0.0
        call_utility = 0.0
        raise_utility = 0.0
        
        # Playability adjustment - hands with higher playability have better continuation
        playability_factor = hand_profile.playability_score / 100
        
        # Set mining adjustment
        if state.set_mining_viable:
            if state.stack_depth >= 50 and len(state.action_sequence) <= 1:
                # Deep enough for set mining, facing single raise or less
                implied_odds = state.stack_depth / 7.5  # Need 7.5:1
                if implied_odds >= 7.5:
                    call_utility += 0.4 * playability_factor
                    raise_utility -= 0.2  # Don't blow up pot with small pairs
                else:
                    call_utility -= 0.2  # Not deep enough
            else:
                # Not ideal for set mining
                call_utility -= 0.3
        
        # Blocker adjustment
        if state.blocker_value > 0.7:
            # High blocker value increases bluffing utility
            if len(state.action_sequence) >= 2:  # In 3bet+ pot
                raise_utility += 0.3 * state.blocker_value
            else:
                raise_utility += 0.15 * state.blocker_value
        
        # Suited connector adjustment
        if state.suited_connector_value > 0.7:
            if state.pot_type == PotType.MULTIWAY:
                # Suited connectors love multiway pots
                call_utility += 0.3 * state.suited_connector_value
                raise_utility -= 0.1  # Don't isolate with SCs
            elif state.stack_depth >= 100:
                # Deep stacks favor SCs
                call_utility += 0.2 * state.suited_connector_value
        
        # Position adjustment
        position_value = {
            'BTN': 0.3, 'CO': 0.2, 'MP': 0.1, 
            'EP': 0.0, 'SB': -0.1, 'BB': -0.05
        }.get(state.position, 0)
        
        raise_utility += position_value * playability_factor
        
        # Stack depth adjustments
        if state.stack_depth < 30:
            # Short stack - reduce speculative play
            if not hand_profile.category in [HandCategory.PREMIUM_PAIR, 
                                            HandCategory.PREMIUM_BROADWAY]:
                call_utility -= 0.3
                raise_utility = max(raise_utility, fold_utility)  # Push or fold
        elif state.stack_depth > 150:
            # Deep stack - increase speculative play
            if state.suited_connector_value > 0 or state.set_mining_viable:
                call_utility += 0.2
        
        # Hand category specific adjustments
        if hand_profile.category == HandCategory.PREMIUM_PAIR:
            raise_utility += 0.6
            call_utility -= 0.2  # Don't trap too much
        elif hand_profile.category == HandCategory.SUITED_CONNECTOR:
            call_utility += 0.2
            raise_utility += 0.1 * position_value  # Position-dependent aggression
        elif hand_profile.category == HandCategory.TRASH:
            fold_utility = 0.1  # Almost always fold
            call_utility = -0.5
            raise_utility = -0.3
        
        # Action sequence adjustments
        if '4bet' in state.action_sequence:
            # Only premiums continue vs 4bet
            if hand_profile.category not in [HandCategory.PREMIUM_PAIR,
                                            HandCategory.PREMIUM_BROADWAY]:
                fold_utility = 0.2
                call_utility = -0.8
                raise_utility = -0.9
        elif '3bet' in state.action_sequence:
            # Tighten up vs 3bet
            if hand_profile.strength_percentile < 70:
                fold_utility = 0.1
                call_utility -= 0.3
                raise_utility -= 0.4
        
        # Add controlled randomness for exploration
        noise = np.random.normal(0, self.exploration_factor, 3)
        
        utilities = np.array([fold_utility, call_utility, raise_utility]) + noise
        
        return utilities
    
    def _apply_taxonomic_weighting(self, regrets: np.ndarray, 
                                   state: TaxonomicState,
                                   hand_profile: HandProfile) -> np.ndarray:
        """
        Weight regrets based on taxonomic importance
        Some decisions matter more for certain hand types
        """
        weights = np.ones(3)
        
        # Set mining hands care more about calling decisions
        if state.set_mining_viable:
            weights[1] *= 1.5  # Call decision is critical
            weights[2] *= 0.8  # Raise decision less important
        
        # Suited connectors care about position
        if state.suited_connector_value > 0.7:
            if state.position in ['BTN', 'CO']:
                weights[2] *= 1.3  # Raising matters more in position
            else:
                weights[1] *= 1.3  # Calling matters more out of position
        
        # Premium hands care more about raising
        if hand_profile.category in [HandCategory.PREMIUM_PAIR, 
                                    HandCategory.PREMIUM_BROADWAY]:
            weights[2] *= 1.5  # Raise decision critical
            weights[0] *= 0.5  # Fold decision almost irrelevant
        
        # Trash hands learn faster to fold
        if hand_profile.category == HandCategory.TRASH:
            weights[0] *= 2.0  # Learn to fold quickly
        
        return regrets * weights
    
    def _update_strategy_from_regrets(self, state: TaxonomicState, 
                                     initial_strategy: np.ndarray) -> np.ndarray:
        """Update strategy using regret matching with taxonomic considerations"""
        state_id = state.get_state_id()
        regrets = self.taxonomic_regrets[state_id]
        
        # Use positive regrets
        positive_regrets = np.maximum(regrets, 0)
        sum_regrets = np.sum(positive_regrets)
        
        if sum_regrets > 0:
            strategy = positive_regrets / sum_regrets
        else:
            # Use taxonomically-informed default
            strategy = initial_strategy
        
        # Apply minimum frequency constraints based on taxonomy
        if state.set_mining_viable and state.stack_depth >= 50:
            # Ensure minimum calling frequency for set mining
            strategy[1] = max(strategy[1], 0.15)
        
        if state.hand_category == HandCategory.PREMIUM_PAIR:
            # Ensure minimum raising frequency for premiums
            strategy[2] = max(strategy[2], 0.5)
        
        # Normalize
        strategy = strategy / strategy.sum()
        
        return strategy
    
    def _generate_coaching_notes(self, state: TaxonomicState, 
                                hand_profile: HandProfile,
                                strategy: np.ndarray):
        """Generate human-readable coaching notes for this decision"""
        state_id = state.get_state_id()
        notes = []
        
        # Analyze the strategy
        fold_pct, call_pct, raise_pct = strategy * 100
        
        # Set mining note
        if state.set_mining_viable and call_pct > 40:
            notes.append(f"Set mining viable with {state.stack_depth}BB stack")
        
        # Position note
        if state.position == 'BTN' and raise_pct > 60:
            notes.append("Aggressive button play with position")
        elif state.position == 'EP' and fold_pct > 70:
            notes.append("Tight early position, waiting for premiums")
        
        # Suited connector note
        if state.suited_connector_value > 0.7:
            if state.pot_type == PotType.MULTIWAY:
                notes.append("Suited connector plays well multiway")
            else:
                notes.append("Be cautious with suited connector heads-up")
        
        # Stack depth note
        if state.stack_depth < 30:
            notes.append("Short stack: push or fold mode")
        elif state.stack_depth > 150:
            notes.append("Deep stack: implied odds for speculative hands")
        
        # Action sequence note
        if '3bet' in state.action_sequence:
            if call_pct > raise_pct:
                notes.append("Calling 3bet to see flop in position")
            elif raise_pct > 30:
                notes.append("4betting for value or as bluff")
        
        self.coaching_notes[state_id] = notes
    
    def _print_sample_strategies(self):
        """Print sample strategies with taxonomic context"""
        print("\n📊 Sample Taxonomy-Aware Strategies:")
        
        # Example: 77 in BB facing raise, 100BB deep
        example_state = TaxonomicState(
            hand_category=HandCategory.MEDIUM_PAIR,
            playability_score=65,
            set_mining_viable=True,
            blocker_value=0.4,
            suited_connector_value=0,
            position='BB',
            board_texture=None,
            pot_type=PotType.HEADS_UP,
            stack_depth=100,
            action_sequence=['open']
        )
        
        state_id = example_state.get_state_id()
        if state_id in self.taxonomic_strategies:
            strategy = self.taxonomic_strategies[state_id]
            notes = self.coaching_notes.get(state_id, [])
            
            print(f"\n77 in BB facing raise (100BB):")
            print(f"  Fold: {strategy[0]*100:.1f}%")
            print(f"  Call: {strategy[1]*100:.1f}% ← Set mining")
            print(f"  Raise: {strategy[2]*100:.1f}%")
            if notes:
                print(f"  Notes: {'; '.join(notes)}")
    
    def get_strategy_with_coaching(self, hole_cards: List[str], 
                                   position: str,
                                   stack_depth: float,
                                   action_sequence: List[str],
                                   pot_type: str = 'heads_up') -> Dict:
        """
        Get strategy with full taxonomic context and coaching notes
        """
        # Classify hand
        hand_profile = self.taxonomy_engine.classify_hand(hole_cards)
        
        # Create state
        state = TaxonomicState(
            hand_category=hand_profile.category,
            playability_score=hand_profile.playability_score,
            set_mining_viable=hand_profile.set_mining_viable,
            blocker_value=hand_profile.blocker_value,
            suited_connector_value=hand_profile.suited_connector_value,
            position=position,
            board_texture=None,
            pot_type=PotType[pot_type.upper()],
            stack_depth=stack_depth,
            action_sequence=action_sequence
        )
        
        state_id = state.get_state_id()
        
        # Get strategy
        if state_id in self.taxonomic_strategy_sum:
            strategy_sum = self.taxonomic_strategy_sum[state_id]
            total = strategy_sum.sum()
            if total > 0:
                strategy = strategy_sum / total
            else:
                strategy = np.array([0.33, 0.33, 0.34])
        else:
            # Use taxonomically-informed default
            strategy = self._get_position_aware_initial_strategy(hand_profile, position)
        
        # Get coaching notes
        notes = self.coaching_notes.get(state_id, [])
        
        # Auto-generate notes if none exist
        if not notes:
            if hand_profile.set_mining_viable and strategy[1] > 0.5:
                notes.append(f"Set mining opportunity with {stack_depth}BB")
            if hand_profile.suited_connector_value > 0.7:
                notes.append("Suited connector - position and stack depth matter")
            if hand_profile.category == HandCategory.PREMIUM_PAIR:
                notes.append("Premium pair - usually raise for value")
        
        return {
            'hand': hole_cards,
            'category': hand_profile.category.name,
            'playability': hand_profile.playability_score,
            'set_mining_viable': hand_profile.set_mining_viable,
            'optimal_actions': {
                'fold': strategy[0],
                'call': strategy[1],
                'raise': strategy[2]
            },
            'recommended_notes': notes,
            'taxonomic_factors': {
                'blocker_value': hand_profile.blocker_value,
                'suited_connector_value': hand_profile.suited_connector_value,
                'position_vpip': getattr(hand_profile, f'{position.lower()}_vpip', 0)
            }
        }
    
    def save_taxonomic_strategies(self, filename: str = "taxonomic_strategies.pkl"):
        """Save taxonomy-aware strategies with metadata"""
        save_data = {
            'strategies': dict(self.taxonomic_strategy_sum),
            'coaching_notes': dict(self.coaching_notes),
            'metadata': {
                'iterations': self.iteration,
                'taxonomy_version': '2.0',
                'features': [
                    'hand_category_aware',
                    'set_mining_optimization',
                    'suited_connector_handling',
                    'blocker_value_integration',
                    'position_specific_training'
                ]
            }
        }
        
        with open(filename, 'wb') as f:
            pickle.dump(save_data, f)
        
        # Also save human-readable version
        json_filename = filename.replace('.pkl', '_readable.json')
        
        # Convert sample strategies to readable format
        readable_samples = {}
        for state_id in list(self.taxonomic_strategy_sum.keys())[:10]:
            strategy_sum = self.taxonomic_strategy_sum[state_id]
            total = strategy_sum.sum()
            if total > 0:
                strategy = strategy_sum / total
                readable_samples[state_id] = {
                    'fold': float(strategy[0]),
                    'call': float(strategy[1]),
                    'raise': float(strategy[2]),
                    'notes': self.coaching_notes.get(state_id, [])
                }
        
        with open(json_filename, 'w') as f:
            json.dump({
                'sample_strategies': readable_samples,
                'metadata': save_data['metadata']
            }, f, indent=2)
        
        print(f"\n💾 Taxonomic strategies saved to {filename}")
        print(f"   Human-readable samples in {json_filename}")
        
        import os
        size = os.path.getsize(filename)
        print(f"   File size: {size / 1024:.1f} KB")


# Demo function
def demo_taxonomic_training():
    """Demonstrate taxonomy-aware CFR training"""
    print("\n" + "🧬"*30)
    print("TAXONOMY-AWARE CFR TRAINING DEMO")
    print("🧬"*30)
    
    trainer = TaxonomyAwareCFRTrainer()
    
    # Quick training for demo
    print("\n🎯 Training with taxonomy integration...")
    trainer.train_with_taxonomy(num_iterations=1000)
    
    # Test some hands
    test_cases = [
        (['7s', '7h'], 'BB', 100, ['open'], "77 - Set Mining Hand"),
        (['8s', '9s'], 'BTN', 100, [], "89s - Suited Connector"),
        (['As', 'Kh'], 'CO', 100, [], "AKo - Premium Broadway"),
        (['7s', '2h'], 'EP', 50, [], "72o - Trash Hand"),
    ]
    
    print("\n" + "="*60)
    print("TAXONOMY-AWARE STRATEGIES")
    print("="*60)
    
    for cards, position, stack, action_seq, description in test_cases:
        result = trainer.get_strategy_with_coaching(
            cards, position, stack, action_seq
        )
        
        print(f"\n📊 {description}")
        print(f"   Position: {position}, Stack: {stack}BB")
        print(f"   Category: {result['category']}")
        print(f"   Playability: {result['playability']}/100")
        
        print("\n   Strategy:")
        for action, prob in result['optimal_actions'].items():
            bar = "█" * int(prob * 20)
            print(f"   {action:5s}: {bar:<20s} {prob*100:>5.1f}%")
        
        if result['recommended_notes']:
            print("\n   Coaching Notes:")
            for note in result['recommended_notes']:
                print(f"   • {note}")
        
        print("\n   Taxonomic Factors:")
        for factor, value in result['taxonomic_factors'].items():
            print(f"   • {factor}: {value:.2f}")
    
    # Save strategies
    trainer.save_taxonomic_strategies()
    
    print("\n" + "="*60)
    print("✅ Taxonomy-aware training produces strategies that understand:")
    print("   • Set mining viability and stack requirements")
    print("   • Suited connector playability by position")
    print("   • Blocker effects for bluffing")
    print("   • Position-specific opening ranges")
    print("   • Stack depth adaptations")
    print("\nThis is TRUE poker understanding, not just math!")
    print("="*60)


if __name__ == "__main__":
    demo_taxonomic_training()

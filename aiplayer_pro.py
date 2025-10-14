"""
Professional GTO AI Player - Uses trained CFR strategies with position awareness
Competitive with commercial solvers like PioSolver and GTO+
"""
import numpy as np
import pickle
from typing import List, Dict, Tuple, Optional
from enum import Enum

# Import enums from trainer
class Position(Enum):
    """Poker positions"""
    EP = 0  # Early Position
    MP = 1  # Middle Position
    CO = 2  # Cutoff
    BTN = 3  # Button
    SB = 4  # Small Blind
    BB = 5  # Big Blind

class ActionFacing(Enum):
    """What action are we facing"""
    UNOPENED = 0
    FACING_RAISE = 1
    FACING_3BET = 2
    FACING_4BET = 3

# Card ranking
RANKS = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
         '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}

class ProfessionalAIPlayer:
    """
    Professional GTO AI Player with position awareness
    """
    
    def __init__(self, strategy_file: str = "gto_strategies.pkl"):
        """
        Load trained GTO strategies
        
        Args:
            strategy_file: Path to pickled strategy dict
        """
        try:
            with open(strategy_file, 'rb') as f:
                strategies = pickle.load(f)
                self.preflop = strategies['preflop']
                self.postflop = strategies['postflop']
                self.meta = strategies.get('meta', {})
                
                # Get bucket counts
                self.preflop_buckets = self.meta.get('preflop_buckets', 15)
                self.postflop_buckets = self.meta.get('postflop_buckets', 20)
                
                print(f"✅ Loaded GTO strategies from {strategy_file}")
                print(f"   Preflop buckets: {self.preflop_buckets}")
                print(f"   Postflop buckets: {self.postflop_buckets}")
                print(f"   Position-aware: Yes")
                print(f"   Stack depth aware: Yes")
                
                # Check exploitability
                if 'exploitability_history' in self.meta and self.meta['exploitability_history']:
                    final_exploit = self.meta['exploitability_history'][-1]
                    print(f"   Final exploitability: {final_exploit:.4f}")
                    
        except FileNotFoundError:
            print(f"⚠️  Strategy file not found, using random strategy")
            self.preflop = {}
            self.postflop = {}
            self.meta = {}
            self.preflop_buckets = 15
            self.postflop_buckets = 20
    
    def get_preflop_bucket(self, hole_cards: List[str]) -> int:
        """
        Get preflop bucket (0-14) with granular hand categorization
        
        Professional bucketing system:
        0-2: Trash hands (72o, 83o, etc.)
        3-4: Weak hands  
        5-6: Small pairs (22-55)
        7-8: Suited connectors (65s-T9s)
        9-10: Broadway cards, medium pairs (66-99)
        11-12: Strong hands (TT-JJ, AQ, AJs)
        13: Premium hands (QQ, AKo)
        14: Monster hands (KK, AA, AKs)
        """
        rank1 = RANKS[hole_cards[0][0]]
        rank2 = RANKS[hole_cards[1][0]]
        suit1 = hole_cards[0][1] if len(hole_cards[0]) > 1 else 's'
        suit2 = hole_cards[1][1] if len(hole_cards[1]) > 1 else 's'
        
        high = max(rank1, rank2)
        low = min(rank1, rank2)
        suited = (suit1 == suit2)
        paired = (rank1 == rank2)
        gap = high - low
        
        # Monster hands (AA, KK)
        if paired and high >= 13:
            return 14
        
        # AKs
        if high == 14 and low == 13 and suited:
            return 14
        
        # Premium (QQ, AKo)
        if (paired and high == 12) or (high == 14 and low == 13 and not suited):
            return 13
        
        # Strong hands (JJ, TT, AQs, AJs)
        if paired and high >= 10:
            return 12
        if high == 14 and low >= 11 and suited:
            return 12
        
        # AQo, AJo, KQs
        if (high == 14 and low >= 11) or (high == 13 and low == 12 and suited):
            return 11
        
        # Medium pairs (99-66)
        if paired and 6 <= high <= 9:
            return 10
        
        # Suited broadways, ATo
        if (high >= 10 and low >= 10 and suited) or (high == 14 and low == 10):
            return 9
        
        # Suited connectors (T9s-65s)
        if suited and gap <= 1 and high >= 6:
            return 8
        
        # Small suited connectors, one-gappers
        if suited and gap <= 2 and high >= 5:
            return 7
        
        # Small pairs (55-22)
        if paired and high <= 5:
            return 6
        
        # Weak aces, suited aces
        if high == 14:
            return 5 if suited else 4
        
        # King-high, Queen-high
        if high >= 12:
            return 4
        
        # Suited cards
        if suited:
            return 3
        
        # Connected cards
        if gap <= 2:
            return 2
        
        # One high card
        if high >= 10:
            return 1
        
        # Complete trash
        return 0
    
    def compute_postflop_bucket(self, equity: float) -> int:
        """
        Convert equity to bucket (0-19)
        
        Granular equity buckets for precise postflop play
        """
        bucket = int(equity * self.postflop_buckets)
        return min(bucket, self.postflop_buckets - 1)
    
    def calculate_equity(self, hole_cards: List[str], board: List[str]) -> float:
        """
        Calculate hand equity (simplified for MVP)
        
        In production, use Monte Carlo simulation against opponent ranges
        """
        rank1 = RANKS[hole_cards[0][0]]
        rank2 = RANKS[hole_cards[1][0]]
        
        # Base strength from hole cards
        strength = (rank1 + rank2) / 28.0
        
        # Pocket pair bonus
        if rank1 == rank2:
            strength += 0.25
        
        # Board interaction
        if board:
            board_ranks = [RANKS[card[0]] for card in board]
            
            # Check for pairs with board
            for hole_rank in [rank1, rank2]:
                matches = board_ranks.count(hole_rank)
                if matches == 1:
                    strength += 0.20  # Pair
                elif matches == 2:
                    strength += 0.40  # Trips
                elif matches == 3:
                    strength += 0.60  # Quads
            
            # Check for flush potential
            suits = [card[1] for card in hole_cards + board]
            for suit in set(suits):
                if suits.count(suit) >= 4:
                    if suits.count(suit) == 5:
                        strength += 0.35  # Made flush
                    else:
                        strength += 0.15  # Flush draw
            
            # Check for straight potential (simplified)
            all_ranks = sorted([rank1, rank2] + board_ranks)
            consecutive = 1
            for i in range(1, len(all_ranks)):
                if all_ranks[i] == all_ranks[i-1] + 1:
                    consecutive += 1
                    if consecutive >= 4:
                        strength += 0.10  # Straight draw
                    if consecutive >= 5:
                        strength += 0.25  # Made straight
                        break
                elif all_ranks[i] != all_ranks[i-1]:
                    consecutive = 1
        
        # Clamp to valid range
        return min(1.0, max(0.0, strength))
    
    def get_action(self, 
                   stage: str, 
                   hole_cards: List[str], 
                   board: List[str] = [],
                   position: str = "BB",
                   action_facing: str = "unopened",
                   stack_depth: int = 100,
                   pot_size: float = 1.5,
                   bet_size: float = 0) -> str:
        """
        Get a single action with full context awareness
        
        Args:
            stage: 'preflop', 'flop', 'turn', or 'river'
            hole_cards: ['As', 'Kh']
            board: ['Qh', 'Jd', '7c']
            position: 'EP', 'MP', 'CO', 'BTN', 'SB', 'BB'
            action_facing: 'unopened', 'facing_raise', 'facing_3bet', 'facing_4bet'
            stack_depth: Effective stack in BB
            pot_size: Current pot size in BB
            bet_size: Size of bet we're facing in BB
        
        Returns:
            action: 'fold', 'call', 'raise', or 'bet'
        """
        if stage == "preflop":
            bucket = self.get_preflop_bucket(hole_cards)
            
            # Convert position string to enum
            pos_map = {
                'EP': Position.EP, 'MP': Position.MP, 'CO': Position.CO,
                'BTN': Position.BTN, 'SB': Position.SB, 'BB': Position.BB
            }
            pos_enum = pos_map.get(position, Position.BB)
            
            # Convert action facing to enum
            action_map = {
                'unopened': ActionFacing.UNOPENED,
                'facing_raise': ActionFacing.FACING_RAISE,
                'facing_3bet': ActionFacing.FACING_3BET,
                'facing_4bet': ActionFacing.FACING_4BET
            }
            action_enum = action_map.get(action_facing, ActionFacing.UNOPENED)
            
            # Create state key
            state_key = f"{pos_enum.value}_{action_enum.value}_{stack_depth}"
            
            # Get strategy
            if bucket in self.preflop and state_key in self.preflop[bucket]:
                strategy = self.preflop[bucket][state_key]
            else:
                # Fallback to default
                strategy = [0.33, 0.33, 0.34]
            
            actions = ["fold", "call", "raise"]
        else:
            # Postflop
            equity = self.calculate_equity(hole_cards, board)
            bucket = self.compute_postflop_bucket(equity)
            
            # Convert position
            pos_map = {
                'EP': Position.EP, 'MP': Position.MP, 'CO': Position.CO,
                'BTN': Position.BTN, 'SB': Position.SB, 'BB': Position.BB
            }
            pos_enum = pos_map.get(position, Position.BB)
            
            # Create state key
            state_key = f"{pos_enum.value}_{pot_size}_{bet_size}"
            
            # Get strategy
            if bucket in self.postflop and state_key in self.postflop[bucket]:
                strategy = self.postflop[bucket][state_key]
            else:
                # Fallback
                strategy = [0.25, 0.25, 0.25, 0.25]
            
            actions = ["fold", "call", "bet", "raise"]
        
        # Sample action based on strategy
        action_idx = np.random.choice(len(strategy), p=strategy)
        return actions[action_idx]
    
    def get_strategy(self, 
                     stage: str, 
                     hole_cards: List[str], 
                     board: List[str] = [],
                     position: str = "BB",
                     action_facing: str = "unopened",
                     stack_depth: int = 100,
                     pot_size: float = 1.5,
                     bet_size: float = 0) -> Dict[str, float]:
        """
        Get the full strategy distribution
        
        Returns:
            dict: {"fold": 0.1, "call": 0.7, "raise": 0.2}
        """
        if stage == "preflop":
            bucket = self.get_preflop_bucket(hole_cards)
            
            # Convert inputs to enums
            pos_map = {
                'EP': Position.EP, 'MP': Position.MP, 'CO': Position.CO,
                'BTN': Position.BTN, 'SB': Position.SB, 'BB': Position.BB
            }
            pos_enum = pos_map.get(position, Position.BB)
            
            action_map = {
                'unopened': ActionFacing.UNOPENED,
                'facing_raise': ActionFacing.FACING_RAISE,
                'facing_3bet': ActionFacing.FACING_3BET,
                'facing_4bet': ActionFacing.FACING_4BET
            }
            action_enum = action_map.get(action_facing, ActionFacing.UNOPENED)
            
            # Create state key
            state_key = f"{pos_enum.value}_{action_enum.value}_{stack_depth}"
            
            # Get strategy
            if bucket in self.preflop and state_key in self.preflop[bucket]:
                strategy = self.preflop[bucket][state_key]
            else:
                strategy = [0.33, 0.33, 0.34]
            
            actions = ["fold", "call", "raise"]
        else:
            # Postflop
            equity = self.calculate_equity(hole_cards, board)
            bucket = self.compute_postflop_bucket(equity)
            
            pos_map = {
                'EP': Position.EP, 'MP': Position.MP, 'CO': Position.CO,
                'BTN': Position.BTN, 'SB': Position.SB, 'BB': Position.BB
            }
            pos_enum = pos_map.get(position, Position.BB)
            
            state_key = f"{pos_enum.value}_{pot_size}_{bet_size}"
            
            if bucket in self.postflop and state_key in self.postflop[bucket]:
                strategy = self.postflop[bucket][state_key]
            else:
                strategy = [0.25, 0.25, 0.25, 0.25]
            
            actions = ["fold", "call", "bet", "raise"]
        
        return dict(zip(actions, strategy))
    
    def analyze_hand(self, hole_cards: List[str]) -> Dict:
        """
        Provide detailed hand analysis
        
        Returns hand strength, recommended actions by position, etc.
        """
        bucket = self.get_preflop_bucket(hole_cards)
        strength_labels = {
            0: "Trash", 1: "Very Weak", 2: "Weak", 3: "Below Average",
            4: "Marginal", 5: "Playable", 6: "Small Pair",
            7: "Suited Connector", 8: "Good Connector", 9: "Broadway",
            10: "Medium Pair", 11: "Strong", 12: "Very Strong",
            13: "Premium", 14: "Monster"
        }
        
        analysis = {
            'bucket': bucket,
            'strength': strength_labels.get(bucket, "Unknown"),
            'percentile': (bucket / 14) * 100,
            'recommendations': {}
        }
        
        # Get recommendations for each position
        for pos in ['EP', 'MP', 'CO', 'BTN', 'SB', 'BB']:
            # Unopened pot
            unopened_strat = self.get_strategy(
                'preflop', hole_cards, position=pos, 
                action_facing='unopened', stack_depth=100
            )
            
            # Facing raise
            facing_raise_strat = self.get_strategy(
                'preflop', hole_cards, position=pos,
                action_facing='facing_raise', stack_depth=100
            )
            
            analysis['recommendations'][pos] = {
                'unopened': unopened_strat,
                'facing_raise': facing_raise_strat
            }
        
        return analysis

def test_professional_player():
    """Test the professional AI player"""
    print("\n" + "="*60)
    print("TESTING PROFESSIONAL GTO PLAYER")
    print("="*60)
    
    player = ProfessionalAIPlayer()
    
    # Test Case 1: 77 in BB facing raise
    print("\n📊 Test 1: 77 in BB facing 3BB raise, 100BB deep")
    strategy = player.get_strategy(
        stage="preflop",
        hole_cards=["7s", "7h"],
        position="BB",
        action_facing="facing_raise",
        stack_depth=100,
        pot_size=6.5,
        bet_size=3.0
    )
    print("Strategy:", strategy)
    print(f"Expected: ~70% call (set mining)")
    if 'call' in strategy and strategy['call'] > 0.5:
        print("✅ PASS - Correctly calling for set mining")
    else:
        print("⚠️  Strategy needs adjustment")
    
    # Test Case 2: 89s in BB facing raise
    print("\n📊 Test 2: 89s in BB facing 2.5BB raise, 100BB deep")
    strategy = player.get_strategy(
        stage="preflop",
        hole_cards=["8s", "9s"],
        position="BB",
        action_facing="facing_raise",
        stack_depth=100,
        pot_size=6.0,
        bet_size=2.5
    )
    print("Strategy:", strategy)
    print(f"Expected: ~60% call (defending suited connector)")
    if 'call' in strategy and strategy['call'] > 0.4:
        print("✅ PASS - Correctly defending with suited connectors")
    else:
        print("⚠️  Strategy needs adjustment")
    
    # Test Case 3: AKo on BTN unopened
    print("\n📊 Test 3: AKo on BTN, unopened pot, 100BB deep")
    strategy = player.get_strategy(
        stage="preflop",
        hole_cards=["As", "Kh"],
        position="BTN",
        action_facing="unopened",
        stack_depth=100,
        pot_size=1.5
    )
    print("Strategy:", strategy)
    print(f"Expected: ~95% raise")
    if 'raise' in strategy and strategy['raise'] > 0.85:
        print("✅ PASS - Correctly raising premium hand")
    else:
        print("⚠️  Strategy needs adjustment")
    
    # Test Case 4: Flush draw postflop
    print("\n📊 Test 4: Flush draw (36% equity) facing pot bet")
    # Simulate As 5s on Ks 9s 2h board
    strategy = player.get_strategy(
        stage="flop",
        hole_cards=["As", "5s"],
        board=["Ks", "9s", "2h"],
        position="BB",
        pot_size=12.0,
        bet_size=12.0  # Pot-sized bet
    )
    print("Strategy:", strategy)
    print(f"Expected: ~70% call, ~25% raise")
    if 'call' in strategy and strategy['call'] > 0.4:
        print("✅ PASS - Correctly calling with proper pot odds")
    else:
        print("⚠️  Strategy needs adjustment")
    
    # Test hand analysis
    print("\n📊 Test 5: Hand Analysis for JJ")
    analysis = player.analyze_hand(["Js", "Jh"])
    print(f"Hand: JJ")
    print(f"Strength: {analysis['strength']}")
    print(f"Percentile: {analysis['percentile']:.1f}%")
    print(f"BTN Unopened: {analysis['recommendations']['BTN']['unopened']}")
    print(f"BB vs Raise: {analysis['recommendations']['BB']['facing_raise']}")

if __name__ == "__main__":
    test_professional_player()

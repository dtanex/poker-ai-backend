"""
CFR AI Player - Uses trained CFR strategies to make poker decisions
"""
import numpy as np
import pickle
from typing import List, Dict, Tuple

# Card ranking for preflop bucketing
RANKS = {'2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, 
         '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14}


def compute_equity_buckets(hole_cards: List[str], board: List[str], num_buckets: int = 10) -> int:
    """
    Simple equity bucketing based on hand strength
    In production, you'd use actual equity calculation vs opponent ranges
    """
    # For now, use a simplified heuristic
    # TODO: Replace with actual Monte Carlo equity calculation
    
    # Parse cards
    rank1 = RANKS[hole_cards[0][0]]
    rank2 = RANKS[hole_cards[1][0]]
    
    # High card strength
    strength = (rank1 + rank2) / 28.0  # Normalize to 0-1
    
    # Pocket pair bonus
    if rank1 == rank2:
        strength += 0.2
    
    # Board texture (if any)
    if board:
        board_ranks = [RANKS[card[0]] for card in board]
        # Check for pairs/trips on board
        for hole_rank in [rank1, rank2]:
            if hole_rank in board_ranks:
                strength += 0.15  # Made a pair
    
    # Clamp to 0-1 range
    strength = min(1.0, max(0.0, strength))
    
    # Convert to bucket (0 to num_buckets-1)
    bucket = int(strength * num_buckets)
    if bucket >= num_buckets:
        bucket = num_buckets - 1
    
    return bucket


class CFRAIPlayer:
    """
    AI Player that uses CFR-trained strategies
    """
    
    def __init__(self, strategy_file: str = "cfr_strategies.pkl"):
        """
        Load trained CFR strategies
        
        Args:
            strategy_file: Path to pickled strategy dict
        """
        try:
            with open(strategy_file, 'rb') as f:
                strategies = pickle.load(f)
                self.preflop = strategies['preflop']
                self.postflop = strategies['postflop']
                print(f"✅ Loaded strategies from {strategy_file}")
                print(f"   Preflop buckets: {len(self.preflop)}")
                print(f"   Postflop buckets: {len(self.postflop)}")
        except FileNotFoundError:
            print(f"⚠️  Strategy file not found, using uniform random strategy")
            self.preflop = {}
            self.postflop = {}
    
    def get_preflop_bucket(self, hole_cards: List[str]) -> int:
        """
        Get preflop bucket (0-8) based on hand strength
        
        Buckets:
        0: Trash (72o, 83o, etc.)
        1-2: Weak hands
        3-4: Mediocre hands  
        5-6: Good hands (suited connectors, small pairs)
        7: Premium pairs (TT-QQ)
        8: Monster hands (AA, KK, AK)
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
        
        # Premium pairs
        if paired and high >= 13:  # AA, KK
            return 8
        
        # AK
        if high == 14 and low == 13:
            return 8 if suited else 7
        
        # Big pairs (QQ, JJ, TT)
        if paired and high >= 10:
            return 7
        
        # AQ, AJ suited
        if high == 14 and low >= 11 and suited:
            return 7
        
        # Medium pairs (99-66) and AQ, AJ offsuit
        if (paired and high >= 6) or (high == 14 and low >= 11):
            return 6
        
        # Suited connectors, AT+
        if (gap <= 1 and suited and high >= 9) or (high == 14 and low >= 10):
            return 6
        
        # Small pairs (55-22), suited aces, KQ
        if paired or (high == 14 and suited) or (high == 13 and low == 12):
            return 5
        
        # Suited broadways
        if high >= 10 and low >= 10 and suited:
            return 5
        
        # Weak aces, King-high
        if high >= 13 and low >= 7:
            return 4
        
        # Suited gappers, any two broadway
        if (suited and high >= 10) or (low >= 10):
            return 3
        
        # Random suited cards
        if suited:
            return 2
        
        # Connected cards
        if gap <= 2:
            return 1
        
        # Trash
        return 0
    
    def get_action(self, stage: str, hole_cards: List[str], board: List[str] = []) -> str:
        """
        Get a single action by sampling from the strategy
        
        Args:
            stage: 'preflop', 'flop', 'turn', or 'river'
            hole_cards: ['As', 'Kh']
            board: ['Qh', 'Jd', '7c'] (empty for preflop)
        
        Returns:
            action: 'fold', 'call', 'raise', or 'bet'
        """
        if stage == "preflop":
            bucket = self.get_preflop_bucket(hole_cards)
            strategy = self.preflop.get(bucket, [0.33, 0.33, 0.34])
            actions = ["fold", "call", "raise"]
        else:
            bucket = compute_equity_buckets(hole_cards, board)
            strategy = self.postflop.get(bucket, [0.25, 0.25, 0.25, 0.25])
            actions = ["fold", "call", "bet", "raise"]
        
        # Sample action based on strategy probabilities
        action_idx = np.random.choice(len(strategy), p=strategy)
        return actions[action_idx]
    
    def get_strategy(self, stage: str, hole_cards: List[str], board: List[str] = []) -> Dict[str, float]:
        """
        Get the full strategy (probabilities for all actions)
        
        Args:
            stage: 'preflop', 'flop', 'turn', or 'river'
            hole_cards: ['As', 'Kh']
            board: ['Qh', 'Jd', '7c'] (empty for preflop)
        
        Returns:
            dict: {"fold": 0.1, "call": 0.3, "raise": 0.6}
        """
        if stage == "preflop":
            bucket = self.get_preflop_bucket(hole_cards)
            strategy = self.preflop.get(bucket, [0.33, 0.33, 0.34])
            actions = ["fold", "call", "raise"]
        else:
            bucket = compute_equity_buckets(hole_cards, board)
            strategy = self.postflop.get(bucket, [0.25, 0.25, 0.25, 0.25])
            actions = ["fold", "call", "bet", "raise"]
        
        return dict(zip(actions, strategy))
    
    def get_bucket_info(self, stage: str, hole_cards: List[str], board: List[str] = []) -> Tuple[int, List[float]]:
        """
        Get the bucket number and raw strategy for debugging
        
        Returns:
            (bucket_number, strategy_array)
        """
        if stage == "preflop":
            bucket = self.get_preflop_bucket(hole_cards)
            strategy = self.preflop.get(bucket, [0.33, 0.33, 0.34])
        else:
            bucket = compute_equity_buckets(hole_cards, board)
            strategy = self.postflop.get(bucket, [0.25, 0.25, 0.25, 0.25])
        
        return bucket, strategy


if __name__ == "__main__":
    # Quick test
    print("\n🎰 Testing CFR AI Player...\n")
    
    player = CFRAIPlayer()
    
    # Test 1: Pocket Aces preflop
    print("Test 1: Pocket Aces preflop")
    strategy = player.get_strategy("preflop", ["As", "Ah"], [])
    bucket, raw_strat = player.get_bucket_info("preflop", ["As", "Ah"], [])
    print(f"Bucket: {bucket}")
    print(f"Strategy: {strategy}")
    action = player.get_action("preflop", ["As", "Ah"], [])
    print(f"Sampled action: {action}")
    print()
    
    # Test 2: Top pair on flop
    print("Test 2: AK on AQ7 flop")
    strategy = player.get_strategy("flop", ["As", "Kd"], ["Ah", "Qh", "7c"])
    bucket, raw_strat = player.get_bucket_info("flop", ["As", "Kd"], ["Ah", "Qh", "7c"])
    print(f"Bucket: {bucket}")
    print(f"Strategy: {strategy}")
    action = player.get_action("flop", ["As", "Kd"], ["Ah", "Qh", "7c"])
    print(f"Sampled action: {action}")
    print()
    
    # Test 3: Weak hand on flop
    print("Test 3: 72o on AKQ flop (trash)")
    strategy = player.get_strategy("flop", ["7s", "2d"], ["Ah", "Kh", "Qc"])
    bucket, raw_strat = player.get_bucket_info("flop", ["7s", "2d"], ["Ah", "Kh", "Qc"])
    print(f"Bucket: {bucket}")
    print(f"Strategy: {strategy}")
    action = player.get_action("flop", ["7s", "2d"], ["Ah", "Kh", "Qc"])
    print(f"Sampled action: {action}")
    print()
    
    # Test 4: Multiple samples to verify probability distribution
    print("Test 4: Sampling AA preflop 100 times to verify probabilities")
    actions_count = {"fold": 0, "call": 0, "raise": 0}
    for _ in range(100):
        action = player.get_action("preflop", ["As", "Ah"], [])
        actions_count[action] += 1
    
    print(f"Theoretical strategy: {player.get_strategy('preflop', ['As', 'Ah'], [])}")
    empirical = {k: v/100 for k, v in actions_count.items()}
    print(f"Empirical distribution: {empirical}")
    print()
    
    print("✅ All tests complete!")

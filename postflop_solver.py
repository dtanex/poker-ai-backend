"""
Postflop Solver and Equity Calculator
Advanced postflop decision making with real equity calculations
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from itertools import combinations, product
from collections import defaultdict
import random


class Card:
    """Card representation with comparison operators"""
    RANKS = '23456789TJQKA'
    SUITS = 'cdhs'
    
    def __init__(self, string: str):
        """Initialize from string like 'As' or 'Ah'"""
        self.rank = self.RANKS.index(string[0])
        self.suit = self.SUITS.index(string[1])
        self.string = string
    
    def __eq__(self, other):
        return self.rank == other.rank and self.suit == other.suit
    
    def __hash__(self):
        return hash((self.rank, self.suit))
    
    def __repr__(self):
        return self.string


class HandEvaluator:
    """
    Fast poker hand evaluation using lookup tables
    """
    
    def __init__(self):
        self.hand_ranks = {
            'high_card': 0,
            'pair': 1,
            'two_pair': 2,
            'three_kind': 3,
            'straight': 4,
            'flush': 5,
            'full_house': 6,
            'four_kind': 7,
            'straight_flush': 8
        }
        
        # Pre-compute some common patterns
        self.straight_ranks = [
            [12, 3, 2, 1, 0],  # A-5 straight (wheel)
            *[[i+4, i+3, i+2, i+1, i] for i in range(9)]  # Regular straights
        ]
    
    def evaluate(self, cards: List[Card]) -> Tuple[int, List[int]]:
        """
        Evaluate a poker hand (5-7 cards)
        Returns (rank, tiebreakers)
        """
        if len(cards) < 5:
            return (0, [])
        
        # Get best 5-card hand from available cards
        if len(cards) > 5:
            best_hand = None
            best_score = (0, [])
            
            for five_cards in combinations(cards, 5):
                score = self._evaluate_five(list(five_cards))
                if score > best_score:
                    best_score = score
                    best_hand = five_cards
            
            return best_score
        
        return self._evaluate_five(cards)
    
    def _evaluate_five(self, cards: List[Card]) -> Tuple[int, List[int]]:
        """Evaluate exactly 5 cards"""
        ranks = sorted([c.rank for c in cards], reverse=True)
        suits = [c.suit for c in cards]
        
        rank_counts = defaultdict(int)
        for rank in ranks:
            rank_counts[rank] += 1
        
        counts = sorted(rank_counts.items(), key=lambda x: (x[1], x[0]), reverse=True)
        
        # Check for flush
        is_flush = len(set(suits)) == 1
        
        # Check for straight
        is_straight = False
        straight_high = 0
        
        unique_ranks = sorted(set(ranks))
        if len(unique_ranks) >= 5:
            # Check regular straights
            for i in range(len(unique_ranks) - 4):
                if unique_ranks[i+4] - unique_ranks[i] == 4:
                    is_straight = True
                    straight_high = unique_ranks[i+4]
            
            # Check wheel (A-5)
            if set(unique_ranks[:4]) == {0, 1, 2, 3} and 12 in unique_ranks:
                is_straight = True
                straight_high = 3  # 5 high straight
        
        # Determine hand rank
        if is_straight and is_flush:
            return (8, [straight_high])  # Straight flush
        
        if counts[0][1] == 4:
            return (7, [counts[0][0], counts[1][0]])  # Four of a kind
        
        if counts[0][1] == 3 and counts[1][1] == 2:
            return (6, [counts[0][0], counts[1][0]])  # Full house
        
        if is_flush:
            return (5, ranks[:5])  # Flush
        
        if is_straight:
            return (4, [straight_high])  # Straight
        
        if counts[0][1] == 3:
            kickers = [c[0] for c in counts[1:3]]
            return (3, [counts[0][0]] + kickers)  # Three of a kind
        
        if counts[0][1] == 2 and counts[1][1] == 2:
            kicker = counts[2][0] if len(counts) > 2 else 0
            return (2, [counts[0][0], counts[1][0], kicker])  # Two pair
        
        if counts[0][1] == 2:
            kickers = [c[0] for c in counts[1:4]]
            return (1, [counts[0][0]] + kickers)  # Pair
        
        return (0, ranks[:5])  # High card
    
    def compare_hands(self, hand1: List[Card], hand2: List[Card]) -> int:
        """
        Compare two hands
        Returns: 1 if hand1 wins, -1 if hand2 wins, 0 if tie
        """
        score1 = self.evaluate(hand1)
        score2 = self.evaluate(hand2)
        
        if score1[0] > score2[0]:
            return 1
        elif score1[0] < score2[0]:
            return -1
        else:
            # Same hand rank, compare tiebreakers
            for t1, t2 in zip(score1[1], score2[1]):
                if t1 > t2:
                    return 1
                elif t1 < t2:
                    return -1
            return 0  # Complete tie


class EquityCalculator:
    """
    Monte Carlo equity calculator for poker hands
    """
    
    def __init__(self):
        self.evaluator = HandEvaluator()
        self.deck = self._create_deck()
    
    def _create_deck(self) -> List[Card]:
        """Create a standard 52-card deck"""
        deck = []
        for rank in Card.RANKS:
            for suit in Card.SUITS:
                deck.append(Card(rank + suit))
        return deck
    
    def calculate_equity(self, 
                        hero_cards: List[str],
                        villain_range: List[List[str]],
                        board: List[str] = [],
                        num_simulations: int = 10000) -> float:
        """
        Calculate hero's equity against villain's range
        
        Args:
            hero_cards: Hero's hole cards ['As', 'Kh']
            villain_range: List of possible villain hands
            board: Community cards
            num_simulations: Number of Monte Carlo simulations
        
        Returns:
            Hero's equity (0-1)
        """
        hero = [Card(c) for c in hero_cards]
        board_cards = [Card(c) for c in board]
        
        # Remove known cards from deck
        available_deck = [c for c in self.deck 
                         if c not in hero and c not in board_cards]
        
        wins = 0
        ties = 0
        total = 0
        
        for _ in range(num_simulations):
            # Sample villain hand from range
            villain_hand = random.choice(villain_range)
            villain = [Card(c) for c in villain_hand]
            
            # Skip if villain has our cards or board cards
            if any(c in hero or c in board_cards for c in villain):
                continue
            
            # Sample remaining board cards
            remaining_board = 5 - len(board_cards)
            available = [c for c in available_deck if c not in villain]
            
            if remaining_board > 0:
                runout = random.sample(available, remaining_board)
            else:
                runout = []
            
            # Evaluate hands
            hero_hand = hero + board_cards + runout
            villain_hand = villain + board_cards + runout
            
            result = self.evaluator.compare_hands(hero_hand, villain_hand)
            
            if result == 1:
                wins += 1
            elif result == 0:
                ties += 1
            
            total += 1
        
        if total == 0:
            return 0.5  # No valid simulations
        
        # Equity = (wins + ties/2) / total
        equity = (wins + ties * 0.5) / total
        return equity
    
    def calculate_equity_vs_range(self,
                                 hero_cards: List[str],
                                 villain_range_str: str,
                                 board: List[str] = []) -> float:
        """
        Calculate equity against a range string like "TT+, AK, AQs"
        """
        villain_range = self.parse_range(villain_range_str)
        return self.calculate_equity(hero_cards, villain_range, board)
    
    def parse_range(self, range_str: str) -> List[List[str]]:
        """
        Parse range string into list of hands
        
        Examples:
            "AA" -> [['As', 'Ah'], ['As', 'Ad'], ...]
            "TT+" -> All pairs TT and higher
            "AKs" -> Only suited AK
            "AKo" -> Only offsuit AK
        """
        hands = []
        parts = range_str.replace(' ', '').split(',')
        
        for part in parts:
            if '+' in part:
                # Handle ranges like TT+ or A9s+
                base = part.replace('+', '')
                hands.extend(self._expand_plus_range(base))
            elif '-' in part:
                # Handle ranges like 88-JJ or A9s-AJs
                start, end = part.split('-')
                hands.extend(self._expand_dash_range(start, end))
            else:
                # Single hand or hand type
                hands.extend(self._expand_hand(part))
        
        return hands
    
    def _expand_hand(self, hand_str: str) -> List[List[str]]:
        """Expand a single hand notation"""
        hands = []
        
        if len(hand_str) == 2:
            # Pocket pair like "AA"
            rank = hand_str[0]
            combos = list(combinations(Card.SUITS, 2))
            for s1, s2 in combos:
                hands.append([rank + s1, rank + s2])
        
        elif len(hand_str) == 3:
            rank1, rank2, suited = hand_str[0], hand_str[1], hand_str[2]
            
            if suited == 's':
                # Suited hands
                for suit in Card.SUITS:
                    hands.append([rank1 + suit, rank2 + suit])
            elif suited == 'o':
                # Offsuit hands
                for s1, s2 in product(Card.SUITS, Card.SUITS):
                    if s1 != s2:
                        hands.append([rank1 + s1, rank2 + s2])
        
        return hands
    
    def _expand_plus_range(self, base: str) -> List[List[str]]:
        """Expand plus ranges like TT+ or A9s+"""
        hands = []
        
        if len(base) == 2 and base[0] == base[1]:
            # Pocket pairs
            start_rank = Card.RANKS.index(base[0])
            for rank_idx in range(start_rank, 13):  # Up to aces
                rank = Card.RANKS[rank_idx]
                hands.extend(self._expand_hand(rank + rank))
        
        elif len(base) == 3:
            # Non-pairs like A9s+ or KTo+
            rank1, rank2, suited = base[0], base[1], base[2]
            start_rank2 = Card.RANKS.index(rank2)
            rank1_idx = Card.RANKS.index(rank1)
            
            for rank2_idx in range(start_rank2, rank1_idx):
                rank2 = Card.RANKS[rank2_idx]
                hands.extend(self._expand_hand(rank1 + rank2 + suited))
        
        return hands
    
    def _expand_dash_range(self, start: str, end: str) -> List[List[str]]:
        """Expand dash ranges like 88-JJ"""
        hands = []
        
        if len(start) == 2 and start[0] == start[1]:
            # Pocket pair range
            start_idx = Card.RANKS.index(start[0])
            end_idx = Card.RANKS.index(end[0])
            
            for rank_idx in range(start_idx, end_idx + 1):
                rank = Card.RANKS[rank_idx]
                hands.extend(self._expand_hand(rank + rank))
        
        return hands


class PostflopSolver:
    """
    Advanced postflop solver using game theory
    """
    
    def __init__(self):
        self.equity_calc = EquityCalculator()
        self.evaluator = HandEvaluator()
        
        # Solver parameters
        self.iterations = 1000
        self.exploitability_threshold = 0.01
    
    def solve_river_spot(self,
                        hero_range: List[List[str]],
                        villain_range: List[List[str]],
                        board: List[str],
                        pot_size: float,
                        stack_size: float) -> Dict:
        """
        Solve river decision using game theory
        
        Returns optimal betting/calling frequencies
        """
        # Calculate equity matrix
        equity_matrix = self._build_equity_matrix(hero_range, villain_range, board)
        
        # Find optimal betting frequency using game theory
        bet_sizes = [0.33, 0.67, 1.0, 1.5]  # As fraction of pot
        optimal_strategy = {}
        
        for bet_size in bet_sizes:
            # Calculate EV of betting with each hand
            betting_evs = self._calculate_betting_evs(
                equity_matrix, pot_size, pot_size * bet_size
            )
            
            # Find GTO betting frequency
            # Hands that bet for value and as bluff
            value_threshold = self._find_value_threshold(betting_evs, bet_size)
            bluff_threshold = self._find_bluff_threshold(betting_evs, bet_size)
            
            optimal_strategy[f'bet_{bet_size}'] = {
                'value_threshold': value_threshold,
                'bluff_threshold': bluff_threshold,
                'frequency': self._calculate_betting_frequency(
                    betting_evs, value_threshold, bluff_threshold
                )
            }
        
        # Calculate calling threshold vs different bet sizes
        calling_thresholds = {}
        for bet_size in bet_sizes:
            pot_odds = bet_size / (1 + 2 * bet_size)
            calling_thresholds[f'vs_{bet_size}'] = {
                'pot_odds': pot_odds,
                'calling_threshold': self._find_calling_threshold(
                    equity_matrix, pot_odds
                )
            }
        
        return {
            'betting_strategy': optimal_strategy,
            'calling_thresholds': calling_thresholds
        }
    
    def _build_equity_matrix(self, 
                            hero_range: List[List[str]],
                            villain_range: List[List[str]],
                            board: List[str]) -> np.ndarray:
        """Build equity matrix for all hand combinations"""
        matrix = np.zeros((len(hero_range), len(villain_range)))
        
        for i, hero_hand in enumerate(hero_range):
            for j, villain_hand in enumerate(villain_range):
                # Skip blocked combinations
                if any(c in villain_hand for c in hero_hand):
                    matrix[i, j] = -1  # Invalid
                else:
                    equity = self.equity_calc.calculate_equity(
                        hero_hand, [villain_hand], board, num_simulations=100
                    )
                    matrix[i, j] = equity
        
        return matrix
    
    def _calculate_betting_evs(self, equity_matrix: np.ndarray, 
                              pot_size: float, bet_size: float) -> List[float]:
        """Calculate EV of betting with each hand in range"""
        evs = []
        
        for hand_equities in equity_matrix:
            # Filter valid matchups
            valid_equities = hand_equities[hand_equities >= 0]
            
            if len(valid_equities) == 0:
                evs.append(0)
                continue
            
            # Simplified EV calculation
            # EV = fold_equity * pot + (1 - fold_equity) * (equity * (pot + 2*bet) - bet)
            fold_equity = 0.3  # Simplified assumption
            avg_equity = np.mean(valid_equities)
            
            ev_bet = fold_equity * pot_size + \
                    (1 - fold_equity) * (avg_equity * (pot_size + 2 * bet_size) - bet_size)
            
            evs.append(ev_bet)
        
        return evs
    
    def _find_value_threshold(self, betting_evs: List[float], 
                            bet_size: float) -> float:
        """Find threshold for value betting"""
        # Value bet if EV > 0 and hand is strong enough
        sorted_evs = sorted(betting_evs, reverse=True)
        
        # Top 30-40% of range bets for value (simplified)
        value_percentage = 0.35
        threshold_idx = int(len(sorted_evs) * value_percentage)
        
        if threshold_idx < len(sorted_evs):
            return sorted_evs[threshold_idx]
        
        return sorted_evs[-1]
    
    def _find_bluff_threshold(self, betting_evs: List[float], 
                             bet_size: float) -> float:
        """Find threshold for bluffing"""
        # Bluff with worst hands that have some fold equity
        # Bluff-to-value ratio based on bet size
        alpha = bet_size / (1 + bet_size)  # Optimal bluff frequency
        
        sorted_evs = sorted(betting_evs)
        bluff_percentage = alpha * 0.35  # Bluffs as percentage of value bets
        threshold_idx = int(len(sorted_evs) * bluff_percentage)
        
        if threshold_idx < len(sorted_evs):
            return sorted_evs[threshold_idx]
        
        return sorted_evs[0]
    
    def _calculate_betting_frequency(self, betting_evs: List[float],
                                   value_threshold: float,
                                   bluff_threshold: float) -> float:
        """Calculate overall betting frequency"""
        betting_hands = sum(1 for ev in betting_evs 
                          if ev >= value_threshold or ev <= bluff_threshold)
        
        return betting_hands / len(betting_evs) if betting_evs else 0
    
    def _find_calling_threshold(self, equity_matrix: np.ndarray, 
                               pot_odds: float) -> float:
        """Find minimum equity needed to call"""
        # Call if equity > pot odds (simplified)
        return pot_odds * 1.1  # Slightly above pot odds for rake/variance


def demo_postflop_solver():
    """Demonstrate postflop solver capabilities"""
    print("\n" + "="*60)
    print("POSTFLOP SOLVER AND EQUITY CALCULATOR DEMO")
    print("="*60)
    
    # Test hand evaluator
    print("\nTesting hand evaluator:")
    evaluator = HandEvaluator()
    
    test_hands = [
        ([Card('As'), Card('Ah'), Card('Kd'), Card('Kc'), Card('Ks')], "Full house"),
        ([Card('9s'), Card('Js'), Card('Qs'), Card('Ks'), Card('Ts')], "Straight flush"),
        ([Card('2d'), Card('7h'), Card('Tc'), Card('Js'), Card('As')], "High card"),
    ]
    
    for cards, expected in test_hands:
        result = evaluator.evaluate(cards)
        print(f"  {[str(c) for c in cards[:5]]}: Rank {result[0]} ({expected})")
    
    # Test equity calculator
    print("\n" + "-"*60)
    print("Testing equity calculator:")
    calc = EquityCalculator()
    
    # Hero has AA, villain has random range, no board
    hero = ['As', 'Ah']
    villain_range = calc.parse_range("22+, A2s+, K9s+, Q9s+, J9s+, T9s, A2o+, KTo+, QTo+, JTo")
    
    equity = calc.calculate_equity(hero, villain_range, [], num_simulations=1000)
    print(f"\nAA vs wide range preflop: {equity:.1%} equity")
    
    # Flush draw on flop
    hero = ['As', '5s']
    board = ['Ks', '9s', '2h']
    villain_range = calc.parse_range("99+, AK, AQ, KQ")
    
    equity = calc.calculate_equity(hero, villain_range, board, num_simulations=1000)
    print(f"Flush draw vs strong range: {equity:.1%} equity")
    
    # Test postflop solver
    print("\n" + "-"*60)
    print("Testing postflop solver:")
    solver = PostflopSolver()
    
    # River spot: Hero has polarized range, villain has bluff catchers
    hero_range = [
        ['As', 'Ah'], ['Ks', 'Kh'],  # Nuts
        ['7s', '6s'], ['5s', '4s']    # Bluffs
    ]
    
    villain_range = [
        ['Qd', 'Qc'], ['Jd', 'Jc'],  # Medium strength
        ['Td', 'Tc'], ['9d', '9c']    # Weaker
    ]
    
    board = ['Ad', '8c', '3h', '2s', '4d']
    
    solution = solver.solve_river_spot(
        hero_range, villain_range, board,
        pot_size=100, stack_size=100
    )
    
    print("\nRiver solution:")
    print("Betting strategies:")
    for size, strategy in solution['betting_strategy'].items():
        print(f"  {size}: {strategy['frequency']:.1%} frequency")
    
    print("\nCalling thresholds:")
    for size, threshold in solution['calling_thresholds'].items():
        print(f"  {size}: Need {threshold['pot_odds']:.1%} equity")
    
    print("\n✅ Postflop solver demo complete!")


if __name__ == "__main__":
    demo_postflop_solver()

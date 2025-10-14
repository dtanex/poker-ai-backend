"""
Simplified CFR Strategy Trainer
Generates poker strategies through Counterfactual Regret Minimization

This is a SIMPLIFIED version for MVP. For production, you'd want:
- More buckets (50+ preflop, 200+ postflop)
- Real equity calculations
- More sophisticated game tree
- Longer training (millions of iterations)
"""
import numpy as np
import pickle
from typing import List, Dict, Tuple
from collections import defaultdict


class SimplifiedCFRTrainer:
    """
    Trains basic GTO poker strategies using CFR
    
    Simplifications for MVP:
    - Only heads-up (2 players)
    - Fixed bet sizes (1x pot)
    - Simplified game tree
    - Coarse abstractions (9 preflop, 10 postflop)
    """
    
    def __init__(self):
        # Strategy tables
        self.preflop_regrets = defaultdict(lambda: np.zeros(3))  # fold, call, raise
        self.preflop_strategies = defaultdict(lambda: np.ones(3) / 3)
        self.preflop_strategy_sum = defaultdict(lambda: np.zeros(3))
        
        self.postflop_regrets = defaultdict(lambda: np.zeros(4))  # fold, call, bet, raise
        self.postflop_strategies = defaultdict(lambda: np.ones(4) / 4)
        self.postflop_strategy_sum = defaultdict(lambda: np.zeros(4))
        
        print("🎰 CFR Trainer initialized")
        print("   Preflop actions: fold, call, raise")
        print("   Postflop actions: fold, call, bet, raise")
    
    def update_strategy(self, regrets: np.ndarray) -> np.ndarray:
        """
        Convert regrets to strategy using regret matching
        """
        # Only use positive regrets
        positive_regrets = np.maximum(regrets, 0)
        
        # Normalize to probability distribution
        sum_regrets = np.sum(positive_regrets)
        if sum_regrets > 0:
            strategy = positive_regrets / sum_regrets
        else:
            # If no positive regrets, use uniform distribution
            strategy = np.ones(len(regrets)) / len(regrets)
        
        return strategy
    
    def train_preflop(self, num_iterations: int = 10000):
        """
        Train preflop strategies for all 9 buckets
        
        Buckets:
        0: Trash (72o, etc)
        1-2: Weak
        3-4: Mediocre
        5-6: Good (suited connectors, small pairs)
        7: Premium (TT-QQ, AK)
        8: Monsters (AA, KK)
        """
        print(f"\n🏋️ Training preflop strategies ({num_iterations} iterations)...")
        
        for iteration in range(num_iterations):
            # Simulate for each bucket
            for bucket in range(9):
                # Get current strategy
                strategy = self.update_strategy(self.preflop_regrets[bucket])
                self.preflop_strategies[bucket] = strategy
                
                # Accumulate strategy for averaging
                self.preflop_strategy_sum[bucket] += strategy
                
                # Simulate outcomes and calculate regrets
                # This is simplified - real CFR traverses full game tree
                
                # Sample an action
                action = np.random.choice(3, p=strategy)
                
                # Calculate utility for each action (simplified)
                utilities = self._calculate_preflop_utilities(bucket)
                
                # Calculate regret for each action
                action_utility = utilities[action]
                regrets = utilities - action_utility
                
                # Update cumulative regrets
                self.preflop_regrets[bucket] += regrets
            
            # Progress bar
            if (iteration + 1) % 2000 == 0:
                print(f"   Progress: {iteration + 1}/{num_iterations} iterations")
        
        print("✅ Preflop training complete")
    
    def _calculate_preflop_utilities(self, bucket: int) -> np.ndarray:
        """
        Calculate utility for each action based on hand strength
        
        Returns: [fold_utility, call_utility, raise_utility]
        """
        # Stronger hands should prefer aggressive actions
        hand_strength = bucket / 8.0  # Normalize to 0-1
        
        # Base utilities (these are heuristic, real CFR learns these)
        fold_utility = 0  # Always 0 (you lose the hand)
        
        # Call utility increases with hand strength
        call_utility = hand_strength * 0.5 - 0.1
        
        # Raise utility increases even more with hand strength
        # But has higher variance (risk of getting reraised)
        raise_utility = hand_strength * 1.0 - 0.2
        
        # Add some noise to encourage exploration
        noise = np.random.normal(0, 0.1, 3)
        
        return np.array([fold_utility, call_utility, raise_utility]) + noise
    
    def train_postflop(self, num_iterations: int = 10000):
        """
        Train postflop strategies for all 10 buckets
        
        Buckets based on equity:
        0-1: Very weak (0-20% equity)
        2-3: Weak (20-40%)
        4-5: Mediocre (40-60%)
        6-7: Strong (60-80%)
        8-9: Very strong (80-100%)
        """
        print(f"\n🏋️ Training postflop strategies ({num_iterations} iterations)...")
        
        for iteration in range(num_iterations):
            # Simulate for each bucket
            for bucket in range(10):
                # Get current strategy
                strategy = self.update_strategy(self.postflop_regrets[bucket])
                self.postflop_strategies[bucket] = strategy
                
                # Accumulate strategy for averaging
                self.postflop_strategy_sum[bucket] += strategy
                
                # Sample an action
                action = np.random.choice(4, p=strategy)
                
                # Calculate utility for each action
                utilities = self._calculate_postflop_utilities(bucket)
                
                # Calculate regret
                action_utility = utilities[action]
                regrets = utilities - action_utility
                
                # Update cumulative regrets
                self.postflop_regrets[bucket] += regrets
            
            # Progress bar
            if (iteration + 1) % 2000 == 0:
                print(f"   Progress: {iteration + 1}/{num_iterations} iterations")
        
        print("✅ Postflop training complete")
    
    def _calculate_postflop_utilities(self, bucket: int) -> np.ndarray:
        """
        Calculate utility for each action based on equity
        
        Returns: [fold_utility, call_utility, bet_utility, raise_utility]
        """
        equity = bucket / 10.0  # Normalize to 0-1
        
        # Base utilities (heuristic)
        fold_utility = 0  # Always 0
        
        # Call utility scales with equity
        call_utility = equity * 0.4 - 0.1
        
        # Bet utility (semi-bluff or value bet)
        # Good for very weak (bluff) and very strong (value)
        if equity < 0.3 or equity > 0.7:
            bet_utility = 0.5
        else:
            bet_utility = 0.2
        
        # Raise utility (for very strong hands)
        raise_utility = equity * 0.8 - 0.3
        
        # Add noise
        noise = np.random.normal(0, 0.1, 4)
        
        return np.array([fold_utility, call_utility, bet_utility, raise_utility]) + noise
    
    def get_average_strategies(self) -> Dict[str, Dict[int, List[float]]]:
        """
        Get time-averaged strategies (Nash equilibrium approximation)
        """
        print("\n📊 Computing average strategies...")
        
        preflop_final = {}
        for bucket in range(9):
            strategy_sum = self.preflop_strategy_sum[bucket]
            total = np.sum(strategy_sum)
            if total > 0:
                preflop_final[bucket] = (strategy_sum / total).tolist()
            else:
                preflop_final[bucket] = [0.33, 0.33, 0.34]  # Uniform
        
        postflop_final = {}
        for bucket in range(10):
            strategy_sum = self.postflop_strategy_sum[bucket]
            total = np.sum(strategy_sum)
            if total > 0:
                postflop_final[bucket] = (strategy_sum / total).tolist()
            else:
                postflop_final[bucket] = [0.25, 0.25, 0.25, 0.25]  # Uniform
        
        print("✅ Average strategies computed")
        return {
            'preflop': preflop_final,
            'postflop': postflop_final
        }
    
    def save_strategies(self, filename: str = "cfr_strategies.pkl"):
        """
        Save trained strategies to file
        """
        strategies = self.get_average_strategies()
        
        with open(filename, 'wb') as f:
            pickle.dump(strategies, f)
        
        print(f"\n💾 Strategies saved to {filename}")
        print(f"   File size: {self._get_file_size(filename)}")
    
    def _get_file_size(self, filename: str) -> str:
        """Get human-readable file size"""
        import os
        size = os.path.getsize(filename)
        for unit in ['B', 'KB', 'MB']:
            if size < 1024:
                return f"{size:.1f} {unit}"
            size /= 1024
        return f"{size:.1f} GB"
    
    def print_sample_strategies(self):
        """
        Print some example strategies for verification
        """
        print("\n" + "="*60)
        print("SAMPLE STRATEGIES")
        print("="*60)
        
        strategies = self.get_average_strategies()
        
        # Preflop examples
        print("\n📈 PREFLOP STRATEGIES:")
        examples = [
            (0, "Trash (72o)"),
            (4, "Mediocre"),
            (8, "Premium (AA, KK)")
        ]
        
        for bucket, description in examples:
            strat = strategies['preflop'][bucket]
            print(f"\nBucket {bucket}: {description}")
            print(f"   Fold:  {strat[0]:>6.1%}")
            print(f"   Call:  {strat[1]:>6.1%}")
            print(f"   Raise: {strat[2]:>6.1%}")
        
        # Postflop examples
        print("\n📈 POSTFLOP STRATEGIES:")
        examples = [
            (0, "Very Weak (0-10% equity)"),
            (5, "Medium (50% equity)"),
            (9, "Very Strong (90%+ equity)")
        ]
        
        for bucket, description in examples:
            strat = strategies['postflop'][bucket]
            print(f"\nBucket {bucket}: {description}")
            print(f"   Fold:  {strat[0]:>6.1%}")
            print(f"   Call:  {strat[1]:>6.1%}")
            print(f"   Bet:   {strat[2]:>6.1%}")
            print(f"   Raise: {strat[3]:>6.1%}")


def main():
    """
    Main training pipeline
    """
    print("\n" + "🎰"*30)
    print("CFR POKER STRATEGY TRAINER")
    print("🎰"*30)
    
    # Initialize trainer
    trainer = SimplifiedCFRTrainer()
    
    # Train preflop (fast - only 9 buckets)
    trainer.train_preflop(num_iterations=10000)
    
    # Train postflop (also fast - only 10 buckets)
    trainer.train_postflop(num_iterations=10000)
    
    # Show sample strategies
    trainer.print_sample_strategies()
    
    # Save to file
    trainer.save_strategies("cfr_strategies.pkl")
    
    print("\n" + "="*60)
    print("🎉 TRAINING COMPLETE!")
    print("="*60)
    print("\nNext steps:")
    print("1. Test the strategies: python test_complete.py")
    print("2. Run the AI player: python aiplayer.py")
    print("3. Deploy the FastAPI backend")
    print("4. Connect to your Next.js frontend")
    print("5. LAUNCH! 🚀")


if __name__ == "__main__":
    main()

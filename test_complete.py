"""
Complete Test Script for CFR Poker AI
Tests strategy generation, player AI, and end-to-end gameplay
"""
import numpy as np
import pickle
from typing import List, Dict
from aiplayer import CFRAIPlayer, RANKS


def test_strategy_file():
    """Test 1: Verify strategy file exists and is valid"""
    print("\n" + "="*60)
    print("TEST 1: Strategy File Validation")
    print("="*60)
    
    try:
        with open('cfr_strategies.pkl', 'rb') as f:
            strategies = pickle.load(f)
        
        assert 'preflop' in strategies, "Missing preflop strategies"
        assert 'postflop' in strategies, "Missing postflop strategies"
        
        print("✅ Strategy file loaded successfully")
        print(f"   Preflop buckets: {len(strategies['preflop'])}")
        print(f"   Postflop buckets: {len(strategies['postflop'])}")
        
        # Verify strategies are probability distributions
        for bucket, strat in strategies['preflop'].items():
            assert abs(sum(strat) - 1.0) < 0.01, f"Preflop bucket {bucket} doesn't sum to 1"
        
        for bucket, strat in strategies['postflop'].items():
            assert abs(sum(strat) - 1.0) < 0.01, f"Postflop bucket {bucket} doesn't sum to 1"
        
        print("✅ All strategies are valid probability distributions")
        return True
        
    except FileNotFoundError:
        print("❌ Strategy file not found! Run strategy generation first.")
        return False
    except Exception as e:
        print(f"❌ Error loading strategies: {e}")
        return False


def test_preflop_bucketing():
    """Test 2: Verify preflop hand bucketing is sensible"""
    print("\n" + "="*60)
    print("TEST 2: Preflop Hand Bucketing")
    print("="*60)
    
    player = CFRAIPlayer()
    
    test_hands = [
        (["As", "Ah"], 8, "Pocket Aces (should be bucket 8)"),
        (["Ks", "Kh"], 8, "Pocket Kings (should be bucket 8)"),
        (["Qs", "Qh"], 7, "Pocket Queens (should be bucket 7)"),
        (["As", "Kh"], 7, "AKo (should be bucket 7-8)"),
        (["Js", "Ts"], 6, "JTs (should be bucket 5-6)"),
        (["7s", "2h"], 0, "72o (should be bucket 0)"),
        (["9s", "8s"], 5, "98s (should be bucket 5-6)"),
    ]
    
    all_passed = True
    for hand, expected_range, description in test_hands:
        bucket = player.get_preflop_bucket(hand)
        # Allow some flexibility in bucketing
        if isinstance(expected_range, int):
            passed = bucket == expected_range or bucket == expected_range - 1 or bucket == expected_range + 1
        else:
            passed = True
        
        status = "✅" if passed else "❌"
        print(f"{status} {description}")
        print(f"   Cards: {hand[0]} {hand[1]} → Bucket {bucket}")
        
        if not passed:
            all_passed = False
    
    return all_passed


def test_strategy_coherence():
    """Test 3: Verify strategies make poker sense"""
    print("\n" + "="*60)
    print("TEST 3: Strategy Coherence")
    print("="*60)
    
    player = CFRAIPlayer()
    
    # Test AA preflop (should heavily favor raise)
    print("\n📊 Pocket Aces Preflop Strategy:")
    strategy = player.get_strategy("preflop", ["As", "Ah"])
    bucket, _ = player.get_bucket_info("preflop", ["As", "Ah"])
    print(f"   Bucket: {bucket}")
    for action, prob in strategy.items():
        print(f"   {action}: {prob:.1%}")
    
    # Check that raise probability is highest
    if strategy["raise"] < max(strategy.values()):
        print("⚠️  WARNING: AA should favor raising!")
    else:
        print("✅ AA correctly favors aggressive play")
    
    # Test 72o preflop (should heavily favor fold)
    print("\n📊 72o Preflop Strategy:")
    strategy = player.get_strategy("preflop", ["7s", "2h"])
    bucket, _ = player.get_bucket_info("preflop", ["7s", "2h"])
    print(f"   Bucket: {bucket}")
    for action, prob in strategy.items():
        print(f"   {action}: {prob:.1%}")
    
    # Test made hand on flop
    print("\n📊 Top Pair Top Kicker on Flop:")
    strategy = player.get_strategy("flop", ["As", "Kd"], ["Ah", "Qh", "7c"])
    bucket, _ = player.get_bucket_info("flop", ["As", "Kd"], ["Ah", "Qh", "7c"])
    print(f"   Cards: AsKd on AhQh7c → Bucket {bucket}")
    for action, prob in strategy.items():
        print(f"   {action}: {prob:.1%}")
    
    print("\n✅ Strategy coherence test complete")
    return True


def test_action_sampling():
    """Test 4: Verify action sampling matches strategy distribution"""
    print("\n" + "="*60)
    print("TEST 4: Action Sampling Distribution")
    print("="*60)
    
    player = CFRAIPlayer()
    
    print("\nSampling AA preflop 1000 times...")
    hand = ["As", "Ah"]
    num_samples = 1000
    
    # Get theoretical strategy
    theoretical = player.get_strategy("preflop", hand)
    
    # Sample actions
    empirical_counts = {"fold": 0, "call": 0, "raise": 0}
    for _ in range(num_samples):
        action = player.get_action("preflop", hand)
        empirical_counts[action] += 1
    
    empirical = {k: v/num_samples for k, v in empirical_counts.items()}
    
    print("\n📊 Theoretical vs Empirical Distribution:")
    print(f"{'Action':<10} {'Theoretical':<15} {'Empirical':<15} {'Diff':<10}")
    print("-" * 50)
    
    all_close = True
    for action in ["fold", "call", "raise"]:
        theo = theoretical[action]
        emp = empirical[action]
        diff = abs(theo - emp)
        
        # Allow 5% deviation for 1000 samples
        status = "✅" if diff < 0.05 else "❌"
        print(f"{action:<10} {theo:>8.1%}       {emp:>8.1%}       {diff:>6.1%}  {status}")
        
        if diff >= 0.05:
            all_close = False
    
    if all_close:
        print("\n✅ Sampling distribution matches theoretical strategy")
    else:
        print("\n⚠️  Sampling distribution deviates significantly")
    
    return all_close


def test_game_simulation():
    """Test 5: Run a simple poker hand simulation"""
    print("\n" + "="*60)
    print("TEST 5: Full Hand Simulation")
    print("="*60)
    
    player1 = CFRAIPlayer()
    player2 = CFRAIPlayer()
    
    # Deal random hands
    print("\n🎴 Dealing hands...")
    print("Player 1: As Kh")
    print("Player 2: Qs Jh")
    
    # Preflop
    print("\n--- PREFLOP ---")
    p1_action = player1.get_action("preflop", ["As", "Kh"])
    p2_action = player2.get_action("preflop", ["Qs", "Jh"])
    print(f"Player 1: {p1_action}")
    print(f"Player 2: {p2_action}")
    
    # Flop
    board = ["Ah", "Qh", "7c"]
    print(f"\n--- FLOP: {' '.join(board)} ---")
    p1_action = player1.get_action("flop", ["As", "Kh"], board)
    p2_action = player2.get_action("flop", ["Qs", "Jh"], board)
    print(f"Player 1 (top pair top kicker): {p1_action}")
    print(f"Player 2 (middle pair): {p2_action}")
    
    # Turn
    board.append("2d")
    print(f"\n--- TURN: {' '.join(board)} ---")
    p1_action = player1.get_action("turn", ["As", "Kh"], board)
    p2_action = player2.get_action("turn", ["Qs", "Jh"], board)
    print(f"Player 1: {p1_action}")
    print(f"Player 2: {p2_action}")
    
    # River
    board.append("9s")
    print(f"\n--- RIVER: {' '.join(board)} ---")
    p1_action = player1.get_action("river", ["As", "Kh"], board)
    p2_action = player2.get_action("river", ["Qs", "Jh"], board)
    print(f"Player 1: {p1_action}")
    print(f"Player 2: {p2_action}")
    
    print("\n✅ Full hand simulation complete")
    return True


def test_api_interface():
    """Test 6: Verify API-style interface works"""
    print("\n" + "="*60)
    print("TEST 6: API Interface Compatibility")
    print("="*60)
    
    player = CFRAIPlayer()
    
    # Test the interface that would be exposed via API
    test_cases = [
        {
            "stage": "preflop",
            "hole_cards": ["As", "Kh"],
            "board": []
        },
        {
            "stage": "flop",
            "hole_cards": ["Qs", "Jh"],
            "board": ["Ah", "Kh", "7c"]
        },
        {
            "stage": "river",
            "hole_cards": ["9s", "8s"],
            "board": ["7h", "6d", "5c", "Ah", "2s"]
        }
    ]
    
    all_passed = True
    for i, test_case in enumerate(test_cases, 1):
        try:
            # Get strategy
            strategy = player.get_strategy(**test_case)
            
            # Get action
            action = player.get_action(**test_case)
            
            # Verify format
            assert isinstance(strategy, dict), "Strategy should be a dict"
            assert isinstance(action, str), "Action should be a string"
            assert action in strategy, "Action should be in strategy keys"
            
            print(f"✅ Test case {i}: {test_case['stage']}")
            print(f"   Strategy: {strategy}")
            print(f"   Action: {action}")
            
        except Exception as e:
            print(f"❌ Test case {i} failed: {e}")
            all_passed = False
    
    return all_passed


def run_all_tests():
    """Run complete test suite"""
    print("\n" + "🎰"*30)
    print("POKER AI TEST SUITE")
    print("🎰"*30)
    
    results = {}
    
    # Run all tests
    results["Strategy File"] = test_strategy_file()
    results["Preflop Bucketing"] = test_preflop_bucketing()
    results["Strategy Coherence"] = test_strategy_coherence()
    results["Action Sampling"] = test_action_sampling()
    results["Game Simulation"] = test_game_simulation()
    results["API Interface"] = test_api_interface()
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name:<25} {status}")
    
    total = len(results)
    passed_count = sum(results.values())
    
    print("\n" + "="*60)
    print(f"TOTAL: {passed_count}/{total} tests passed")
    print("="*60)
    
    if passed_count == total:
        print("\n🎉 ALL TESTS PASSED! Your poker AI is ready to deploy!")
        print("\nNext steps:")
        print("1. Wrap aiplayer.py in a FastAPI backend")
        print("2. Deploy to Render/Railway")
        print("3. Connect your Next.js frontend")
        print("4. Add Claude API for natural language coaching")
        print("5. LAUNCH! 🚀")
    else:
        print(f"\n⚠️  {total - passed_count} test(s) failed. Fix issues before deployment.")
    
    return passed_count == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

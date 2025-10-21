"""
Test the new CFR+ backend with sample hands
"""
import requests
import json

BASE_URL = "http://localhost:8000"

def test_hand(hand, position, action_facing, description):
    """Test a specific hand scenario"""
    print(f"\n{'='*70}")
    print(f"TEST: {description}")
    print(f"{'='*70}")

    payload = {
        "stage": "preflop",
        "hole_cards": hand,
        "position": position,
        "action_facing": action_facing,
        "stack_depth": 100
    }

    try:
        response = requests.post(f"{BASE_URL}/analyze-hand", json=payload)
        data = response.json()

        if data['success']:
            gto = data['gto']
            print(f"\n✅ HAND: {' '.join(hand)}")
            print(f"   Position: {position} | Action: {action_facing}")
            print(f"\n📊 GTO STRATEGY:")
            print(f"   Recommended: {gto['recommended_action'].upper()}")
            print(f"   Fold:  {gto['strategy']['fold']:.1%}")
            print(f"   Call:  {gto['strategy']['call']:.1%}")
            print(f"   Raise: {gto['strategy']['raise']:.1%}")
            print(f"   Category: {gto['bucket']}")
            print(f"   From Trained Model: {'✅' if gto.get('from_trained_model') else '❌ (Fallback)'}")

            # print(f"\n🎯 COACHING:")
            # print(data['coaching'][:200] + "...")  # First 200 chars

        else:
            print(f"❌ Error: {data}")

    except Exception as e:
        print(f"❌ Request failed: {e}")


def run_tests():
    """Run comprehensive tests"""
    print("\n" + "="*70)
    print("TESTING CFR+ BACKEND V4.0")
    print("="*70)

    # Check health
    try:
        health = requests.get(f"{BASE_URL}/health").json()
        print(f"\n🏥 HEALTH CHECK:")
        print(f"   Status: {health['status']}")
        print(f"   Version: {health['version']}")
        print(f"   Strategies Loaded: {health['strategies_loaded']}")
        print(f"   Trained States: {health['num_trained_states']:,}")
        print(f"   Claude AI: {'✅ Enabled' if health['claude_ai_enabled'] else '❌ Disabled'}")
    except Exception as e:
        print(f"❌ Backend not running: {e}")
        print(f"   Start with: python3 fastapi_backend_v4_cfr_plus.py")
        return

    # Test Cases
    tests = [
        {
            "hand": ["As", "Ah"],
            "position": "BTN",
            "action_facing": "UNOPENED",
            "description": "AA on Button (should raise ~95%)"
        },
        {
            "hand": ["7h", "7s"],
            "position": "BB",
            "action_facing": "FACING_RAISE",
            "description": "77 in BB facing raise (should call for set mining)"
        },
        {
            "hand": ["As", "Ks"],
            "position": "CO",
            "action_facing": "UNOPENED",
            "description": "AKs in CO (should raise ~90%)"
        },
        {
            "hand": ["7s", "2s"],
            "position": "MP",
            "action_facing": "UNOPENED",
            "description": "72o in MP (should fold ~95%)"
        },
        {
            "hand": ["Qs", "Js"],
            "position": "BTN",
            "action_facing": "UNOPENED",
            "description": "QJs on Button (should raise ~85%)"
        }
    ]

    for test in tests:
        test_hand(
            hand=test['hand'],
            position=test['position'],
            action_facing=test['action_facing'],
            description=test['description']
        )

    print(f"\n{'='*70}")
    print("TESTS COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    run_tests()

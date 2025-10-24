"""
Generate ALL Defending Ranges for Complete Coverage
Includes: SB, BTN, CO defending vs raises from different positions
"""

import json

# Sklansky-Chubukov Rankings (1 = best, 169 = worst)
HAND_RANKINGS = {
    'AA': 1, 'KK': 2, 'QQ': 3, 'JJ': 4, 'AKs': 5, 'TT': 6, 'AQs': 7, 'AJs': 8,
    '99': 9, 'AKo': 10, 'ATs': 11, '88': 12, 'KQs': 13, 'AQo': 14, 'KJs': 15,
    '77': 16, 'KTs': 17, 'AJo': 18, 'QJs': 19, '66': 20, 'KQo': 21, 'ATo': 22,
    'QTs': 23, 'KJo': 24, 'A9s': 25, '55': 26, 'JTs': 27, 'A8s': 28, 'KTo': 29,
    'QJo': 30, 'A7s': 31, 'K9s': 32, '44': 33, 'A9o': 34, 'QTo': 35, 'A6s': 36,
    'JTo': 37, 'K8s': 38, '33': 39, 'A5s': 40, 'Q9s': 41, 'K9o': 42, 'A8o': 43,
    'J9s': 44, 'A4s': 45, 'QJo': 46, 'K7s': 47, 'A7o': 48, '22': 49, 'A3s': 50,
    'K8o': 51, 'T9s': 52, 'Q8s': 53, 'A6o': 54, 'A2s': 55, 'K6s': 56, 'J8s': 57,
    'Q9o': 58, 'K7o': 59, 'A5o': 60, 'K5s': 61, 'T8s': 62, 'J9o': 63, 'Q7s': 64,
    'K6o': 65, 'A4o': 66, 'K4s': 67, '98s': 68, 'J7s': 69, 'K5o': 70, 'Q8o': 71,
    'K3s': 72, 'T9o': 73, 'A3o': 74, 'K2s': 75, 'Q6s': 76, 'J8o': 77, 'T7s': 78,
    '97s': 79, 'K4o': 80, 'Q7o': 81, 'J6s': 82, '87s': 83, 'A2o': 84, 'Q5s': 85,
    '98o': 86, 'K3o': 87, 'T8o': 88, 'J7o': 89, 'Q4s': 90, 'T6s': 91, 'K2o': 92,
    '96s': 93, 'Q6o': 94, '86s': 95, 'Q3s': 96, 'J5s': 97, '97o': 98, 'T7o': 99,
    '76s': 100, 'Q2s': 101, 'J4s': 102, '87o': 103, 'Q5o': 104, 'T5s': 105,
    '95s': 106, 'J6o': 107, 'T6o': 108, '85s': 109, 'Q4o': 110, 'J3s': 111,
    '96o': 112, 'T4s': 113, '75s': 114, '86o': 115, 'J2s': 116, 'Q3o': 117,
    '65s': 118, '95o': 119, 'T5o': 120, '94s': 121, 'Q2o': 122, '76o': 123,
    'J4o': 124, '84s': 125, 'T3s': 126, '54s': 127, '85o': 128, 'J3o': 129,
    '74s': 130, 'T2s': 131, '75o': 132, '64s': 133, '94o': 134, 'T4o': 135,
    '93s': 136, '65o': 137, 'J2o': 138, '53s': 139, '84o': 140, '43s': 141,
    '73s': 142, '54o': 143, 'T3o': 144, '63s': 145, '83s': 146, '74o': 147,
    '52s': 148, '64o': 149, 'T2o': 150, '42s': 151, '62s': 152, '93o': 153,
    '53o': 154, '32s': 155, '73o': 156, '82s': 157, '43o': 158, '63o': 159,
    '72s': 160, '52o': 161, '62o': 162, '42o': 163, '92s': 164, '32o': 165,
    '82o': 166, '72o': 167, '92o': 168
}

def generate_defense_range(hero_pos, villain_pos, target_vpip, has_position=False):
    """
    Generate defending range for hero vs villain raise.

    Args:
        hero_pos: Position of hero (SB, BTN, CO, etc.)
        villain_pos: Position of villain raiser
        target_vpip: Target VPIP percentage
        has_position: True if hero has position on villain
    """
    ranges = {}
    sorted_hands = sorted(HAND_RANKINGS.items(), key=lambda x: x[1])
    target_hands = int(169 * (target_vpip / 100))

    for hand, rank in sorted_hands:
        if rank <= target_hands:
            # In defense range
            is_suited = 's' in hand
            is_pair = len(hand) == 2

            # Premium hands (AA-JJ, AK) - always 3bet heavy
            if rank <= 10:
                fold_freq = 0.02
                if has_position:
                    call_freq = 0.15  # More calls with position
                    raise_freq = 0.83
                else:
                    call_freq = 0.08
                    raise_freq = 0.90

            # Strong hands (TT-77, AQ-AT suited, top broadway)
            elif rank <= 25:
                fold_freq = 0.02
                if has_position:
                    call_freq = 0.80  # Prefer calls with position
                    raise_freq = 0.18
                else:
                    call_freq = 0.73
                    raise_freq = 0.25

            # Playable hands (66-22, suited connectors, Ax suited)
            elif rank <= 60:
                fold_freq = 0.05
                if has_position:
                    call_freq = 0.92  # Almost always call with position
                    raise_freq = 0.03
                else:
                    call_freq = 0.90
                    raise_freq = 0.05

            # Marginal hands
            elif rank <= target_hands:
                if is_suited or is_pair:
                    if has_position:
                        fold_freq = 0.15
                        call_freq = 0.82
                        raise_freq = 0.03
                    else:
                        fold_freq = 0.20
                        call_freq = 0.75
                        raise_freq = 0.05
                else:
                    if has_position:
                        fold_freq = 0.30
                        call_freq = 0.67
                        raise_freq = 0.03
                    else:
                        fold_freq = 0.40
                        call_freq = 0.57
                        raise_freq = 0.03
        else:
            # Outside defense range
            fold_freq = 0.96
            call_freq = 0.03
            raise_freq = 0.01

        ranges[hand] = [
            round(fold_freq, 2),
            round(call_freq, 2),
            round(raise_freq, 2)
        ]

    # Calculate actual VPIP
    actual_vpip = sum(1 for hand in ranges.values() if (hand[1] + hand[2]) > 0.5) / 169 * 100
    return ranges, actual_vpip

print("Generating ALL Defending Ranges...")
print("=" * 60)

all_defense_ranges = {}

# SB defending ranges (OOP, but good pot odds)
scenarios = [
    ('SB_vs_BTN', 'SB', 'BTN', 42, False),  # OOP but BTN is wide
    ('SB_vs_CO', 'SB', 'CO', 38, False),
    ('SB_vs_MP', 'SB', 'MP', 35, False),
    ('SB_vs_EP', 'SB', 'EP', 32, False),

    # BTN defending ranges (IP - has position advantage)
    ('BTN_vs_CO', 'BTN', 'CO', 40, True),   # IP advantage
    ('BTN_vs_MP', 'BTN', 'MP', 35, True),
    ('BTN_vs_EP', 'BTN', 'EP', 30, True),

    # CO defending ranges (IP vs early positions)
    ('CO_vs_MP', 'CO', 'MP', 35, True),
    ('CO_vs_EP', 'CO', 'EP', 28, True),

    # MP defending ranges
    ('MP_vs_EP', 'MP', 'EP', 25, True),
]

for range_key, hero, villain, target, has_pos in scenarios:
    ranges, actual_vpip = generate_defense_range(hero, villain, target, has_pos)
    all_defense_ranges[range_key] = {
        'comment': f'{hero} vs {villain} raise - Actual VPIP: {actual_vpip:.1f}%',
        'ranges': ranges
    }
    pos_str = "IP" if has_pos else "OOP"
    print(f"{range_key}: {actual_vpip:.1f}% (target {target}%) [{pos_str}]")

print("\n" + "=" * 60)
print("Sample Test Cases:")
print("=" * 60)

# Test some sample hands
test_hands = ['AA', '77', '87s', 'K9o', '92s']

for range_key in ['SB_vs_BTN', 'BTN_vs_CO', 'CO_vs_EP']:
    print(f"\n{range_key}:")
    range_data = all_defense_ranges[range_key]['ranges']
    for hand in test_hands:
        fold, call, raise_freq = range_data[hand]
        print(f"  {hand:4s}: fold={fold*100:4.0f}%, call={call*100:4.0f}%, 3bet={raise_freq*100:4.0f}%")

# Load existing precomputed ranges and merge
with open('precomputed_ranges.json', 'r') as f:
    precomputed = json.load(f)

# Add all defense ranges to the 6max_cash_100bb scenario
for key, value in all_defense_ranges.items():
    precomputed['6max_cash_100bb'][key] = value

# Save updated ranges
with open('precomputed_ranges.json', 'w') as f:
    json.dump(precomputed, f, indent=2)

print(f"\n✅ Merged all defense ranges into precomputed_ranges.json")
print(f"Total scenarios in 6max_cash_100bb: {len(precomputed['6max_cash_100bb'])}")

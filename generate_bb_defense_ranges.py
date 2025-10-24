"""
Generate BB Defending Ranges vs Raises from Different Positions
Uses Sklansky-Chubukov rankings + pot odds considerations
"""

import json

# Sklansky-Chubukov Rankings (1 = best, 169 = worst)
# These are objectively correct preflop hand strength rankings
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

def generate_bb_defense_range(villain_position, target_vpip):
    """
    Generate BB defending range vs raise from villain_position.

    BB has good pot odds but is OOP (out of position).
    Defending frequency depends on:
    - Villain's position (tighter vs EP, wider vs BTN)
    - Pot odds (getting ~2.5:1 or better)
    - Playability OOP
    """
    ranges = {}

    # Sort hands by ranking
    sorted_hands = sorted(HAND_RANKINGS.items(), key=lambda x: x[1])

    # Calculate how many hands to include based on target VPIP
    target_hands = int(169 * (target_vpip / 100))

    for hand, rank in sorted_hands:
        if rank <= target_hands:
            # In defense range
            # Determine if we should call, raise (3bet), or mix

            # Top tier hands (AA-JJ, AK) - always 3bet heavy
            if rank <= 10:
                fold_freq = 0.02
                call_freq = 0.08
                raise_freq = 0.90  # 3betting for value

            # Strong hands (TT-77, AQ-AT suited, top broadway) - mix of call and 3bet
            elif rank <= 25:
                fold_freq = 0.02
                call_freq = 0.73
                raise_freq = 0.25  # Some 3bets for value/balance

            # Playable hands (66-22, suited connectors, Ax suited) - mostly call
            # BB gets great pot odds, so we defend wider
            elif rank <= 60:
                fold_freq = 0.05
                call_freq = 0.90
                raise_freq = 0.05  # Occasional 3bet bluffs

            # Marginal hands - call or fold mix, some bluff 3bets
            elif rank <= target_hands:
                # Marginal hands: mix of call and fold based on playability
                # Suited hands and connectors get more calls (better playability OOP)
                is_suited = 's' in hand
                is_pair = len(hand) == 2

                if is_suited or is_pair:
                    fold_freq = 0.20
                    call_freq = 0.75
                    raise_freq = 0.05  # Some bluff 3bets
                else:
                    fold_freq = 0.40
                    call_freq = 0.57
                    raise_freq = 0.03
            else:
                # Should not reach here
                fold_freq = 0.96
                call_freq = 0.03
                raise_freq = 0.01
        else:
            # Outside defense range - mostly fold
            fold_freq = 0.96
            call_freq = 0.03
            raise_freq = 0.01

        # Store as array [fold, call, raise]
        ranges[hand] = [
            round(fold_freq, 2),
            round(call_freq, 2),
            round(raise_freq, 2)
        ]

    # Calculate actual VPIP
    actual_vpip = sum(1 for hand in ranges.values() if (hand[1] + hand[2]) > 0.5) / 169 * 100

    return ranges, actual_vpip

# Generate BB defending ranges vs different positions
print("Generating BB Defending Ranges...")
print("=" * 60)

bb_defense_ranges = {}

# BB vs EP raise - tightest defense (EP has strong range)
bb_defense_ranges['BB_vs_EP'] = {
    'comment': f'BB vs EP raise - Actual VPIP: TBD',
    'ranges': generate_bb_defense_range('EP', target_vpip=38)[0]
}
actual_vpip = generate_bb_defense_range('EP', target_vpip=38)[1]
bb_defense_ranges['BB_vs_EP']['comment'] = f'BB vs EP raise - Actual VPIP: {actual_vpip:.1f}%'
print(f"BB vs EP: {actual_vpip:.1f}% (target 38%)")

# BB vs MP raise
bb_defense_ranges['BB_vs_MP'] = {
    'comment': f'BB vs MP raise - Actual VPIP: TBD',
    'ranges': generate_bb_defense_range('MP', target_vpip=42)[0]
}
actual_vpip = generate_bb_defense_range('MP', target_vpip=42)[1]
bb_defense_ranges['BB_vs_MP']['comment'] = f'BB vs MP raise - Actual VPIP: {actual_vpip:.1f}%'
print(f"BB vs MP: {actual_vpip:.1f}% (target 42%)")

# BB vs CO raise
bb_defense_ranges['BB_vs_CO'] = {
    'comment': f'BB vs CO raise - Actual VPIP: TBD',
    'ranges': generate_bb_defense_range('CO', target_vpip=46)[0]
}
actual_vpip = generate_bb_defense_range('CO', target_vpip=46)[1]
bb_defense_ranges['BB_vs_CO']['comment'] = f'BB vs CO raise - Actual VPIP: {actual_vpip:.1f}%'
print(f"BB vs CO: {actual_vpip:.1f}% (target 46%)")

# BB vs BTN raise - widest defense (BTN is stealing wide)
bb_defense_ranges['BB_vs_BTN'] = {
    'comment': f'BB vs BTN raise - Actual VPIP: TBD',
    'ranges': generate_bb_defense_range('BTN', target_vpip=52)[0]
}
actual_vpip = generate_bb_defense_range('BTN', target_vpip=52)[1]
bb_defense_ranges['BB_vs_BTN']['comment'] = f'BB vs BTN raise - Actual VPIP: {actual_vpip:.1f}%'
print(f"BB vs BTN: {actual_vpip:.1f}% (target 52%)")

print("\n" + "=" * 60)
print("Sample Test Cases:")
print("=" * 60)

# Test some sample hands
test_hands = ['AA', '77', '87s', 'K9o', '92s', '72o']

for pos in ['EP', 'MP', 'CO', 'BTN']:
    print(f"\nBB vs {pos} raise:")
    range_data = bb_defense_ranges[f'BB_vs_{pos}']['ranges']
    for hand in test_hands:
        fold, call, raise_freq = range_data[hand]
        print(f"  {hand:4s}: fold={fold*100:4.0f}%, call={call*100:4.0f}%, 3bet={raise_freq*100:4.0f}%")

# Save to file
output_file = 'bb_defense_ranges.json'
with open(output_file, 'w') as f:
    json.dump(bb_defense_ranges, f, indent=2)

print(f"\n✅ Saved to {output_file}")

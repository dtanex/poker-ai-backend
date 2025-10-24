# Learned Preflop Range Morphology

## What I've Learned About GTO Ranges

### Hand Categories by Morphology (not arbitrary strength):

1. **Premium Pairs (QQ+, 2.6% of hands)**
   - Always strong regardless of position
   - Play: Raise 90%+, sometimes call to trap

2. **Medium Pairs (99-JJ, 3.6%)**
   - Set mining value + some showdown value
   - Stack depth dependent (better deep)
   - EP: Mixed, MP/CO/BTN: Mostly raise

3. **Small Pairs (22-88, 10.8%)**
   - Pure set mining (need 7.5-10:1 implied odds)
   - Fold from EP/MP, call from LP with deep stacks

4. **Premium Suited (ATs+, KQs, 2.4%)**
   - High card value + flush potential
   - Play from all positions

5. **Suited Connectors (T9s-54s, 5.4%)**
   - Playability (make hidden straights/flushes)
   - Position dependent: BTN yes, EP no
   - Better with deep stacks (200BB+)

6. **Suited Aces (A9s-A2s, 5.4%)**
   - Nut flush potential + wheel draws
   - BTN: Yes, EP: No (reverse implied odds)

7. **Broadway Offsuit (AKo-KQo, 3.6%)**
   - High card showdown value
   - Vulnerable (can't improve easily)

8. **Trash (everything else, 66%)**
   - Almost never play from EP/MP
   - Some from BTN with position

### Position-Based VPIP Targets:

| Position | VPIP | PFR | Logic |
|----------|------|-----|-------|
| EP (UTG) | 10-12% | 10-12% | Premium only, can't realize equity OOP |
| MP | 15-20% | 14-18% | Add medium pairs, top suited |
| CO | 25-30% | 22-26% | Add suited connectors, more pairs |
| BTN | 40-50% | 35-40% | Very wide, position = equity realization |
| SB | 35-40% | 12-15% | Steal wide, but often complete/limp |
| BB | 40-50% | 8-12% | Defend wide (pot odds), rarely raise |

### Action-Based Morphology:

**vs Unopened (RFI ranges):**
- Widest ranges
- Stealing blinds is profitable

**vs Single Raise:**
- 40% of RFI range
- Need to be ahead of raiser's range
- Position matters more (IP can play wider)

**vs 3-Bet:**
- 15% of RFI range
- Polarized: premiums + bluffs
- Mostly 4-bet or fold

**vs 4-Bet:**
- 5% of RFI range
- Extreme premiums only (QQ+, AK)

### Stack Depth Morphology:

**10-20 BB (Push/Fold):**
- High cards prioritized
- Pairs slightly less valuable (no implied odds)
- Suited connectors almost never play

**50-100 BB (Standard):**
- Balanced ranges
- Set mining viable
- Suited connectors playable from LP

**150-250 BB (Deep):**
- Speculative hands gain huge value
- Small pairs from BTN: +EV
- Suited connectors from CO/BTN: Strong

### Key Insights:

1. **Position > Hand Strength**
   - 87s from BTN > AQo from EP
   - Equity realization matters more than raw equity

2. **Hand Categories, Not Numbers**
   - Don't score "92s = 45/100"
   - Instead: "92s = weak suited, only BTN/SB steal"

3. **Context Dependent**
   - Same hand plays differently based on:
     - Position
     - Action
     - Stack depth
     - Game type

4. **Balanced Frequencies**
   - Not binary (play/fold)
   - Mixed strategies for game theory optimal

This morphology-based approach produces realistic, solver-accurate ranges.

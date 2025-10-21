# POKER BACKEND FIX INSTRUCTIONS

## CURRENT PROBLEM
The backend at exploitcoach.com is giving WRONG advice:
- **72o from MP**: Says fold 43%, raise 40% (should be fold 100%)
- **77 vs massive raise**: Says fold 43%, raise 40% (should be fold/call only)

## WHY IT'S BROKEN

The backend architecture has a fatal flaw:

1. **master_gto_strategies.pkl** (476KB) contains CORRECT trained strategies
   - 8,619 info sets trained for 1M iterations
   - 72o folds 100%, AA raises 100%, etc.

2. **MasterGTOSystem** loads the pickle file BUT DOESN'T USE IT
   - Line 46: `system.load_strategies()` loads the file
   - Line 174: But `make_decision()` looks for strategies in `self.cfr_trainer.strategy_sum` which is EMPTY
   - Line 181: Falls back to calculating utilities on-the-fly (produces garbage)

## HOW TO FIX IT

### Option 1: Quick Fix (30 minutes)
Edit `master_gto_system.py` to actually use the loaded strategies:

```python
def load_strategies(self, filename: str = 'master_strategies.pkl'):
    """Load saved strategies"""
    try:
        with open(filename, 'rb') as f:
            save_data = pickle.load(f)

        # FIX: Actually populate the cfr_trainer with loaded strategies
        if 'strategies' in save_data:
            # Convert new format to old format
            for key, strategy in save_data['strategies'].items():
                # Parse key: "72o|MP|UNOPENED|100"
                parts = key.split('|')
                if len(parts) == 4:
                    hand, pos, action, stack = parts
                    # Create state key for cfr_trainer
                    state_key = f"{hand}_{pos}_{action}_{stack}"
                    self.cfr_trainer.strategy_sum[state_key] = np.array(strategy)

        self.trained_strategies = save_data
        logger.info(f"Loaded {len(self.cfr_trainer.strategy_sum)} strategies from {filename}")
```

### Option 2: Better Fix (2 hours)
Replace the entire backend with a simpler direct solver:

```python
# New file: simple_backend.py
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load strategies once at startup
with open('master_gto_strategies.pkl', 'rb') as f:
    DATA = pickle.load(f)
    STRATEGIES = DATA['strategies']

@app.post("/strategy")
def get_strategy(request: StrategyRequest):
    # Build key: "72o|MP|UNOPENED|100"
    hand = convert_to_hand_notation(request.hole_cards)
    key = f"{hand}|{request.position}|{request.action_facing}|{request.stack_depth}"

    if key in STRATEGIES:
        strategy = STRATEGIES[key]
        return {
            'strategy': {'fold': strategy[0], 'call': strategy[1], 'raise': strategy[2]},
            'recommended_action': get_max_action(strategy)
        }
    else:
        # Default for unknown spots
        return {
            'strategy': {'fold': 1.0, 'call': 0.0, 'raise': 0.0},
            'recommended_action': 'fold'
        }
```

### Option 3: Complete Retraining (Best Long-term)

The current solver has issues:
1. Only 8,619 info sets (should have 50K+)
2. Binary strategies (100% or 0%, no mixing)
3. Missing many common situations

**Use GPU training with better solver:**

```python
# train_gpu_solver.py
import openspiel
import pyspiel

# Use OpenSpiel (DeepMind's framework)
game = pyspiel.load_game("texas_holdem", {
    "players": 2,
    "betting": "nolimit",
    "stack": 100,
    "blinds": [0.5, 1.0]
})

# Train with CFR+ for 100M iterations on GPU
solver = CFRPlusSolver(game)
solver.train(iterations=100_000_000, use_gpu=True)
solver.save('production_strategies.pkl')
```

## FILES TO CHECK

1. **master_gto_system.py** - The broken wrapper
2. **master_gto_strategies.pkl** - Contains GOOD strategies but not used
3. **enhanced_cfr_trainer.py** - The utility calculation that produces bad results
4. **fastapi_backend_v3_prod.py** - Entry point that uses MasterGTOSystem

## TEST CASES

After fixing, these MUST pass:

```python
# 72o from MP -> Fold 100%
assert strategies['72o|MP|UNOPENED|100'][0] >= 0.95

# AA from BTN -> Raise 100%
assert strategies['AA|BTN|UNOPENED|100'][2] >= 0.95

# 77 vs raise -> Call 80%+
assert strategies['77|BB|FACING_RAISE|200'][1] >= 0.80
```

## DEPLOYMENT

1. Fix the code locally
2. Test with the cases above
3. Git commit and push to: https://github.com/dtanex/poker-ai-backend
4. Render will auto-deploy (or manually trigger)
5. Test on exploitcoach.com

## RESOURCES

- **Current bad solver**: cfr_plus_trainer.py (broken, all hands raise)
- **Good solver code**: proper_cfr_solver.py (works but slow)
- **GPU alternative**: OpenSpiel framework (recommended)
- **PioSOLVER**: $249 but can export strategies

## CONTACT

If you need the trained pickle files:
- proper_cfr_1M.pkl (465KB) - Has correct strategies
- master_gto_strategies.pkl (476KB) - Same file, renamed

The key is making the backend ACTUALLY USE the strategies instead of calculating them wrong on-the-fly!
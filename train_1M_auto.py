"""Production Training - 1 Million Iterations (Auto-run)"""
import logging
from master_gto_system import MasterGTOSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("=" * 70)
print("🎰 PRODUCTION GTO TRAINING - 1,000,000 ITERATIONS")
print("=" * 70)
print()
print("This will take approximately 2-3 hours on M1 Mac")
print("Training proper GTO strategies for production use")
print()

# Initialize
logger.info("Initializing Master GTO System...")
system = MasterGTOSystem()

# Train
logger.info("Starting 1M iteration training...")
print()
system.train_from_scratch(iterations=1000000)

# Save
logger.info("Saving production strategies...")
system.save_strategies('master_gto_strategies_1M.pkl')

# Also save to default name for backend to load
system.save_strategies('master_gto_strategies.pkl')

print()
print("=" * 70)
print("✅ PRODUCTION TRAINING COMPLETE!")
print("=" * 70)
print()
print("Strategies saved to:")
print("  - master_gto_strategies_1M.pkl (backup)")
print("  - master_gto_strategies.pkl (for backend)")
print()
print("🚀 Ready to deploy to Render!")
print()

"""Quick Training Script - 10K iterations for fast testing"""
import logging
from master_gto_system import MasterGTOSystem

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("=" * 70)
print("🎰 QUICK TRAINING - 10,000 iterations")
print("=" * 70)
print()

# Initialize system
logger.info("Initializing Master GTO System...")
system = MasterGTOSystem()

# Train
logger.info("Training with 10,000 iterations...")
print()
system.train_from_scratch(iterations=10000)

# Save
logger.info("Saving strategies...")
system.save_strategies('master_gto_strategies.pkl')

print()
print("=" * 70)
print("✅ TRAINING COMPLETE!")
print("=" * 70)
print()
print("Strategies saved to: master_gto_strategies.pkl")
print("Restart the backend to load the new strategies.")
print()

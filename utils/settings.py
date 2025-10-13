USE_TEMP_SCALING = True
AVG_TWO_SEEDS_INNER = True
OUTER_SEEDS = 3
USE_AP_SELECTION = True
LABEL_SMOOTH_EPS = 0.05
EXPLAIN_CLASS = "active"  # or "SHAM"
USE_LOGIT_DELTAS_CLASSICAL = True
from config import CFG
RANDOM_SEED = getattr(CFG, "random_seed", getattr(getattr(CFG, "training", object()), "random_seed", 42))

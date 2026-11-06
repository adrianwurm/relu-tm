import os
import sys
from pathlib import Path

from hypothesis import HealthCheck, settings

sys.path.insert(0, str(Path(__file__).parent))  # make ``strategies`` importable

settings.register_profile("dev", max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
settings.register_profile(
    "ci", max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow], derandomize=True
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

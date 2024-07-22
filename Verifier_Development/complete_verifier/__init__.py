import sys
import os

print("Adding complete_verifier to sys.path")
sys.path = [os.path.dirname(__file__)] + sys.path

from abcrown import ABCROWN  # pylint: disable=wrong-import-position

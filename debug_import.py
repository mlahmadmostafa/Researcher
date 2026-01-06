import sys
import os

print(f"CWD: {os.getcwd()}")
sys.path.append(os.getcwd())

try:
    from src.models import PaperRelevance
    print("Import src.models: OK")
except Exception as e:
    print(f"Import src.models FAILED: {e}")

try:
    from src.agents import ResearchCrew
    print("Import src.agents: OK")
except Exception as e:
    print(f"Import src.agents FAILED: {e}")

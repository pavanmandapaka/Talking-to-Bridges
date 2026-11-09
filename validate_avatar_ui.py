"""
Validation script for AVT-4: Lip-sync and Avatar Refinement (Krishna).
Executes all unit tests and runtime checks for the refined avatar presentation.
"""

import sys
import unittest
from pathlib import Path


def run_validation():
    print("==================================================")
    print("*    TALKING TO BRIDGES - AVT-4 AVATAR VALIDATION *")
    print("==================================================")
    
    project_root = Path(__file__).resolve().parent
    voice_ui_path = project_root / "frontend" / "voice_ui.html"
    
    if not voice_ui_path.exists():
        print(f"[FAIL] Missing frontend/voice_ui.html at {voice_ui_path}")
        sys.exit(1)
        
    print(f"[OK] Located frontend/voice_ui.html ({voice_ui_path.stat().st_size:,} bytes)")
    
    # Load and run the test suite
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(project_root / "tests"),
        pattern="test_avatar_refinement.py"
    )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    if not result.wasSuccessful():
        print("[FAIL] Avatar refinement tests failed.")
        sys.exit(1)
        
    print("\n[SUCCESS] All AVT-4 Avatar and Lip-Sync tests passed!")


if __name__ == "__main__":
    run_validation()


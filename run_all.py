"""
Run all 4 lab steps sequentially

Usage:
    python run_all.py              # Run all steps
    python run_all.py --step 1     # Run only step 1
    python run_all.py --step 4     # Run only step 4
"""

import sys
import time
import subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent

STEPS = {
    1: ("01_langsmith_rag_pipeline.py", "LangSmith RAG Pipeline"),
    2: ("02_prompt_hub_ab_routing.py", "Prompt Hub A/B Routing"),
    3: ("03_ragas_evaluation.py", "RAGAS Evaluation (15-25 min)"),
    4: ("04_guardrails_validator.py", "Guardrails Validators"),
}


def run_step(step_num: int) -> bool:
    """Run a single step and return success status"""
    if step_num not in STEPS:
        print(f"❌ Unknown step: {step_num}")
        return False
    
    script_file, description = STEPS[step_num]
    script_path = PROJECT_ROOT / script_file
    
    if not script_path.exists():
        print(f"❌ Script not found: {script_path}")
        return False
    
    print(f"\n{'=' * 70}")
    print(f"  Running Step {step_num}: {description}")
    print(f"{'=' * 70}")
    
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            check=False
        )
        
        if result.returncode == 0:
            print(f"\n✅ Step {step_num} completed successfully")
            return True
        else:
            print(f"\n❌ Step {step_num} failed with exit code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error running Step {step_num}: {e}")
        return False


def main():
    """Parse args and run steps"""
    steps_to_run = list(STEPS.keys())
    
    # Check for --step argument
    if len(sys.argv) > 1 and sys.argv[1] == "--step":
        if len(sys.argv) > 2:
            try:
                step_num = int(sys.argv[2])
                steps_to_run = [step_num]
            except ValueError:
                print(f"❌ Invalid step number: {sys.argv[2]}")
                sys.exit(1)
    
    # Run steps
    start_time = time.time()
    results = {}
    
    for step_num in steps_to_run:
        results[step_num] = run_step(step_num)
        time.sleep(1)
    
    elapsed = time.time() - start_time
    
    # Summary
    print(f"\n{'=' * 70}")
    print("  Summary")
    print(f"{'=' * 70}")
    
    for step_num in steps_to_run:
        status = "✅" if results[step_num] else "❌"
        print(f"  Step {step_num}: {status} {STEPS[step_num][1]}")
    
    print(f"\nTotal time: {elapsed / 60:.1f} minutes")
    
    # Exit code
    if all(results.values()):
        print(f"\n✅ All steps completed successfully!")
        sys.exit(0)
    else:
        print(f"\n❌ Some steps failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

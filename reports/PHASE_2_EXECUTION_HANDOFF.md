# PHASE 2 EXECUTION HANDOFF

**Target Working Directory:** `C:\Users\ASUS\Desktop\cv`  
**Activated Python Environment:** Python 3.14.7, PyTorch 2.11.0+cu128  

---

## 1. Fast Verification Commands (Autonomous Smoke Tests Completed)

### Validate Dataset Manifest Integrity
```powershell
python scripts/validate_dataset_manifest.py
```
*Expected Output:*
`SUCCESS: All dataset manifest integrity checks PASSED perfectly.` (Exit code 0).

### Run Complete Automated Test Suite (121 Tests)
```powershell
python -m pytest tests/
```
*Expected Output:*
`121 passed in ~11s` (Exit code 0).

---

## 2. Resource-Intensive Benchmark Evaluation (Optional User Execution)

To evaluate all 8 legacy models and Bicubic baselines across Set5, Set14, and BSD100 on GPU:
```powershell
python evaluate_benchmarks.py --dataset all --scale 0 --model all --device cuda --export-json results/canonical_benchmark_results.json --export-csv results/canonical_summary.csv
```
- **Execution Time:** ~45 seconds on RTX 2050 GPU.
- **Outputs Produced:**
  - `results/canonical_benchmark_results.json`
  - `results/canonical_summary.csv`
- **Output Files to Return:** `results/canonical_summary.csv`

---

## 3. Preparation for Phase 3 (Training & Controlled Experiments)

In accordance with the **External Dataset & Download Policy**, no external datasets have been downloaded.

When ready to begin Phase 3, you may download the official DIV2K training dataset:
- **Official Source:** [DIV2K Official Dataset (NTIRE Challenge)](https://data.vision.ee.ethz.ch/cvl/DIV2K/)
- **Required File:** `DIV2K_train_HR.zip` (800 high-resolution images, ~3.5 GB)
- **Local Extraction Target:** `data/raw/DIV2K_train_HR/` (containing `0001.png` through `0800.png`)
- **Note:** External download is purely optional. Phase 3 can also proceed with clean augmented T91 (80 train / 11 val).

# Wi-Sense CSI Results (Modal H100)

This file summarizes completed runs for next-day presentation.

## Task Definition
- Target: `type`
- Binary classes: `organic` vs `metalic`
- Input: 52 amplitude + 52 phase CSI features reshaped to `(2, 52)`

## Experiment A: Standard Split (i.i.d.)

| Model | Best Val Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---:|---:|---:|
| CNN1D | 0.7985 | 0.7918 | 0.7899 |
| CNN-BiLSTM | 0.9353 | 0.9397 | 0.9373 |

Modal runs:
- CNN1D: https://modal.com/apps/nikunjkumar1405/main/ap-MOC5QFOYr2mvLZ2pirgZY8
- CNN-BiLSTM: https://modal.com/apps/nikunjkumar1405/main/ap-A5QjZcyKWdEjBsuTPrnZOo

Stored canonical metrics and downloaded artifacts:
- Metrics JSON: modal_results_standard.json
- Local model files:
	- outputs_modal/cnn1d_type_baseline/best_model.pt
	- outputs_modal/cnn_bilstm_type_baseline/best_model.pt

## Experiment B: Day-Shift Generalization
- Train: day=1
- Test: day=2

| Model | Best Val Macro-F1 | Test Accuracy | Test Macro-F1 |
|---|---:|---:|---:|
| CNN1D (day-shift) | 0.8050 | 0.5741 | 0.4927 |
| CNN-BiLSTM (day-shift) | 0.9240 | 0.5823 | 0.5561 |

Modal runs:
- CNN1D day-shift: https://modal.com/apps/nikunjkumar1405/main/ap-JBWGWCJEih315sq4iLwV2q
- CNN-BiLSTM day-shift: https://modal.com/apps/nikunjkumar1405/main/ap-RvmHAcR8hKPlj8WBDlxyOu

## Key Takeaways for Presentation
1. CNN-BiLSTM strongly improves i.i.d. performance over CNN1D.
2. Day-shift evaluation reveals significant domain shift for both models.
3. CNN-BiLSTM is still more robust than CNN1D under domain shift.
4. This supports the claim that sequence-aware modeling helps CSI recognition.

## Repro Commands
```bash
# Standard split
./scripts/run_modal.sh base.yaml 20 0
./scripts/run_modal.sh model_cnn_bilstm.yaml 20 0

# Day-shift split
./scripts/run_modal.sh model_cnn1d_day.yaml 20 0
./scripts/run_modal.sh model_cnn_bilstm_day.yaml 20 0
```

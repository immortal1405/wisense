# Wi-Sense PPT Outline (10 Slides)

## Slide 1: Title and Team
- Title: Wi-Sense: Contactless Recognition via Wi-Fi CSI
- Subtitle: End-to-End ML Pipeline with Interactive Inference Dashboard
- Developers:
  - Nikunj Kumar - 2023UCI6614
  - Pranav Garg - 2023UCI3622
  - Raghav Chugh - 2023UCI6597
  - Ankrish Sharma - 2023UCI6542
- Suggested visual:
  - Wi-Fi signal icon + pipeline thumbnail

## Slide 2: Problem Statement and Motivation
- Contactless sensing is useful where cameras/wearables are inconvenient.
- Wi-Fi CSI can capture environmental and object-induced signal changes.
- Project goal: classify material type from CSI (organic vs metallic).
- Why this matters:
  - Low-cost sensing
  - Privacy-friendly compared to vision
  - Useful base for smart environments and activity sensing
- Suggested visual:
  - One-line problem diagram: Wi-Fi signals -> CSI -> material prediction

## Slide 3: What We Built
- Complete ML workflow:
  - Data loading and preprocessing
  - Train/validation/test setup
  - Two deep models for comparison
  - Evaluation with accuracy and macro-F1
- Presentation-ready product:
  - React dashboard
  - NestJS backend API
  - Interactive inference lab
- Cloud training support using Modal H100.
- Suggested visual:
  - System architecture block diagram (Data -> Training -> API -> Web UI)

## Slide 4: Data and Input Representation
- CSI features used per sample:
  - 52 amplitude values: amp_0 to amp_51
  - 52 phase values: phase_0 to phase_51
- Final model input shape: 2 x 52
  - Channel 1: amplitude
  - Channel 2: phase
- Target used for primary demo:
  - Material class: organic or metallic
- Suggested visual:
  - Example row transformed into 2-channel tensor

## Slide 5: Models Implemented
- Model 1: CNN1D (baseline)
  - Learns local subcarrier patterns quickly
- Model 2: CNN-BiLSTM (improved)
  - CNN extracts local features
  - BiLSTM captures sequence dependencies across subcarriers
- Training setup (high level):
  - Cross-entropy loss
  - AdamW optimizer
  - Early stopping based on validation behavior
- Suggested visual:
  - Side-by-side architecture comparison (CNN1D vs CNN-BiLSTM)

## Slide 6: Experiment Design
- Standard split experiment:
  - Train and test under similar distribution
- Day-shift (domain-shift) experiment:
  - Train on day 1, test on day 2
- Purpose:
  - Measure both normal performance and robustness under shift
- Metrics reported:
  - Accuracy
  - Macro-F1
- Suggested visual:
  - Split diagram: Day 1 -> Train, Day 2 -> Test

## Slide 7: Results Achieved
- Standard split:
  - CNN1D: Accuracy = 0.7918, Macro-F1 = 0.7899
  - CNN-BiLSTM: Accuracy = 0.9397, Macro-F1 = 0.9373
- Domain shift (day 1 -> day 2):
  - CNN1D: Accuracy = 0.5741, Macro-F1 = 0.4927
  - CNN-BiLSTM: Accuracy = 0.5823, Macro-F1 = 0.5561
- Net outcome:
  - Best standard run: CNN-BiLSTM (0.9397 Acc, 0.9373 Macro-F1)
  - Best day-shift run: CNN-BiLSTM (0.5823 Acc, 0.5561 Macro-F1)
- Key takeaway:
  - Sequence-aware modeling gives stronger and more stable performance
- Suggested visual:
  - Bar chart for 4 runs (CNN1D/CNN-BiLSTM x Standard/Day-shift)

## Slide 8: Interactive Inference and Explainability
- Inference modes in dashboard:
  - Random replay
  - Manual vector input
  - CSV upload
  - Scenario simulator
- Explainability:
  - Top contributing subcarriers via perturbation analysis
  - Base confidence vs perturbed confidence
- Demo insight:
  - Hard/noisy samples show confidence drop, highlighting robustness limits
- Suggested visual:
  - Screenshot of Inference tab + top subcarriers table

## Slide 9: Challenges and Learnings
- Challenges faced:
  - Domain shift reduced confidence and performance
  - CSI sensitivity to environment/time changes
  - Making technical outputs understandable for reviewers
- Learnings:
  - Stronger model architecture improves generalization
  - Human-interactive inference is critical for presentation value
  - Explainability helps justify predictions beyond raw scores
- Suggested visual:
  - Before/after confidence comparison (clean sample vs hard sample)

## Slide 10: Conclusion and Future Work
- Conclusion:
  - Successfully built an end-to-end contactless recognition prototype using Wi-Fi CSI.
  - Achieved strong standard performance and meaningful day-shift evaluation.
  - Delivered deployable demo with training, inference, and explainability.
- Future work:
  - Real-time CSI stream integration from hardware
  - Domain adaptation for better day-shift robustness
  - Expand to object identity/position multi-task setup
- Final line:
  - Wi-Sense demonstrates that wireless signals can drive practical, privacy-aware sensing systems.
- Suggested visual:
  - Final architecture + roadmap timeline

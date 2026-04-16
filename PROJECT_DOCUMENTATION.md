# Wi-Sense: Comprehensive Project Documentation

## 1. Introduction

### Explain Like I'm 5 (ELI5)
Imagine you are blindfolded in a room, but the room is filled with water. If someone walks in, they make ripples in the water. Even without seeing them, you can feel the ripples bouncing off the walls and hitting you. By feeling the size and shape of the ripples, you can guess if it's a small dog, a heavy person, or if someone just fell down! 

Wi-Sense does exactly this, but instead of water, it uses **Wi-Fi signals**. It "feels" how Wi-Fi waves bounce around the room to know what materials objects are made of, what activities people are doing, and if someone has fallen—all without any cameras!

### Technical Depth
Wi-Sense is an end-to-end Machine Learning pipeline and demonstrator that performs **Contactless Sensing using Wi-Fi Channel State Information (CSI)**. CSI provides fine-grained, physical-layer metrics describing how a wireless signal propagates from a transmitter to a receiver. 

By analyzing the amplitude and phase shifts across multiple OFDM (Orthogonal Frequency-Division Multiplexing) subcarriers, we can map the unique multipath reflections and distortions caused by human bodies and objects. This project covers three distinct tasks:
1. **Material Classification (Organic vs. Metallic)**
2. **Human Fall Detection (Binary Safety Alerting)**
3. **Multi-class Human Activity Recognition (HAR)**

---

## 2. The Tech Stack

### ELI5
We have a factory (the **Python ML** code) that learns how to understand the Wi-Fi ripples. We use a super-fast brain in the cloud (**Modal**) to do the heavy thinking. Then we have a waiter (**NestJS Backend**) who carries requests from the customer. Finally, a beautifully painted menu (**React Frontend**) lets the user interact with the system.

### Technical Depth
Why this stack?
* **Model Training (Python, PyTorch, Scikit-learn):** Python is the defacto standard for ML. PyTorch allows dynamic computation graphs and easy hardware acceleration for sequence models (LSTMs).
* **Remote Compute (Modal):** Training requires heavily parallelized ops and VRAM (H100 GPUs). Modal allows serverless containerized execution, keeping the local dev environment light while orchestrating vast data volumes.
* **Backend (NestJS / Node.js):** Acts as the API gateway. It receives HTTP requests from the UI, formats them, and spawns local Python processes (`execFileSync`) passing data via `stdin/stdout`. *Why?* Strict typing (TypeScript) ensures the JSON payloads perfectly match the Python inference expectations.
* **Frontend (React + Vite):** A fast, component-driven UI. We use pure SVG for sparklines (no heavy charting libraries needed) to render signal profiles and probability outputs dynamically.

---

## 3. The Data: Wi-Fi CSI

### ELI5
Standard Wi-Fi is like a big flashlight, but CSI is like a laser grid. Instead of just knowing "how bright" the light is, CSI tells us the color and angle of 50 different tiny laser beams.

### Technical Depth
We use two primary data flavors in this repository:
1. **Object Dataset (Material Classification):** Uses 52 subcarriers. Each snapshot is an array of 52 Amplitudes and 52 Phases.
2. **HAR / Fall Dataset (Mendeley/Kaggle v38wjmz6f6-1):** 
   - Collected across 3 indoor environments from 30 subjects performing 5 specific activities.
   - Environment 1 & 2 are **LOS (Line-of-Sight)**, meaning the space between the Wi-Fi transmitter and receiver is clear and unobstructed.
   - Environment 3 is **NLOS (Non-Line-of-Sight)**, meaning there are walls, furniture, or other physical obstacles between the devices.
   - **Hardware Setup:** 3 antennas at the Receiver, 1 at the Transmitter.
   - 30 subcarriers sampled per antenna.
   - Total raw channels = 1 (Tx) × 3 (Rx) × 30 (Subcarriers) = 90 complex numbers per temporal snapshot.
   - We extract `amp_log_sincos` features (Amplitude, sin(Phase), cos(Phase)) for each complex number, resulting in a feature tensor of 270 columns per sequence frame.

**Why are we training on LOS and testing on NLOS?**
This is a deliberate, difficult evaluation protocol designed to test *cross-domain robustness* and *generalization*. If we combined all environments (LOS + NLOS) into a single large dataset and randomly split it for training and testing, the task would be significantly easier (an *in-distribution* test). The model would simply memorize the specific structural reflections of the 3 rooms. 

However, in the real world, a sensing system must be deployed in a room it has *never* seen during training. By explicitly training only on Line-of-Sight data (Env 1 & 2) and testing strictly on unseen Non-Line-of-Sight data (Env 3), we impose a harsh but realistic domain-shift benchmark. It proves whether the model is learning the true biomechanical CSI distortions of a human falling/walking, or if it is just memorizing the room geometry.

**Why not just use Camera Video?**
Privacy preservation. Wi-Fi doesn't take photos, so it can be deployed in bathrooms and bedrooms. It also works in the dark and sees through some non-metallic walls.

---

## 4. Task 1: Material (Object) Classification

### Objective
Detect whether an object placed in the Wi-Fi field is Organic (wood, paper) or Metallic.

### Physical Principle (Why it works)
Organic materials (wood, paper, cloth, plastic) and Metallic objects (conductive metals) interact with Radio Frequency waves entirely differently. Metals generally create much stronger, sharper, hard-edged scatterings and reflections. Conversely, organic materials produce softer transmission changes and absorptions. By analyzing the 52 Amplitude and 52 Phase vectors of standard CSI, the Neural Network learns to map these physical propagation patterns to specific material classes. 

### Achievements & Workflow
We used a **1D-CNN (Local patterns)** compared against a **CNN-BiLSTM (Temporal/Sequence patterns across subcarriers)**.
* **Why CNN-BiLSTM?** CNN extracts spatial correlation among adjacent frequency subcarriers. The BiLSTM reads these feature maps sequentially to find broader relationships across the entire frequency band.
* **Result:** The CNN-BiLSTM outperformed the CNN1D significantly on standard testing but also maintained a better Accuracy and Macro-F1 lead during **Day-shift testing**. Training on Day 1 and forecasting on Day 2 deliberately introduces time-based domain shift.
* **Interactive Inference UI:** Instead of live hardware capture, we built 4 inference modes for human interaction: Random Replay, Manual Vector Input, CSV Upload, and Scenario Simulation.

### Explainability (How to read the UI)
We built an interpretability system natively in the frontend:
* **The Perturbation System:** We neutralize a single specific subcarrier (setting amplitude & phase to zero) and send the modified snapshot back to the model. 
* **Importance metric (Confidence Drop):** We measure the *Base prediction confidence* vs the *Perturbed prediction confidence*. Larger confidence drops imply that specific subcarrier was highly influential to the current prediction.

---

## 5. Task 2: Fall Detection (Binary Safety)

### ELI5
If someone falls down, we need to sound an alarm immediately. It's better to accidentally ring the alarm when they just sat down quickly than to *not* ring the alarm when they broke their leg.

### Technical Depth
This is a binary classification task derived from the HAR dataset.
* **Why is it significantly harder than Material Detection?**
  1. *Rarity & Imbalance*: Falls are rare. Training data is heavily skewed towards walking, sitting, etc.
  2. *Label Ambiguity*: Sitting down quickly or bending down produces a CSI distortion that is mathematically very near to a fall.
  3. *Transient Signal*: An object sits there forever, but a fall happens in 1 second. The temporal windowing must catch the exact moment.
* **Architecture used:** CNN-BiLSTM-Attention optimized with **Weighted Focal Loss**.
* **Why Focal Loss?** It dynamically scales the cross-entropy loss based on prediction confidence, heavily penalizing the model for missing difficult positive classes (falls) while down-weighting easy negatives (standing still).
* **Metric Focus:** Instead of Raw Accuracy, we focus on **Fall Recall (Sensitivity)**. Missing a fall is catastrophic.
* **Simulation & Robustness Checks:** The UI implements a "Scenario Simulator" to apply highly realistic perturbations to base samples. This assesses if the model can survive harsh distribution shifts:
  * **Noise (Std Dev):** Adds random Gaussian jitter to all channels to simulate a fundamentally noisy wireless medium.
  * **Phase Offset:** Uniform drift added to the phase components simulating uncalibrated hardware or clock drift.
  * **Attenuation:** Scales down amplitude to emulate a weaker signal or increased router-to-target distance.
  * **Channel Dropout:** Systematically zeroes out a fraction of subcarriers to test the robustness of the spatial features.
  * **Temporal Jitter:** Emulates misalignment inside the time-window (e.g. the fall occurring randomly earlier or later than the 0.8-second slice).

---

## 6. Task 3: Multi-Class Human Activity Recognition (HAR)

### The Challenge
Predict specific activities (e.g., walking, sitting, standing, falling) across different subjects. 

### Why is this the hardest problem? (Domain Shift)
* **ELI5 Domain Shift:** Imagine learning to play a video game on easy mode in a bright room. Suddenly, you have to play the final boss in a dark room with the controller upside down. The rules are the same, but the environment makes you fail.
* **Technical Depth Base Protocol:** Train on Environment 1 & 2 (Line-of-Sight or LOS), Test on Environment 3 (Non-Line-of-Sight or NLOS).
* **The Problem:** The CSI signal is *drastically* altered when an RF obstacle blocks the Line of Sight. In NLOS, high-frequency reflections bounce differently, attenuating the precise Doppler signatures of limbs moving. The multi-class model suffers heavily here. In-domain (LOS) validation F1 is comparatively high (~0.60+), but out-of-domain (NLOS) test F1 drops severely (typically ~0.17-0.20 for 4 or 5 tough overlapping classes). 
* **The "Hardest" Element:** The multi-class activity splits expose the crucial limitation of current CSI AI: *Good in-domain learning does not translate directly to strong out-of-domain (cross-environment) generalization*. Trying to explicitly distinguish Walking vs Bending vs Jumping across thick walls is nearly impossible without vast transfer-learning.
* **Issues Faced & Mitigations:**
  1. *Class Overlap*: Experiment 1 Class 2 might be the same physical motion as Experiment 2 Class 2. We generated `composite (experiment_activity)` labels to strictly separate them.
  2. *Phase Wrapping*: Wi-Fi hardware phase drifts randomly. We added dynamic phase unwrapping (removing $2\pi$ jumps) and phase centering.
  3. *Class Balancing*: We tested oversampling vs Focal Loss. (Lesson learned: combining both can cause the model to over-correct and destroy majority-class accuracy).

---

## 7. Deep Dive: Model Components Explained

### 1D Convolutional Neural Network (CNN1D)
* **ELI5:** Looking at a painting through a small magnifying glass and sliding it across.
* **Tech:** Extracts localized feature mappings from the 270-channel input. In our architecture, it reduces the high-dimensional CSI matrix into a denser latent representation before sequence processing.

### Bidirectional Long Short-Term Memory (BiLSTM)
* **ELI5:** Reading a sentence from left-to-right, and then right-to-left, to fully understand the context of a tricky word in the middle.
* **Tech:** An RNN architecture that avoids the vanishing gradient problem using gating mechanisms (Input, Output, Forget gates). By reading the sequence in both directions, the hidden states capture contextual information from both the past and the future of the time-window.

### Attention Mechanism
* **ELI5:** Highlighting the most important words in a textbook so you know exactly what is on the test.
* **Tech:** A learned dense layer that assigns a probability weight (Softmax) to each timestep in the LSTM output. It allows the model to "focus" on the 300 milliseconds where the actual fall occurred, drastically improving recall on transient events over simple global average pooling.

---

## 8. Deep Dive: CSI Preprocessing Pipeline

To clean the chaotic Wi-Fi data before feeding it to the Neural Network, we do the following:

1. **Complex Construction:** Combine Amplitude and Phase into Complex numbers.
2. **Phase Calibration:**
   - *Phase Unwrapping:* Hardware introduces jumps where phase wraps from $+\pi$ to $-\pi$. `np.unwrap` smooths this out.
   - *Phase Centering:* We center the phase to remove time-varying offset (CFO - Carrier Frequency Offset).
3. **Feature Expansion:** We convert the complex signal to `[Amplitude, sin(Phase), cos(Phase)]`. *Why?* Neural networks struggle with raw circular angular logic. Sine and Cosine explicitly map the circle into continuous [-1, 1] Cartesian coordinates.
4. **Windowing:** The sequence is split into chunks of `256` frames (about 0.8 seconds at 320 packets/sec). We use sliding windows to multiply the amount of training data.
5. **Normalization (Train-Only):** Z-score standardization applied *strictly* from the training set statistics to prevent data-leakage from the test set.

---

## 9. Shortcomings and Next Steps

### Shortcomings
1. **NLOS Generalization:** The multi-class HAR struggles to cross the Domain Shift barrier into NLOS environments. The physical loss of high-fidelity signal in NLOS makes fine-motor distinction incredibly blurred.
2. **Hardware Dependence:** CSI is specific to the Intel 5300 NIC used. A model trained on Intel 5300 data won't map cleanly to an Atheros or Nexmon setup without transfer learning.
3. **Threshold Sensitivity in Fall Detection:** There is a razor-thin margin between False Positives (over-alerting) and False Negatives (missing a fall). 

### Achievements
1. **End-to-End MLOps:** Seamless integration from Kaggle CSVs $\rightarrow$ PyTorch $\rightarrow$ Modal remote execution $\rightarrow$ locally retrievable artifacts.
2. **Interactive UI:** Built a demonstrator capable of dynamic replay and simulated channel disruptions without requiring live hardware.
3. **Robust Feature Engineering:** Solved phase unwrapping and sequence packing dynamically via caching hashes.

### Next Steps / Future Work
* **Data Augmentation:** Apply GANs or contrastive learning to synthesize NLOS data from LOS data to bridge the domain gap.
* **All-Environment Benchmark:** Test a scenario where the model splits training data by *subject* across *all* environments, rather than predicting an entirely unseen environment structure. 
* **Model Pruning:** Distill the CNN-BiLSTM for edge deployment to run locally on small IoT gateways (like a Raspberry Pi acting as an Access Point).
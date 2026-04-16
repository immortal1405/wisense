import { useEffect, useMemo, useState } from 'react';

const API = import.meta.env.VITE_API_URL ?? 'http://localhost:4000/api';

const tabs = [
  { key: 'overview', label: 'Overview' },
  { key: 'guide', label: 'Project Guide' },
  { key: 'results', label: 'Results' },
  { key: 'pipeline', label: 'Pipeline' },
  { key: 'fall_pipeline', label: 'Fall Pipeline' },
  { key: 'inference', label: 'Inference Lab' },
  { key: 'fall_detection', label: 'Fall Detection' },
];

function MetricCard({ title, value, subtitle }) {
  return (
    <article className="metric-card">
      <p className="metric-title">{title}</p>
      <h3>{value}</h3>
      <p className="metric-subtitle">{subtitle}</p>
    </article>
  );
}

function sparklinePath(values, width = 420, height = 130) {
  if (!values || values.length === 0) return '';
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(1e-6, max - min);
  return values
    .map((v, i) => {
      const x = (i / (values.length - 1 || 1)) * width;
      const y = height - ((v - min) / span) * height;
      return `${i === 0 ? 'M' : 'L'}${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(' ');
}

function defaultVector(type) {
  const base = type === 'amp' ? 0.45 : 0.52;
  return Array.from({ length: 52 }, (_, i) => Number((base + 0.15 * Math.sin(i / 6)).toFixed(4))).join(',');
}

function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [bundle, setBundle] = useState(null);
  const [loading, setLoading] = useState(true);
  const [inferenceLoading, setInferenceLoading] = useState(false);
  const [inferModel, setInferModel] = useState('cnn_bilstm');
  const [inferMode, setInferMode] = useState('random');
  const [inferCount, setInferCount] = useState('8');
  const [ampInput, setAmpInput] = useState(defaultVector('amp'));
  const [phaseInput, setPhaseInput] = useState(defaultVector('phase'));
  const [csvText, setCsvText] = useState('');
  const [csvPreset, setCsvPreset] = useState('custom');
  const [baseIndex, setBaseIndex] = useState('0');
  const [noise, setNoise] = useState('0.05');
  const [phaseOffset, setPhaseOffset] = useState('0.02');
  const [attenuation, setAttenuation] = useState('0.1');
  const [inference, setInference] = useState(null);
  const [inferError, setInferError] = useState('');
  const [lastRunContext, setLastRunContext] = useState({ mode: null, csvPreset: null });
  
  // Fall detection states
  const [fallInferenceLoading, setFallInferenceLoading] = useState(false);
  const [fallInference, setFallInference] = useState(null);
  const [fallInferError, setFallInferError] = useState('');
  const [fallInferMode, setFallInferMode] = useState('random_replay');
  const [fallInferCount, setFallInferCount] = useState('5');
  const [fallBaseIndex, setFallBaseIndex] = useState('0');
  const [fallNoise, setFallNoise] = useState('0.03');
  const [fallPhaseOffset, setFallPhaseOffset] = useState('0.02');
  const [fallAttenuation, setFallAttenuation] = useState('0.1');
  const [fallChannelDropout, setFallChannelDropout] = useState('0.05');
  const [fallTemporalJitter, setFallTemporalJitter] = useState('0');

  useEffect(() => {
    fetch(`${API}/results`)
      .then((res) => res.json())
      .then((data) => setBundle(data))
      .finally(() => setLoading(false));
  }, []);

  const standard = bundle?.standard?.runs ?? {};
  const day = bundle?.day?.runs ?? {};

  const modelRows = useMemo(() => {
    if (!standard.cnn1d_type_baseline || !standard.cnn_bilstm_type_baseline) return [];
    return [
      {
        model: 'CNN1D',
        split: 'Standard',
        acc: standard.cnn1d_type_baseline.test_accuracy,
        f1: standard.cnn1d_type_baseline.test_macro_f1,
      },
      {
        model: 'CNN-BiLSTM',
        split: 'Standard',
        acc: standard.cnn_bilstm_type_baseline.test_accuracy,
        f1: standard.cnn_bilstm_type_baseline.test_macro_f1,
      },
      {
        model: 'CNN1D',
        split: 'Day-shift',
        acc: day.cnn1d_type_day_shift?.test_accuracy,
        f1: day.cnn1d_type_day_shift?.test_macro_f1,
      },
      {
        model: 'CNN-BiLSTM',
        split: 'Day-shift',
        acc: day.cnn_bilstm_type_day_shift?.test_accuracy,
        f1: day.cnn_bilstm_type_day_shift?.test_macro_f1,
      },
    ];
  }, [standard, day]);

  const runInference = async () => {
    setInferenceLoading(true);
    setInferError('');
    try {
      const parsedCount = Math.max(1, Math.min(20, Number.parseInt(inferCount || '8', 10) || 8));
      const body = { model: inferModel, mode: inferMode, count: parsedCount };
      const runContext = { mode: inferMode, csvPreset: inferMode === 'csv' ? csvPreset : null };

      if (inferMode === 'manual') {
        body.amp = ampInput
          .split(',')
          .map((v) => Number(v.trim()))
          .filter((v) => !Number.isNaN(v));
        body.phase = phaseInput
          .split(',')
          .map((v) => Number(v.trim()))
          .filter((v) => !Number.isNaN(v));
      }

      if (inferMode === 'csv') {
        body.csv_text = csvText;
      }

      if (inferMode === 'simulate') {
        body.base_index = Number.parseInt(baseIndex || '0', 10) || 0;
        body.noise = Number.parseFloat(noise || '0.05') || 0.05;
        body.phase_offset = Number.parseFloat(phaseOffset || '0.02') || 0.02;
        body.attenuation = Number.parseFloat(attenuation || '0.1') || 0.1;
      }

      const res = await fetch(`${API}/inference`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || 'Inference failed');
      }
      const data = await res.json();
      setInference(data);
      setLastRunContext(runContext);
    } catch (err) {
      setInference(null);
      setInferError(err.message || 'Inference failed');
    } finally {
      setInferenceLoading(false);
    }
  };

  const loadSampleCsv = async () => {
    try {
      const res = await fetch('/sample_inference_row.csv');
      const txt = await res.text();
      setCsvText(txt);
      setCsvPreset('sample');
    } catch {
      setInferError('Could not load sample CSV from /sample_inference_row.csv');
    }
  };

  const loadNoisySampleCsv = async () => {
    try {
      const res = await fetch('/sample_inference_row_noisy.csv');
      const txt = await res.text();
      setCsvText(txt);
      setCsvPreset('hard');
    } catch {
      setInferError('Could not load noisy sample CSV from /sample_inference_row_noisy.csv');
    }
  };

  const runFallInference = async () => {
    setFallInferenceLoading(true);
    setFallInferError('');
    try {
      const body = { task: 'fall', mode: fallInferMode };
      
      if (fallInferMode === 'random_replay') {
        body.count = Math.max(1, Math.min(20, Number.parseInt(fallInferCount || '5', 10) || 5));
      } else if (fallInferMode === 'simulate') {
        body.base_index = Number.parseInt(fallBaseIndex || '0', 10) || 0;
        body.noise_std = Number.parseFloat(fallNoise || '0.03') || 0.03;
        body.phase_offset = Number.parseFloat(fallPhaseOffset || '0.02') || 0.02;
        body.attenuation = Number.parseFloat(fallAttenuation || '0.1') || 0.1;
        body.channel_dropout = Number.parseFloat(fallChannelDropout || '0.05') || 0.05;
        body.temporal_jitter = Number.parseInt(fallTemporalJitter || '0', 10) || 0;
      }

      const res = await fetch(`${API}/inference`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const txt = await res.text();
        throw new Error(txt || 'Fall detection inference failed');
      }
      const data = await res.json();
      setFallInference(data);
    } catch (err) {
      setFallInference(null);
      setFallInferError(err.message || 'Fall detection inference failed');
    } finally {
      setFallInferenceLoading(false);
    }
  };

  if (loading) return <div className="shell">Loading showcase...</div>;

  return (
    <div className="shell">
      <header className="hero">
        <div>
          <p className="eyebrow">Wi-Sense CSI Demo</p>
          <h1>Contactless Recognition Dashboard</h1>
          <p className="lede">
            Offline contactless sensing prototype using CSI data replay and simulation, with interactive inference
            and explainability.
          </p>
        </div>
      </header>

      <nav className="tab-row">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            className={activeTab === tab.key ? 'tab active' : 'tab'}
            onClick={() => setActiveTab(tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {activeTab === 'overview' && (
        <section className="panel">
          <div className="metric-grid">
            <MetricCard
              title="Best Standard Macro-F1"
              value={standard.cnn_bilstm_type_baseline.test_macro_f1.toFixed(4)}
              subtitle="CNN-BiLSTM"
            />
            <MetricCard
              title="Best Standard Accuracy"
              value={standard.cnn_bilstm_type_baseline.test_accuracy.toFixed(4)}
              subtitle="CNN-BiLSTM"
            />
            <MetricCard
              title="Domain Shift Macro-F1"
              value={day.cnn_bilstm_type_day_shift.test_macro_f1.toFixed(4)}
              subtitle="Day 1 -> Day 2"
            />
          </div>
          <div className="story-card">
            <h2>What we built</h2>
            <ul>
              <li>
                End-to-end CSI ML pipeline: dataset loading, label curation, feature tensorization (2 x 52),
                deterministic splitting, training, evaluation, and artifact export.
              </li>
              <li>
                Binary fall-detection extension with alert-oriented inference, including random replay and
                perturbation simulation modes for robustness checks.
              </li>
              <li>
                Two-model comparison framework: CNN1D as a compact baseline and CNN-BiLSTM as a sequence-aware
                improved model.
              </li>
              <li>
                Multi-class HAR experimentation with environment-aware evaluation to analyze LOS to NLOS
                generalization limits.
              </li>
              <li>
                Reproducible experiment execution on Modal H100 with locally retrievable checkpoints and summaries.
              </li>
              <li>
                Interactive inference system with four modes: random replay, manual vectors, CSV upload, and
                scenario simulation.
              </li>
              <li>
                Explainability layer that surfaces signal profiles and top contributing subcarriers for each
                prediction.
              </li>
            </ul>
            <h3>What This Project Demonstrates</h3>
            <ul>
              <li>
                Wi-Fi CSI can be used as a contactless sensing signal to infer material class from channel response.
              </li>
              <li>
                The same CSI stack can drive a binary safety task (fall versus non-fall) with thresholded alert output.
              </li>
              <li>
                CNN-BiLSTM consistently outperforms CNN1D in standard evaluation, indicating value from sequential
                context across subcarriers.
              </li>
              <li>
                Multi-class activity recognition is significantly harder under LOS-to-NLOS transfer because motion
                classes overlap and domain shift is stronger.
              </li>
              <li>
                Day-shift testing reveals domain shift: environmental and temporal changes alter CSI distributions and
                reduce out-of-domain accuracy.
              </li>
              <li>
                Even without live hardware capture, replay + simulation enables rigorous, human-interactive model
                interrogation.
              </li>
              <li>
                The project connects research workflow and practical demo workflow in one interface.
              </li>
            </ul>
            <h3>Clear Framing</h3>
            <ul>
              {(bundle?.project?.framing ?? []).map((line, idx) => (
                <li key={`framing-${idx}`}>{line}</li>
              ))}
            </ul>
            <h3>Limitations and Next Steps</h3>
            <ul>
              <li>
                Multi-class HAR remains the hardest setting in this project due to class overlap, class imbalance,
                and environment shift; further ablation is still required.
              </li>
              <li>
                Fall detection must optimize for high recall and low missed-fall rate, which can conflict with raw
                accuracy and requires careful threshold tuning.
              </li>
              {(bundle?.limitations ?? []).map((line, idx) => (
                <li key={`limit-${idx}`}>{line}</li>
              ))}
            </ul>
          </div>
        </section>
      )}

      {activeTab === 'results' && (
        <section className="panel">
          <h2>Experiment Metrics</h2>
          <div className="result-explainer">
            <h3>What We Achieved</h3>
            <ul>
              <li>
                Standard run: CNN-BiLSTM reached Accuracy {standard.cnn_bilstm_type_baseline.test_accuracy.toFixed(4)} and
                Macro-F1 {standard.cnn_bilstm_type_baseline.test_macro_f1.toFixed(4)}.
              </li>
              <li>
                Standard run: CNN1D reached Accuracy {standard.cnn1d_type_baseline.test_accuracy.toFixed(4)} and
                Macro-F1 {standard.cnn1d_type_baseline.test_macro_f1.toFixed(4)}.
              </li>
              <li>
                Domain-shift (day 1 -&gt; day 2): CNN-BiLSTM reached Accuracy {day.cnn_bilstm_type_day_shift.test_accuracy.toFixed(4)} and
                Macro-F1 {day.cnn_bilstm_type_day_shift.test_macro_f1.toFixed(4)}.
              </li>
              <li>
                Domain-shift (day 1 -&gt; day 2): CNN1D reached Accuracy {day.cnn1d_type_day_shift.test_accuracy.toFixed(4)} and
                Macro-F1 {day.cnn1d_type_day_shift.test_macro_f1.toFixed(4)}.
              </li>
              <li>
                Across both standard and day-shift settings, CNN-BiLSTM stays ahead of CNN1D in this project.
              </li>
            </ul>
          </div>
          <div className="bars-wrap">
            {modelRows.map((row, idx) => (
              <article className="score-bar-card" key={`bar-${row.model}-${row.split}-${idx}`}>
                <p>
                  {row.model} <span>{row.split}</span>
                </p>
                <div className="bar-line">
                  <label>Acc</label>
                  <div className="track">
                    <div className="fill teal" style={{ width: `${(row.acc ?? 0) * 100}%` }} />
                  </div>
                  <strong>{row.acc?.toFixed(3) ?? '-'}</strong>
                </div>
                <div className="bar-line">
                  <label>F1</label>
                  <div className="track">
                    <div className="fill amber" style={{ width: `${(row.f1 ?? 0) * 100}%` }} />
                  </div>
                  <strong>{row.f1?.toFixed(3) ?? '-'}</strong>
                </div>
              </article>
            ))}
          </div>
          <table className="result-table">
            <thead>
              <tr>
                <th>Model</th>
                <th>Split</th>
                <th>Test Accuracy</th>
                <th>Test Macro-F1</th>
              </tr>
            </thead>
            <tbody>
              {modelRows.map((row, idx) => (
                <tr key={`${row.model}-${row.split}-${idx}`}>
                  <td>{row.model}</td>
                  <td>{row.split}</td>
                  <td>{row.acc?.toFixed(4) ?? '-'}</td>
                  <td>{row.f1?.toFixed(4) ?? '-'}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <div className="result-explainer">
            <h3>Result Summary</h3>
            <ul>
              <li>Best overall result in this project is from CNN-BiLSTM on the standard split.</li>
              <li>Under day-shift, both models drop, but CNN-BiLSTM still maintains the better Accuracy and Macro-F1.</li>
              <li>This confirms the improved model is the stronger final choice for the project demo.</li>
            </ul>
          </div>
        </section>
      )}

      {activeTab === 'guide' && (
        <section className="panel">
          <h2>Project Guide: Wi-Sense From Data to Decision</h2>
          <p className="hint">
            This tab explains what the project does, how each data row is interpreted, what output we expect, and how
            inference/perturbation works in this contactless sensing prototype.
          </p>

          <div className="guide-stack">
            <div className="process-flow">
              <article>
                <h3>What Data We Use</h3>
                <p>
                  We use Wi-Fi CSI features from the Kaggle CSI object dataset. Each sample has 52 amplitude features
                  (amp_0...amp_51) and 52 phase features (phase_0...phase_51), plus labels such as type, day, object_id,
                  and position.
                </p>
              </article>
              <article>
                <h3>How Fall Detection Fits In</h3>
                <p>
                  Fall detection uses the same CSI sensing principle but a different objective: binary event detection
                  (fall / non-fall) with an alert threshold. This is a safety-driven mode where missing a true fall is
                  costlier than issuing an occasional false alarm.
                </p>
              </article>
              <article>
                <h3>What One Row Means</h3>
                <p>
                  One row represents one CSI snapshot under a specific environment/object condition. In this project, the
                  row is converted into a 2 x 52 tensor (amplitude channel + phase channel) and passed to the model.
                </p>
              </article>
              <article>
                <h3>Multi-class Activity Work and Current Issues</h3>
                <p>
                  Beyond binary tasks, we also trained multi-class activity models on HAR data. This setting is much
                  harder because multiple activities share similar CSI signatures, the class distribution is uneven,
                  and LOS-to-NLOS transfer introduces strong domain shift that lowers out-of-domain macro-F1.
                </p>
              </article>
              <article>
                <h3>What Output We Expect</h3>
                <p>
                  We expect one material label: organic or metalic. The model outputs class probabilities, and we pick
                  the top class as prediction with confidence.
                </p>
              </article>
              <article>
                <h3>What Organic vs Metallic Means in This Project</h3>
                <p>
                  Organic means non-metal materials such as wood, paper, plastic, or cloth, while metallic means
                  conductive metal objects. This is important because these two material groups interact with Wi-Fi
                  waves differently, so the model can use that difference as a contactless recognition signal.
                </p>
              </article>
              <article>
                <h3>How CSI Tells Us Organic vs Metallic</h3>
                <p>
                  We use 52 amplitude values (amp_0...amp_51) and 52 phase values (phase_0...phase_51) per sample.
                  Together, these 104 CSI features capture reflection strength and phase shift patterns across
                  subcarriers. In simple terms: metal usually creates stronger, sharper channel effects, while organic
                  materials produce softer changes. The model learns these patterns and maps them to organic or
                  metallic labels.
                </p>
              </article>
              <article>
                <h3>Relation to Wi-Sense Goal</h3>
                <p>
                  Wi-Sense is contactless recognition through wireless channel changes. CSI captures signal disturbances
                  caused by objects/materials, so classification from CSI is a direct proxy for contactless sensing.
                </p>
              </article>
            </div>

            <div className="result-explainer">
              <h3>How We Run Inferencing</h3>
              <ul>
                <li>
                  Random Replay: runs prediction on sampled test rows to verify baseline behavior quickly.
                </li>
                <li>
                  Manual Vector: user provides 52 amp + 52 phase values directly.
                </li>
                <li>
                  CSV Upload: user uploads/pastes one row; backend parses amp/phase columns and predicts.
                </li>
                <li>
                  Scenario Simulator: starts from a known row and applies controlled perturbations, then re-predicts.
                </li>
                <li>
                  Fall Detection Lab: runs binary fall replay/simulation and returns alert flags with fall-recall focused metrics.
                </li>
              </ul>
            </div>

            <div className="timeline">
              <article>
                <h3>Inference Parameters</h3>
                <p>
                  Model selects CNN1D or CNN-BiLSTM. Sample count controls replay size. Base index selects the source
                  row for simulation. All parameters are sent to backend and executed by the Python inference module.
                </p>
              </article>
              <article>
                <h3>Noise</h3>
                <p>
                  Adds random jitter to amplitude and phase. This simulates unstable channels or sensing noise.
                </p>
              </article>
              <article>
                <h3>Phase Offset</h3>
                <p>
                  Shifts all phase values by a constant. This simulates calibration mismatch or propagation-phase drift.
                </p>
              </article>
              <article>
                <h3>Attenuation</h3>
                <p>
                  Scales amplitude downward. This simulates weaker reflections, longer distance, or obstacle damping.
                </p>
              </article>
            </div>

            <div className="result-explainer">
              <h3>Perturbation and Perturbed Confidence</h3>
              <ul>
                <li>
                  We estimate subcarrier importance by perturbing one subcarrier at a time (setting amp and phase at
                  that index to neutral values).
                </li>
                <li>
                  Base Confidence: model confidence before perturbation.
                </li>
                <li>
                  Perturbed Confidence: confidence after perturbing one subcarrier.
                </li>
                <li>
                  Importance (confidence drop) = Base Confidence - Perturbed Confidence.
                </li>
                <li>
                  Larger drops imply that subcarrier contributes more to the current prediction.
                </li>
              </ul>
            </div>

            <div className="result-explainer">
              <h3>What Results Show Us</h3>
              <ul>
                <li>
                  Standard split scores show how well the model performs in familiar data conditions.
                </li>
                <li>
                  Day-shift scores show generalization under environment/time drift.
                </li>
                <li>
                  CNN-BiLSTM outperforming CNN1D supports the value of sequence modeling over subcarriers.
                </li>
                <li>
                  Confidence behavior under simulation indicates robustness (or fragility) to channel perturbations.
                </li>
                <li>
                  Multi-class HAR results show the key unresolved challenge: good in-domain learning does not yet
                  translate to strong LOS-to-NLOS generalization.
                </li>
                <li>
                  Fall detection adds a safety perspective, where recall and missed-event risk matter as much as
                  overall accuracy.
                </li>
              </ul>
            </div>
          </div>
        </section>
      )}

      {activeTab === 'pipeline' && (
        <section className="panel">
          <h2>Theory and Implementation (Start to End)</h2>
          <p className="hint">
            This section explains what we do, why we do it, and how each stage maps to measurable outputs.
          </p>

          <div className="process-flow">
            <article>
              <h3>1. CSI Signal Source</h3>
              <p>
                We use normalized CSI amplitudes and phases from 52 subcarriers each. Every sample is represented as
                a 2x52 tensor (amplitude channel + phase channel).
              </p>
            </article>
            <article>
              <h3>2. Preprocessing and Splits</h3>
              <p>
                We clean labels, filter to binary type classification, and create both standard i.i.d. and day-shift
                splits to test in-distribution and generalization behavior.
              </p>
            </article>
            <article>
              <h3>3. Model Learning</h3>
              <p>
                CNN1D learns local subcarrier patterns. CNN-BiLSTM adds sequence context across subcarriers, which
                improves class separability in standard and shifted conditions.
              </p>
            </article>
            <article>
              <h3>4. Evaluation and Evidence</h3>
              <p>
                We report Accuracy and Macro-F1, compare models, and keep reproducible run artifacts and summaries
                from remote training.
              </p>
            </article>
            <article>
              <h3>5. Interactive Inference</h3>
              <p>
                Without hardware, we still support human-interactive inference via manual vectors, CSV uploads, and
                scenario simulation with perturbations.
              </p>
            </article>
          </div>

          <div className="arch-card">
            <h3>Architecture Graph</h3>
            <svg viewBox="0 0 920 180" className="arch-svg" role="img" aria-label="CSI model architecture flow">
              <rect x="10" y="40" width="150" height="80" rx="12" className="arch-node" />
              <text x="85" y="78" textAnchor="middle" className="arch-text">CSI Input</text>
              <text x="85" y="98" textAnchor="middle" className="arch-sub">2 x 52</text>

              <rect x="220" y="40" width="170" height="80" rx="12" className="arch-node" />
              <text x="305" y="78" textAnchor="middle" className="arch-text">Conv Blocks</text>
              <text x="305" y="98" textAnchor="middle" className="arch-sub">local features</text>

              <rect x="450" y="20" width="200" height="60" rx="12" className="arch-node accent" />
              <text x="550" y="56" textAnchor="middle" className="arch-text">CNN1D Head</text>

              <rect x="450" y="100" width="200" height="60" rx="12" className="arch-node accent2" />
              <text x="550" y="136" textAnchor="middle" className="arch-text">BiLSTM + Head</text>

              <rect x="720" y="40" width="180" height="80" rx="12" className="arch-node" />
              <text x="810" y="78" textAnchor="middle" className="arch-text">Prediction</text>
              <text x="810" y="98" textAnchor="middle" className="arch-sub">organic / metalic</text>

              <line x1="160" y1="80" x2="220" y2="80" className="arch-edge" />
              <line x1="390" y1="80" x2="450" y2="50" className="arch-edge" />
              <line x1="390" y1="80" x2="450" y2="130" className="arch-edge" />
              <line x1="650" y1="50" x2="720" y2="80" className="arch-edge" />
              <line x1="650" y1="130" x2="720" y2="80" className="arch-edge" />
            </svg>
          </div>

          <div className="timeline">
            <article>
              <h3>Key Equation</h3>
              <p>
                Weighted cross-entropy optimization over class logits, tracked with Macro-F1 to handle imbalance.
              </p>
            </article>
            <article>
              <h3>Why Day-Shift Matters</h3>
              <p>
                Training on day 1 and testing on day 2 exposes domain shift, showing whether performance survives
                realistic environment drift.
              </p>
            </article>
            <article>
              <h3>Interpretability Strategy</h3>
              <p>
                We perturb each subcarrier and measure confidence drop to rank influential subcarriers for each
                inference case.
              </p>
            </article>
            <article>
              <h3>Deployment Story</h3>
              <p>
                Train remotely on Modal H100, store artifacts, and serve interactive inference locally through
                NestJS + React.
              </p>
            </article>
          </div>
        </section>
      )}

      {activeTab === 'fall_pipeline' && (
        <section className="panel">
          <h2>Fall Detection Theory and Pipeline</h2>
          <p className="hint">
            This tab explains the binary fall detector separately so the demo can present the full fall workflow,
            from CSI replay to alert generation, in the same style as the material pipeline.
          </p>

          <div className="process-flow">
            <article>
              <h3>1. CSI Signal Source</h3>
              <p>
                We reuse CSI amplitude and phase traces from the same wireless sensing stack, but the target is now
                binary: fall versus non-fall. The model reads short CSI windows that capture motion-induced channel
                disturbance rather than object identity.
              </p>
            </article>
            <article>
              <h3>2. Window Construction and Labeling</h3>
              <p>
                Each replayed trial is sliced into windows, normalized with train-only statistics, and labeled by
                whether a fall event is present. This turns the task into a safety-focused event detector instead of a
                multi-class classifier.
              </p>
            </article>
            <article>
              <h3>3. Model Learning</h3>
              <p>
                The CNN-BiLSTM learns local CSI patterns and short temporal motion dynamics, then uses sequence
                context to decide whether the window looks like an ordinary movement or a fall-like trajectory.
              </p>
            </article>
            <article>
              <h3>4. Alert Decision</h3>
              <p>
                The model output is converted into a probability and compared with a threshold. If the score crosses
                the threshold, the UI raises an alert, which makes the demo easy to interpret for real-time
                monitoring.
              </p>
            </article>
            <article>
              <h3>5. Robustness Checks</h3>
              <p>
                Random replay measures aggregate accuracy and fall recall, while the scenario simulator tests whether
                the detector still reacts under noise, attenuation, phase drift, channel dropout, and timing jitter.
              </p>
            </article>
          </div>

          <div className="arch-card">
            <h3>Fall Detection Flow</h3>
            <svg viewBox="0 0 920 180" className="arch-svg" role="img" aria-label="Fall detection pipeline flow">
              <rect x="10" y="40" width="150" height="80" rx="12" className="arch-node" />
              <text x="85" y="78" textAnchor="middle" className="arch-text">CSI Replay</text>
              <text x="85" y="98" textAnchor="middle" className="arch-sub">windowed traces</text>

              <rect x="220" y="40" width="170" height="80" rx="12" className="arch-node" />
              <text x="305" y="78" textAnchor="middle" className="arch-text">Preprocessing</text>
              <text x="305" y="98" textAnchor="middle" className="arch-sub">normalize + slice</text>

              <rect x="450" y="20" width="200" height="60" rx="12" className="arch-node accent" />
              <text x="550" y="56" textAnchor="middle" className="arch-text">CNN Feature Stack</text>

              <rect x="450" y="100" width="200" height="60" rx="12" className="arch-node accent2" />
              <text x="550" y="136" textAnchor="middle" className="arch-text">BiLSTM + Threshold</text>

              <rect x="720" y="40" width="180" height="80" rx="12" className="arch-node" />
              <text x="810" y="78" textAnchor="middle" className="arch-text">Alert Output</text>
              <text x="810" y="98" textAnchor="middle" className="arch-sub">fall / no fall</text>

              <line x1="160" y1="80" x2="220" y2="80" className="arch-edge" />
              <line x1="390" y1="80" x2="450" y2="50" className="arch-edge" />
              <line x1="390" y1="80" x2="450" y2="130" className="arch-edge" />
              <line x1="650" y1="50" x2="720" y2="80" className="arch-edge" />
              <line x1="650" y1="130" x2="720" y2="80" className="arch-edge" />
            </svg>
          </div>

          <div className="timeline">
            <article>
              <h3>Key Decision Rule</h3>
              <p>
                The detector optimizes a binary classification objective and then converts model confidence into an
                alert decision through a fixed threshold.
              </p>
            </article>
            <article>
              <h3>Why Fall Recall Matters</h3>
              <p>
                In safety monitoring, missing a real fall is more costly than flagging an extra warning, so recall on
                actual falls is a primary metric alongside accuracy and macro-F1.
              </p>
            </article>
            <article>
              <h3>Why This Problem Is Harder</h3>
              <p>
                Falls are rare, brief, and visually ambiguous in CSI. Normal sit-downs, bends, quick turns, or sudden
                posture changes can look similar, while true falls vary widely by person, speed, room layout, and
                signal path. That combination creates class imbalance, label ambiguity, and stronger domain shift than
                the material task.
              </p>
            </article>
            <article>
              <h3>Deployment Story</h3>
              <p>
                The UI replays stored scenarios, the backend routes the request to the binary fall model, and the
                Python inference layer returns a probability, an alert flag, and fall-specific metrics for review.
              </p>
            </article>
          </div>

          <div className="result-explainer">
            <h3>Why This Is Significantly Harder Than Material Detection</h3>
            <ul>
              <li>Falls are rare events, so the dataset is naturally imbalanced and the model can overfit to non-fall motion.</li>
              <li>Many everyday movements produce CSI changes that partially overlap with fall signatures, especially fast sitting or bending.</li>
              <li>The true signal is short-lived, so the model must catch a narrow temporal pattern instead of a stable object response.</li>
              <li>Body shape, movement style, room geometry, and channel conditions all change the fall pattern, which weakens generalization.</li>
              <li>For a safety task, recall is more important than raw accuracy, so the model must avoid missed falls even when that makes the problem harder to tune.</li>
            </ul>
          </div>
        </section>
      )}

      {activeTab === 'inference' && (
        <section className="panel">
          <h2>Inference Lab</h2>
          <div className="demo-script-card">
            <h3>Demo Script (60-90 sec)</h3>
            <ol>
              <li>
                Set mode to <strong>CSV Upload</strong>, click <strong>Load Sample CSV</strong>, then run inference.
                Mention this is an in-distribution replay case.
              </li>
              <li>
                Click <strong>Load Hard Sample</strong> and run again.
                Compare prediction confidence to show robustness under harder conditions.
              </li>
              <li>
                Switch to <strong>Scenario Simulator</strong>, raise noise/attenuation, and run.
                Explain how confidence and top subcarriers change under domain shift.
              </li>
            </ol>
          </div>

          <div className="infer-controls">
            <label>
              Model
              <select value={inferModel} onChange={(e) => setInferModel(e.target.value)}>
                <option value="cnn1d">CNN1D</option>
                <option value="cnn_bilstm">CNN-BiLSTM</option>
              </select>
            </label>
            <label>
              Samples
              <input
                type="number"
                min="1"
                max="20"
                value={inferCount}
                onChange={(e) => setInferCount(e.target.value)}
              />
            </label>
            <label>
              Mode
              <select value={inferMode} onChange={(e) => setInferMode(e.target.value)}>
                <option value="random">Random Replay</option>
                <option value="manual">Manual Vector</option>
                <option value="csv">CSV Upload</option>
                <option value="simulate">Scenario Simulator</option>
              </select>
            </label>
            <button className="run-btn" onClick={runInference} disabled={inferenceLoading}>
              {inferenceLoading ? 'Running...' : 'Run Inference'}
            </button>
          </div>

          {inferMode === 'manual' && (
            <div className="manual-grid">
              <label>
                Amplitude Vector (52 comma-separated)
                <textarea value={ampInput} onChange={(e) => setAmpInput(e.target.value)} rows={4} />
              </label>
              <label>
                Phase Vector (52 comma-separated)
                <textarea value={phaseInput} onChange={(e) => setPhaseInput(e.target.value)} rows={4} />
              </label>
            </div>
          )}

          {inferMode === 'csv' && (
            <div className="csv-box">
              <div className="csv-header-row">
                <span>CSV Input (header+row or 104 values row)</span>
                <div className="csv-actions">
                  <button type="button" className="mini-btn" onClick={loadSampleCsv}>
                    Load Sample CSV
                  </button>
                  <button type="button" className="mini-btn warn" onClick={loadNoisySampleCsv}>
                    Load Hard Sample
                  </button>
                  <a href="/sample_inference_row.csv" download className="mini-btn ghost">
                    Download Sample CSV
                  </a>
                  <a href="/sample_inference_row_noisy.csv" download className="mini-btn ghost">
                    Download Hard CSV
                  </a>
                </div>
              </div>
              <textarea
                value={csvText}
                onChange={(e) => {
                  setCsvText(e.target.value);
                  setCsvPreset('custom');
                }}
                rows={6}
                placeholder="amp_0,amp_1,...,phase_51\n0.11,0.22,..."
              />
            </div>
          )}

          {inferMode === 'simulate' && (
            <>
              <div className="sim-grid">
                <label>
                  Base Sample Index
                  <input type="number" min="0" value={baseIndex} onChange={(e) => setBaseIndex(e.target.value)} />
                  <small>Chooses the replay row from the held-out test set before perturbation.</small>
                </label>
                <label>
                  Noise
                  <input
                    type="number"
                    step="0.01"
                    min="0"
                    max="1"
                    value={noise}
                    onChange={(e) => setNoise(e.target.value)}
                  />
                  <small>Random jitter added to both amplitude and phase; higher means noisier CSI.</small>
                </label>
                <label>
                  Phase Offset
                  <input
                    type="number"
                    step="0.01"
                    min="-1"
                    max="1"
                    value={phaseOffset}
                    onChange={(e) => setPhaseOffset(e.target.value)}
                  />
                  <small>Uniform shift added to phase values to simulate calibration or propagation shift.</small>
                </label>
                <label>
                  Attenuation
                  <input
                    type="number"
                    step="0.01"
                    min="0"
                    max="0.9"
                    value={attenuation}
                    onChange={(e) => setAttenuation(e.target.value)}
                  />
                  <small>Scales amplitude downward to emulate weaker reflections or increased distance.</small>
                </label>
              </div>
              <p className="hint">
                Suggested demo sweep: keep base index fixed and increase noise/attenuation to show confidence drop or
                class flips under domain shift.
              </p>
            </>
          )}

          {inferError && <p className="error-text">{inferError}</p>}

          {inference && (
            <div className="infer-results">
              {inference.mode === 'random' ? (
                <>
                  <p>
                    Window Accuracy: <strong>{(inference.window_accuracy * 100).toFixed(1)}%</strong>
                  </p>
                  <div className="result-explainer">
                    <h3>What This Inference Output Shows</h3>
                    <ul>
                      <li>Window accuracy summarizes how many replayed samples were classified correctly.</li>
                      <li>
                        Each row is one sample: True is ground truth, Predicted is model output, Confidence is model
                        certainty for that predicted class.
                      </li>
                      <li>Higher confidence means stronger preference, but does not guarantee correctness.</li>
                      <li>Use this mode to sanity-check baseline model behavior quickly.</li>
                    </ul>
                  </div>
                  <table className="result-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>True</th>
                        <th>Predicted</th>
                        <th>Confidence</th>
                        <th>Status</th>
                      </tr>
                    </thead>
                    <tbody>
                      {inference.samples.map((s) => (
                        <tr key={s.sample_index}>
                          <td>{s.sample_index}</td>
                          <td>{s.true_label}</td>
                          <td>{s.pred_label}</td>
                          <td>{(s.confidence * 100).toFixed(1)}%</td>
                          <td>{s.correct ? 'Correct' : 'Miss'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              ) : (
                <>
                  <p>
                    Prediction: <strong>{inference.prediction?.label}</strong> at{' '}
                    <strong>{((inference.prediction?.confidence ?? 0) * 100).toFixed(1)}%</strong>
                    {typeof inference.prediction?.correct === 'boolean' && (
                      <span> ({inference.prediction.correct ? 'correct' : 'miss'})</span>
                    )}
                  </p>
                  <div className="result-explainer">
                    <h3>What This Inference Output Shows</h3>
                    <ul>
                      <li>
                        Prediction is the model's top class for your exact manual/CSV/simulated input.
                      </li>
                      <li>
                        Confidence is the probability-like score for that top class after softmax normalization.
                      </li>
                      <li>
                        Top contributing subcarriers show where perturbation hurts confidence most, giving an
                        interpretable view of sensitive CSI regions.
                      </li>
                    </ul>
                  </div>

                  {lastRunContext.mode === 'csv' && lastRunContext.csvPreset === 'hard' && (
                    <div className="result-explainer">
                      <h3>Why Hard Sample Confidence Drops</h3>
                      <ul>
                        <li>
                          The hard sample intentionally contains stronger channel distortion than the standard sample.
                        </li>
                        <li>
                          Distortion weakens the clean class-specific pattern the model learned, so class probabilities
                          become less separated.
                        </li>
                        <li>
                          Lower separation means softmax scores are flatter, which appears as lower top-class confidence.
                        </li>
                        <li>
                          This is expected under distribution shift and is exactly why robustness testing with hard cases
                          is important.
                        </li>
                      </ul>
                    </div>
                  )}

                  <div className="explain-grid">
                    <article>
                      <h3>Amplitude Profile</h3>
                      <svg viewBox="0 0 420 130" className="plot-svg">
                        <path d={sparklinePath(inference.input_signal?.amp ?? [])} className="plot-amp" />
                      </svg>
                    </article>
                    <article>
                      <h3>Phase Profile</h3>
                      <svg viewBox="0 0 420 130" className="plot-svg">
                        <path d={sparklinePath(inference.input_signal?.phase ?? [])} className="plot-phase" />
                      </svg>
                    </article>
                  </div>

                  <h3>Top Contributing Subcarriers</h3>
                  <div className="result-explainer">
                    <h3>How To Read This Table</h3>
                    <ul>
                      <li>Subcarrier: index of the CSI subcarrier being tested.</li>
                      <li>
                        Importance (confidence drop): how much predicted confidence falls when that subcarrier is
                        neutralized.
                      </li>
                      <li>Base Confidence: confidence before perturbation.</li>
                      <li>Perturbed Confidence: confidence after perturbing that single subcarrier.</li>
                      <li>Larger drop means that subcarrier contributes more to the current prediction.</li>
                    </ul>
                  </div>
                  <table className="result-table">
                    <thead>
                      <tr>
                        <th>Subcarrier</th>
                        <th>Importance (confidence drop)</th>
                        <th>Base Confidence</th>
                        <th>Perturbed Confidence</th>
                      </tr>
                    </thead>
                    <tbody>
                      {(inference.explanation?.top_subcarriers ?? []).map((row) => (
                        <tr key={row.subcarrier}>
                          <td>{row.subcarrier}</td>
                          <td>{row.importance.toFixed(4)}</td>
                          <td>{row.base_confidence.toFixed(4)}</td>
                          <td>{row.perturbed_confidence.toFixed(4)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              )}
            </div>
          )}
        </section>
      )}

      {activeTab === 'fall_detection' && (
        <section className="panel">
          <h2>Fall Detection Lab</h2>
          <div className="demo-script-card">
            <h3>What This Does</h3>
            <p>
              This module detects human falls using the same CSI-based approach as the material recognition task. 
              It uses the trained gamma-2.5 CNN-BiLSTM model with attention, which achieved ~0.59 test macro-F1.
            </p>
            <h3>Two Modes</h3>
            <ul>
              <li><strong>Random Replay:</strong> Samples random test cases and shows accuracy, per-sample predictions, and fall probabilities.</li>
              <li><strong>Scenario Simulator:</strong> Starts with a base sample and applies realistic perturbations (noise, attenuation, phase shift, channel dropout, temporal jitter) to test robustness.</li>
            </ul>
          </div>

          <div className="infer-controls">
            <label>
              Mode
              <select value={fallInferMode} onChange={(e) => setFallInferMode(e.target.value)}>
                <option value="random_replay">Random Replay</option>
                <option value="simulate">Scenario Simulator</option>
              </select>
            </label>
            {fallInferMode === 'random_replay' && (
              <label>
                Sample Count
                <input
                  type="number"
                  min="1"
                  max="20"
                  value={fallInferCount}
                  onChange={(e) => setFallInferCount(e.target.value)}
                />
              </label>
            )}
            <button className="run-btn" onClick={runFallInference} disabled={fallInferenceLoading}>
              {fallInferenceLoading ? 'Running...' : 'Run Fall Detection'}
            </button>
          </div>

          {fallInferMode === 'simulate' && (
            <div className="sim-grid">
              <label>
                Base Sample Index
                <input type="number" min="0" value={fallBaseIndex} onChange={(e) => setFallBaseIndex(e.target.value)} />
                <small>Test sample index (0-based) for base scenario before perturbation.</small>
              </label>
              <label>
                Noise (Std Dev)
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="0.5"
                  value={fallNoise}
                  onChange={(e) => setFallNoise(e.target.value)}
                />
                <small>Gaussian noise added to all channels.</small>
              </label>
              <label>
                Phase Offset
                <input
                  type="number"
                  step="0.001"
                  min="-1"
                  max="1"
                  value={fallPhaseOffset}
                  onChange={(e) => setFallPhaseOffset(e.target.value)}
                />
                <small>Uniform phase shift across all subcarriers.</small>
              </label>
              <label>
                Attenuation
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="0.9"
                  value={fallAttenuation}
                  onChange={(e) => setFallAttenuation(e.target.value)}
                />
                <small>Scales amplitude down (simulates weaker signal or longer distance).</small>
              </label>
              <label>
                Channel Dropout
                <input
                  type="number"
                  step="0.01"
                  min="0"
                  max="1"
                  value={fallChannelDropout}
                  onChange={(e) => setFallChannelDropout(e.target.value)}
                />
                <small>Fraction of channels zeroed out.</small>
              </label>
              <label>
                Temporal Jitter
                <input
                  type="number"
                  step="1"
                  min="0"
                  max="50"
                  value={fallTemporalJitter}
                  onChange={(e) => setFallTemporalJitter(e.target.value)}
                />
                <small>Sample-level time shift for temporal misalignment.</small>
              </label>
            </div>
          )}

          {fallInferError && <p className="error-text">{fallInferError}</p>}

          {fallInference && (
            <div className="infer-results">
              {fallInferMode === 'random_replay' ? (
                <>
                  <p>
                    <strong>Threshold:</strong> {fallInference.threshold?.toFixed(3)} | 
                    <strong> Accuracy:</strong> {(fallInference.summary?.accuracy * 100).toFixed(1)}% | 
                    <strong> Macro-F1:</strong> {fallInference.summary?.macro_f1.toFixed(4)} |
                    <strong> Fall Recall:</strong> {(fallInference.summary?.fall_recall * 100).toFixed(1)}%
                    {' '}
                    ({fallInference.summary?.fall_tp ?? 0}/{fallInference.summary?.fall_actual ?? 0})
                  </p>
                  <div className="result-explainer">
                    <h3>Metrics Explained</h3>
                    <ul>
                      <li><strong>Accuracy:</strong> Fraction of correct classifications (fall or non-fall).</li>
                      <li><strong>Macro-F1:</strong> Harmonic mean of precision and recall, unweighted across classes.</li>
                      <li><strong>Fall Recall:</strong> Sensitivity for actual falls (true positives / all actual falls).</li>
                      <li><strong>Alert:</strong> Predicted as fall by the model.</li>
                    </ul>
                  </div>
                  <table className="result-table">
                    <thead>
                      <tr>
                        <th>#</th>
                        <th>True</th>
                        <th>Predicted</th>
                        <th>Fall Prob</th>
                        <th>Alert</th>
                        <th>Correct</th>
                      </tr>
                    </thead>
                    <tbody>
                      {fallInference.samples.map((s, idx) => (
                        <tr key={idx}>
                          <td>{idx}</td>
                          <td>{s.true_label}</td>
                          <td>{s.pred_label}</td>
                          <td>{(s.fall_probability * 100).toFixed(1)}%</td>
                          <td>{s.alert ? '🔴 YES' : '🟢 NO'}</td>
                          <td>{s.correct ? '✓' : '✗'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </>
              ) : (
                <>
                  <div className="result-explainer">
                    <h3>Perturbation Scenario Results</h3>
                    <ul>
                      <li><strong>Base:</strong> Original unperturbed CSI sample.</li>
                      <li><strong>Simulated:</strong> Same sample after applying perturbations.</li>
                      <li><strong>Delta:</strong> Change in fall probability and whether prediction flipped.</li>
                    </ul>
                  </div>
                  
                  <div className="sim-results-grid">
                    <article>
                      <h3>Base Prediction</h3>
                      <p><strong>Label:</strong> {fallInference.base?.pred_label}</p>
                      <p><strong>Fall Prob:</strong> {(fallInference.base?.fall_probability * 100).toFixed(1)}%</p>
                      <p><strong>Alert:</strong> {fallInference.base?.alert ? '🔴 YES' : '🟢 NO'}</p>
                      <p><strong>Truth:</strong> {fallInference.base?.true_label}</p>
                    </article>
                    
                    <article>
                      <h3>After Perturbation</h3>
                      <p><strong>Label:</strong> {fallInference.simulated?.pred_label}</p>
                      <p><strong>Fall Prob:</strong> {(fallInference.simulated?.fall_probability * 100).toFixed(1)}%</p>
                      <p><strong>Alert:</strong> {fallInference.simulated?.alert ? '🔴 YES' : '🟢 NO'}</p>
                      <p><strong>Confidence:</strong> {(fallInference.simulated?.confidence * 100).toFixed(1)}%</p>
                    </article>
                    
                    <article>
                      <h3>Impact</h3>
                      <p><strong>Prob Delta:</strong> {(fallInference.delta?.fall_probability_delta * 100).toFixed(2)}%</p>
                      <p><strong>Prediction:</strong> {fallInference.delta?.flipped ? '🔄 FLIPPED' : 'Stable'}</p>
                    </article>
                  </div>
                </>
              )}
            </div>
          )}
        </section>
      )}
    </div>
  );
}

export default App;

import { Injectable } from '@nestjs/common';
import { execFileSync, ExecFileSyncOptions } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';

@Injectable()
export class ShowcaseService {
  private readonly root = resolve(__dirname, '..', '..');

  getResultsBundle() {
    const standard = JSON.parse(
      readFileSync(resolve(this.root, 'modal_results_standard.json'), 'utf-8'),
    );

    const day = {
      task: 'type',
      split: 'day_shift',
      source: 'Modal H100',
      runs: {
        cnn1d_type_day_shift: {
          best_val_macro_f1: 0.805,
          test_accuracy: 0.5741,
          test_macro_f1: 0.4927,
        },
        cnn_bilstm_type_day_shift: {
          best_val_macro_f1: 0.924,
          test_accuracy: 0.5823,
          test_macro_f1: 0.5561,
        },
      },
    };

    return {
      project: {
        title: 'Wi-Sense CSI Showcase',
        summary:
          'Offline contactless sensing prototype using Wi-Fi CSI data replay and simulation.',
        framing: [
          'This demo does not capture live CSI from hardware during runtime.',
          'Interactive inference uses user-provided vectors, CSV rows, and simulation controls.',
          'Model evidence is preserved from Modal H100 training artifacts and reports.',
        ],
      },
      standard,
      day,
      evidence: {
        files: ['PRESENTATION_RESULTS.md', 'modal_results_standard.json'],
      },
      limitations: [
        'Real-time CSI capture is future work and requires NIC and firmware-level capture pipeline.',
        'Current interface demonstrates offline replay, simulation, and model response analysis.',
      ],
      assets: {
        comparisonChart: 'outputs/presentation/model_comparison_standard.png',
        cnn1dConfusion: 'outputs/presentation/confusion_cnn1d.png',
        bilstmConfusion: 'outputs/presentation/confusion_cnn_bilstm.png',
      },
    };
  }

  runInference(task: string, mode: string, body: Record<string, unknown>) {
    const venvPython = resolve(this.root, '.venv', 'bin', 'python');
    const pythonBin = existsSync(venvPython) ? venvPython : 'python3';

    // Material classification task
    if (task === 'material' || !task) {
      const model = (body.model ?? 'cnn_bilstm') as string;
      const count = (body.count ?? 8) as number;
      const countStr = String(count);
      const payload = {
        amp: body.amp,
        phase: body.phase,
        csv_text: body.csv_text,
        base_index: body.base_index,
        noise: body.noise,
        phase_offset: body.phase_offset,
        attenuation: body.attenuation,
      };

      const output = execFileSync(
        pythonBin,
        ['-m', 'src.training.infer_api', '--model', model, '--mode', mode, '--count', countStr, '--json-stdin'],
        {
          cwd: this.root,
          encoding: 'utf-8',
          input: JSON.stringify(payload),
        } as ExecFileSyncOptions & { encoding: 'utf-8'; input: string },
      );
      return JSON.parse(output);
    }

    // Fall detection task
    if (task === 'fall') {
      const count = (body.count ?? 8) as number;
      const countStr = String(count);
      const payload = {
        count,
        base_index: body.base_index,
        noise_std: body.noise_std,
        phase_offset: body.phase_offset,
        attenuation: body.attenuation,
        channel_dropout: body.channel_dropout,
        temporal_jitter: body.temporal_jitter,
      };

      const output = execFileSync(
        pythonBin,
        ['-m', 'src.training.infer_har_fall_api', '--mode', mode, '--count', countStr, '--json-stdin'],
        {
          cwd: this.root,
          encoding: 'utf-8',
          input: JSON.stringify(payload),
        } as ExecFileSyncOptions & { encoding: 'utf-8'; input: string },
      );
      return JSON.parse(output);
    }

    throw new Error(`Unknown task: ${task}`);
  }
}

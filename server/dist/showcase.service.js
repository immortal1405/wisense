"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ShowcaseService = void 0;
const common_1 = require("@nestjs/common");
const node_child_process_1 = require("node:child_process");
const node_fs_1 = require("node:fs");
const node_path_1 = require("node:path");
let ShowcaseService = class ShowcaseService {
    constructor() {
        this.root = (0, node_path_1.resolve)(__dirname, '..', '..');
    }
    getResultsBundle() {
        const standard = JSON.parse((0, node_fs_1.readFileSync)((0, node_path_1.resolve)(this.root, 'modal_results_standard.json'), 'utf-8'));
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
                summary: 'Offline contactless sensing prototype using Wi-Fi CSI data replay and simulation.',
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
    runInference(model, mode, count, body) {
        const venvPython = (0, node_path_1.resolve)(this.root, '.venv', 'bin', 'python');
        const pythonBin = (0, node_fs_1.existsSync)(venvPython) ? venvPython : 'python3';
        const payload = {
            amp: body.amp,
            phase: body.phase,
            csv_text: body.csv_text,
            base_index: body.base_index,
            noise: body.noise,
            phase_offset: body.phase_offset,
            attenuation: body.attenuation,
        };
        const output = (0, node_child_process_1.execFileSync)(pythonBin, ['-m', 'src.training.infer_api', '--model', model, '--mode', mode, '--count', String(count), '--json-stdin'], {
            cwd: this.root,
            encoding: 'utf-8',
            input: JSON.stringify(payload),
        });
        return JSON.parse(output);
    }
};
exports.ShowcaseService = ShowcaseService;
exports.ShowcaseService = ShowcaseService = __decorate([
    (0, common_1.Injectable)()
], ShowcaseService);
//# sourceMappingURL=showcase.service.js.map
"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
var __param = (this && this.__param) || function (paramIndex, decorator) {
    return function (target, key) { decorator(target, key, paramIndex); }
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.ShowcaseController = void 0;
const common_1 = require("@nestjs/common");
const showcase_service_1 = require("./showcase.service");
let ShowcaseController = class ShowcaseController {
    constructor(showcaseService) {
        this.showcaseService = showcaseService;
    }
    health() {
        return { status: 'ok' };
    }
    results() {
        return this.showcaseService.getResultsBundle();
    }
    inference(body) {
        const task = body?.task ?? 'material';
        const mode = body?.mode ?? (task === 'fall' ? 'random_replay' : 'random');
        return this.showcaseService.runInference(task, mode, body ?? {});
    }
};
exports.ShowcaseController = ShowcaseController;
__decorate([
    (0, common_1.Get)('health'),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", []),
    __metadata("design:returntype", void 0)
], ShowcaseController.prototype, "health", null);
__decorate([
    (0, common_1.Get)('results'),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", []),
    __metadata("design:returntype", void 0)
], ShowcaseController.prototype, "results", null);
__decorate([
    (0, common_1.Post)('inference'),
    __param(0, (0, common_1.Body)()),
    __metadata("design:type", Function),
    __metadata("design:paramtypes", [Object]),
    __metadata("design:returntype", void 0)
], ShowcaseController.prototype, "inference", null);
exports.ShowcaseController = ShowcaseController = __decorate([
    (0, common_1.Controller)('api'),
    __metadata("design:paramtypes", [showcase_service_1.ShowcaseService])
], ShowcaseController);
//# sourceMappingURL=showcase.controller.js.map
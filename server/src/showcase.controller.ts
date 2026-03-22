import { Body, Controller, Get, Post } from '@nestjs/common';
import { ShowcaseService } from './showcase.service';

@Controller('api')
export class ShowcaseController {
  constructor(private readonly showcaseService: ShowcaseService) {}

  @Get('health')
  health() {
    return { status: 'ok' };
  }

  @Get('results')
  results() {
    return this.showcaseService.getResultsBundle();
  }

  @Post('inference')
  inference(
    @Body()
    body: {
      model?: string;
      mode?: 'random' | 'manual' | 'csv' | 'simulate';
      count?: number;
      amp?: number[];
      phase?: number[];
      csv_text?: string;
      base_index?: number;
      noise?: number;
      phase_offset?: number;
      attenuation?: number;
    },
  ) {
    const model = body?.model ?? 'cnn_bilstm';
    const mode = body?.mode ?? 'random';
    const count = body?.count ?? 8;
    return this.showcaseService.runInference(model, mode, count, body ?? {});
  }
}

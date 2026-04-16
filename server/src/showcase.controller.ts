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
      task?: 'material' | 'fall';
      model?: string;
      mode?: 'random' | 'manual' | 'csv' | 'simulate' | 'random_replay';
      count?: number;
      amp?: number[];
      phase?: number[];
      csv_text?: string;
      base_index?: number;
      noise?: number;
      noise_std?: number;
      phase_offset?: number;
      attenuation?: number;
      channel_dropout?: number;
      temporal_jitter?: number;
    },
  ) {
    const task = body?.task ?? 'material';
    const mode = body?.mode ?? (task === 'fall' ? 'random_replay' : 'random');
    return this.showcaseService.runInference(task, mode, body ?? {});
  }
}

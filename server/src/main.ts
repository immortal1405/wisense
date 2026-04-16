import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  const corsOrigins = (process.env.CORS_ORIGINS ?? 'http://localhost:5173')
    .split(',')
    .map((x) => x.trim())
    .filter((x) => x.length > 0);

  app.enableCors({
    origin: corsOrigins,
    methods: ['GET', 'POST'],
  });
  const port = Number(process.env.PORT ?? 4000);
  await app.listen(port);
  // eslint-disable-next-line no-console
  console.log(`Wi-Sense showcase API running on http://localhost:${port}`);
}

bootstrap();

import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  app.enableCors({
    origin: ['http://localhost:5173'],
    methods: ['GET', 'POST'],
  });
  await app.listen(4000);
  // eslint-disable-next-line no-console
  console.log('Wi-Sense showcase API running on http://localhost:4000');
}

bootstrap();

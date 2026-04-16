"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
const core_1 = require("@nestjs/core");
const app_module_1 = require("./app.module");
async function bootstrap() {
    const app = await core_1.NestFactory.create(app_module_1.AppModule);
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
    console.log(`Wi-Sense showcase API running on http://localhost:${port}`);
}
bootstrap();
//# sourceMappingURL=main.js.map
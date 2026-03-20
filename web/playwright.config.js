import { defineConfig } from '@playwright/test';

export default defineConfig({
    testDir: './tests',
    timeout: 60000,
    use: {
        baseURL: 'http://localhost:5199',
        headless: true,
        screenshot: 'on',
        trace: 'on-first-retry',
    },
    webServer: {
        command: 'npx vite --port 5199',
        port: 5199,
        reuseExistingServer: true,
        timeout: 30000,
    },
    projects: [
        { name: 'chromium', use: { browserName: 'chromium' } },
    ],
});

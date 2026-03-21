import { test, expect } from '@playwright/test';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.resolve(__dirname, '../../data');
const DOCS_DIR = path.resolve(__dirname, '../../Documents');

test('broker auto-picks best method for cube GLB', async ({ page }) => {
    test.setTimeout(60000);
    const logs = [];
    page.on('console', msg => logs.push(`[${msg.type()}] ${msg.text()}`));

    await page.goto('/');
    await page.waitForTimeout(2000);

    // Load cube
    await page.locator('#fileInput').setInputFiles(path.join(DATA_DIR, 'cube_100mm.glb'));
    await page.waitForTimeout(3000);

    // Auto method, click parameterize
    await page.selectOption('#methodSelect', 'auto');
    await page.click('#paramBtn');

    // Wait for completion
    await page.waitForFunction(
        () => document.getElementById('statusBar')?.textContent?.includes('winner'),
        { timeout: 30000 }
    );

    const status = await page.textContent('#statusBar');
    const winner = await page.textContent('#metMethod');
    const angle = await page.textContent('#metAngle');
    const score = await page.textContent('#metScore');
    console.log(`Status: ${status}`);
    console.log(`Winner: ${winner}, Angle: ${angle}, Score: ${score}`);

    // Check comparison table exists
    const rows = await page.locator('#comparisonBody tr').count();
    console.log(`Comparison table: ${rows} methods`);

    await page.screenshot({ path: 'tests/screenshots/broker-cube.png' });

    expect(status).toContain('winner');
    expect(rows).toBeGreaterThanOrEqual(2);
});

test('broker handles STEP file end-to-end', async ({ page }) => {
    test.setTimeout(120000);
    const logs = [];
    page.on('console', msg => logs.push(`[${msg.type()}] ${msg.text()}`));

    await page.goto('/');
    await page.waitForTimeout(2000);

    // Load STEP
    await page.locator('#fileInput').setInputFiles(path.join(DOCS_DIR, 'teapot.stp'));
    await page.waitForFunction(
        () => document.getElementById('statusBar')?.textContent?.includes('loaded'),
        { timeout: 60000 }
    );

    // Auto parameterize
    await page.selectOption('#methodSelect', 'auto');
    await page.click('#paramBtn');

    await page.waitForFunction(
        () => {
            const s = document.getElementById('statusBar')?.textContent || '';
            return s.includes('winner') || s.includes('Error');
        },
        { timeout: 90000 }
    );

    const status = await page.textContent('#statusBar');
    console.log(`Status: ${status}`);

    await page.screenshot({ path: 'tests/screenshots/broker-teapot.png' });
});

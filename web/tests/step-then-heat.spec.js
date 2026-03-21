import { test, expect } from '@playwright/test';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DOCS_DIR = path.resolve(__dirname, '../../Documents');

test('STEP import then heat parameterization', async ({ page }) => {
    test.setTimeout(180000);
    const logs = [];
    page.on('console', msg => logs.push(`[${msg.type()}] ${msg.text()}`));

    await page.goto('/');
    await page.waitForTimeout(2000);

    // Load STEP
    await page.locator('#fileInput').setInputFiles(path.join(DOCS_DIR, 'teapot.stp'));
    await page.waitForFunction(
        () => document.getElementById('statusBar')?.textContent?.includes('loaded') ||
              document.getElementById('statusBar')?.textContent?.includes('error'),
        { timeout: 60000 }
    );

    let status = await page.textContent('#statusBar');
    console.log('After STEP load:', status);

    // Now run heat parameterization
    await page.selectOption('#methodSelect', 'heat');
    const btnDisabled = await page.locator('#paramBtn').isDisabled();
    console.log('Param button disabled:', btnDisabled);
    if (!btnDisabled) await page.click('#paramBtn');
    else { console.log('ERROR: button is disabled!'); }

    // Wait a bit then dump logs
    await page.waitForTimeout(5000);
    console.log(`Logs so far (${logs.length}):`);
    logs.forEach(l => console.log('  ', l));

    await page.waitForFunction(
        () => {
            const s = document.getElementById('statusBar')?.textContent || '';
            return s.includes('complete') || s.includes('error');
        },
        { timeout: 30000 }
    );

    status = await page.textContent('#statusBar');
    console.log('After parameterization:', status);

    console.log(`Console (${logs.length} entries):`);
    logs.forEach(l => console.log('  ', l));

    await page.screenshot({ path: 'tests/screenshots/step-then-heat.png' });
    expect(status).not.toContain('error');
});

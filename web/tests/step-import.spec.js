import { test, expect } from '@playwright/test';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DOCS_DIR = path.resolve(__dirname, '../../Documents');

test('STEP file import and tessellation', async ({ page }) => {
    test.setTimeout(180000); // 3 min — OCC WASM download is large

    const logs = [];
    page.on('console', msg => logs.push(`[${msg.type()}] ${msg.text()}`));

    await page.goto('/');
    await page.waitForTimeout(2000);

    // Try teapot first (simpler), then Klein bottle
    let stepPath = path.join(DOCS_DIR, 'teapot.stp');
    if (!fs.existsSync(stepPath)) stepPath = path.join(DOCS_DIR, 'KleinBottle.STEP');
    if (!fs.existsSync(stepPath)) {
        console.log('No STEP file found, skipping');
        test.skip();
        return;
    }

    console.log(`Loading STEP file: ${stepPath} (${(fs.statSync(stepPath).size / 1024).toFixed(1)} KB)`);

    // Upload STEP
    await page.locator('#fileInput').setInputFiles(stepPath);

    // Wait for OCC WASM load + tessellation (may take a while on first load)
    await page.waitForFunction(
        () => {
            const s = document.getElementById('statusBar')?.textContent || '';
            return s.includes('loaded') || s.includes('error') || s.includes('Choose') ||
                   s.includes('complete') || s.includes('remeshed');
        },
        { timeout: 150000 }
    );
    // Wait a bit more for auto-remesh to finish
    await page.waitForTimeout(15000);

    const status = await page.textContent('#statusBar');
    console.log('Status:', status);

    const stepTime = await page.textContent('#metStepTime');
    const verts = await page.textContent('#metVerts');
    const tris = await page.textContent('#metTris');
    console.log(`STEP import: ${verts} verts, ${tris} tris, time: ${stepTime}`);

    console.log('Console:');
    logs.filter(l => !l.includes('WebGL') && !l.includes('vite') && !l.includes('GPU stall'))
        .forEach(l => console.log('  ', l));

    await page.screenshot({ path: 'tests/screenshots/step-import.png' });

    expect(status).not.toContain('error');
});

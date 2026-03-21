import { test, expect } from '@playwright/test';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DOCS_DIR = path.resolve(__dirname, '../../Documents');

test('STEP import with auto-remesh', async ({ page }) => {
    test.setTimeout(180000);
    const logs = [];
    page.on('console', msg => logs.push(`[${msg.type()}] ${msg.text()}`));
    page.on('pageerror', err => logs.push(`[PAGE_ERROR] ${err.message}`));

    await page.goto('/');
    await page.waitForTimeout(2000);

    // Ensure auto-remesh is checked
    const autoRemesh = page.locator('#autoRemesh');
    if (!await autoRemesh.isChecked()) await autoRemesh.check();

    const stepPath = path.join(DOCS_DIR, 'teapot.stp');
    await page.locator('#fileInput').setInputFiles(stepPath);

    // Wait up to 2 min for full STEP → GLB → remesh pipeline
    for (let i = 0; i < 24; i++) {
        await page.waitForTimeout(5000);
        const status = await page.textContent('#statusBar');
        console.log(`[${i * 5}s] Status: ${status}`);
        if (status.includes('Choose') || status.includes('error')) break;
    }

    const status = await page.textContent('#statusBar');
    const fileInfoText = await page.textContent('#fileInfo');
    const verts = await page.textContent('#metVerts');
    const tris = await page.textContent('#metTris');
    const remeshTime = await page.textContent('#metRemeshTime');

    console.log(`Final: ${status}`);
    console.log(`File: ${fileInfoText}`);
    console.log(`Mesh: ${verts} verts, ${tris} tris, remesh: ${remeshTime}`);

    console.log(`\nLogs (${logs.length}):`);
    logs.filter(l => !l.includes('WebGL') && !l.includes('GPU')).forEach(l => console.log('  ', l));

    await page.screenshot({ path: 'tests/screenshots/step-autoremesh.png' });
});

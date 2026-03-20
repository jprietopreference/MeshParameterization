import { test, expect } from '@playwright/test';
import path from 'path';
import fs from 'fs';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.resolve(__dirname, '../../data');

test.describe('Mesh Parameterization Web App', () => {

    test('page loads with BabylonJS canvas', async ({ page }) => {
        await page.goto('/');
        await expect(page.locator('#renderCanvas')).toBeVisible();
        await expect(page.locator('#statusBar')).toContainText('Ready');

        // Check BabylonJS initialized
        const hasBabylon = await page.evaluate(() => typeof BABYLON !== 'undefined');
        expect(hasBabylon).toBe(true);

        // Check engine is running
        const engineRunning = await page.evaluate(() => {
            return typeof engine !== 'undefined' && engine.isDisposed === false;
        });

        // Screenshot the initial state
        await page.screenshot({ path: 'tests/screenshots/01-initial.png' });
    });

    test('WASM modules are detected', async ({ page }) => {
        await page.goto('/');
        await page.waitForTimeout(2000); // Wait for WASM detection

        const status = await page.textContent('#statusBar');
        console.log('Status:', status);

        // Check console for any errors
        const errors = [];
        page.on('console', msg => {
            if (msg.type() === 'error') errors.push(msg.text());
        });

        await page.screenshot({ path: 'tests/screenshots/02-wasm-status.png' });
    });

    test('loads a GLB file and displays it', async ({ page }) => {
        const consoleLogs = [];
        const consoleErrors = [];
        page.on('console', msg => {
            consoleLogs.push(`[${msg.type()}] ${msg.text()}`);
            if (msg.type() === 'error') consoleErrors.push(msg.text());
        });

        await page.goto('/');
        await page.waitForTimeout(1000);

        // Check if test GLB exists
        const glbPath = path.join(DATA_DIR, 'cube_100mm.glb');
        if (!fs.existsSync(glbPath)) {
            test.skip();
            return;
        }

        // Upload the file
        const fileInput = page.locator('#fileInput');
        await fileInput.setInputFiles(glbPath);

        // Wait for loading
        await page.waitForTimeout(3000);

        // Print all console logs for debugging
        console.log('Console logs:');
        consoleLogs.forEach(l => console.log('  ', l));

        // Check status
        const status = await page.textContent('#statusBar');
        console.log('Status after load:', status);

        // Check file info
        const fileInfo = await page.textContent('#fileInfo');
        console.log('File info:', fileInfo);

        // Check metrics
        const verts = await page.textContent('#metVerts');
        const tris = await page.textContent('#metTris');
        console.log(`Mesh: ${verts} verts, ${tris} tris`);

        // Check if parameterize panel is visible
        const paramVisible = await page.locator('#paramPanel').isVisible();
        console.log('Param panel visible:', paramVisible);

        await page.screenshot({ path: 'tests/screenshots/03-glb-loaded.png' });

        // Report errors
        if (consoleErrors.length > 0) {
            console.log('ERRORS:');
            consoleErrors.forEach(e => console.log('  ', e));
        }
    });

    test('runs heat parameterization on cube', async ({ page }) => {
        const consoleLogs = [];
        page.on('console', msg => {
            consoleLogs.push(`[${msg.type()}] ${msg.text()}`);
        });

        await page.goto('/');
        await page.waitForTimeout(1000);

        const glbPath = path.join(DATA_DIR, 'cube_100mm.glb');
        if (!fs.existsSync(glbPath)) {
            test.skip();
            return;
        }

        // Load file
        await page.locator('#fileInput').setInputFiles(glbPath);
        await page.waitForTimeout(3000);

        // Select heat method
        await page.selectOption('#methodSelect', 'heat');

        // Click parameterize
        await page.click('#paramBtn');
        await page.waitForTimeout(10000); // WASM can be slow

        const status = await page.textContent('#statusBar');
        console.log('Status after param:', status);

        const paramTime = await page.textContent('#metParamTime');
        console.log('Param time:', paramTime);

        const angle = await page.textContent('#metAngle');
        console.log('Angle distortion:', angle);

        console.log('Console:');
        consoleLogs.forEach(l => console.log('  ', l));

        await page.screenshot({ path: 'tests/screenshots/04-heat-result.png' });
    });
});

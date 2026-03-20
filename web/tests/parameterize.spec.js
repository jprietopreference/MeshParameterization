import { test, expect } from '@playwright/test';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.resolve(__dirname, '../../data');

test('heat parameterization on cube', async ({ page }) => {
    const logs = [];
    page.on('console', msg => logs.push(`[${msg.type()}] ${msg.text()}`));

    await page.goto('/');
    await page.waitForTimeout(2000);

    // Load cube
    await page.locator('#fileInput').setInputFiles(path.join(DATA_DIR, 'cube_100mm.glb'));
    await page.waitForTimeout(3000);

    // Select heat method and run
    await page.selectOption('#methodSelect', 'heat');
    await page.click('#paramBtn');

    // Wait for parameterization (WASM may take a while)
    await page.waitForTimeout(30000);

    // Print logs
    console.log('Console after parameterization:');
    logs.filter(l => !l.includes('WebGL') && !l.includes('vite')).forEach(l => console.log('  ', l));

    const status = await page.textContent('#statusBar');
    console.log('Status:', status);

    const paramTime = await page.textContent('#metParamTime');
    const angle = await page.textContent('#metAngle');
    const stretch = await page.textContent('#metStretch');
    console.log(`Param time: ${paramTime}, Angle: ${angle}, Stretch: ${stretch}`);

    await page.screenshot({ path: 'tests/screenshots/heat-param-cube.png' });
});

test('CGAL conformal parameterization on cube', async ({ page }) => {
    const logs = [];
    page.on('console', msg => logs.push(`[${msg.type()}] ${msg.text()}`));

    await page.goto('/');
    await page.waitForTimeout(2000);

    await page.locator('#fileInput').setInputFiles(path.join(DATA_DIR, 'cube_100mm.glb'));
    await page.waitForTimeout(3000);

    await page.selectOption('#methodSelect', 'cgal_conformal');
    await page.click('#paramBtn');
    await page.waitForTimeout(30000);

    console.log('Console after CGAL parameterization:');
    logs.filter(l => !l.includes('WebGL') && !l.includes('vite')).forEach(l => console.log('  ', l));

    const status = await page.textContent('#statusBar');
    console.log('Status:', status);

    await page.screenshot({ path: 'tests/screenshots/cgal-conformal-cube.png' });
});

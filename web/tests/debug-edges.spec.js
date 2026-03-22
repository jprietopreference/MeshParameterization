import { test } from '@playwright/test';
import path from 'path';
import { fileURLToPath } from 'url';
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.resolve(__dirname, '../../data');

test('check edge visualization', async ({ page }) => {
    test.setTimeout(60000);
    const logs = [];
    page.on('console', msg => logs.push(`[${msg.type()}] ${msg.text()}`));

    await page.goto('/');
    await page.waitForTimeout(2000);

    await page.locator('#fileInput').setInputFiles(path.join(DATA_DIR, '0627778.glb'));
    await page.waitForTimeout(3000);

    await page.locator('#viewWeighted').check();
    await page.selectOption('#methodSelect', 'auto');
    await page.click('#paramBtn');

    await page.waitForFunction(
        () => document.getElementById('statusBar')?.textContent?.includes('winner') ||
              document.getElementById('statusBar')?.textContent?.includes('Error'),
        { timeout: 30000 }
    );

    const status = await page.textContent('#statusBar');
    console.log('Status:', status);

    // Check mesh vertex buffers
    const info = await page.evaluate(() => {
        const mesh = window.__mesh?.();
        if (!mesh) return { error: 'no mesh' };
        const geo = mesh.geometry;
        if (!geo) return { error: 'no geometry' };
        const vbs = geo.getVertexBuffers();
        return {
            keys: vbs ? Object.keys(vbs) : [],
            totalVerts: mesh.getTotalVertices(),
            seamData: mesh.getVerticesData('_SEAM')?.slice(0, 5) || null,
            faceIdData: mesh.getVerticesData('_FACE_ID')?.slice(0, 5) || null,
        };
    });
    console.log('Mesh info:', JSON.stringify(info));

    // Click show seams
    await page.locator('#showSeams').check();
    await page.waitForTimeout(1000);

    // Click show face edges
    await page.locator('#showFaceEdges').check();
    await page.waitForTimeout(1000);

    console.log('Relevant logs:');
    logs.filter(l => l.includes('edges') || l.includes('error'))
        .forEach(l => console.log('  ', l));

    await page.screenshot({ path: 'tests/screenshots/debug-edges.png' });
});

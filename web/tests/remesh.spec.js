import { test, expect } from '@playwright/test';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const DATA_DIR = path.resolve(__dirname, '../../data');

test('Gmsh isotropic remeshing on cube', async ({ page }) => {
    const logs = [];
    page.on('console', msg => logs.push(`[${msg.type()}] ${msg.text()}`));

    await page.goto('/');
    await page.waitForTimeout(2000);

    // Load cube
    await page.locator('#fileInput').setInputFiles(path.join(DATA_DIR, 'cube_100mm.glb'));
    await page.waitForTimeout(3000);

    // Set max triangles and click remesh
    await page.fill('#targetTris', '2000');
    await page.click('#remeshBtn');

    // Wait for Gmsh WASM to load + remesh
    await page.waitForTimeout(30000);

    console.log('Console after remesh:');
    logs.filter(l => !l.includes('WebGL') && !l.includes('vite')).forEach(l => console.log('  ', l));

    const status = await page.textContent('#statusBar');
    console.log('Status:', status);

    const remeshTime = await page.textContent('#metRemeshTime');
    const verts = await page.textContent('#metVerts');
    const tris = await page.textContent('#metTris');
    const remeshInfo = await page.textContent('#remeshInfo');
    console.log(`Remesh: ${verts} verts, ${tris} tris, time: ${remeshTime}`);
    console.log(`Info: ${remeshInfo}`);

    await page.screenshot({ path: 'tests/screenshots/gmsh-remesh-cube.png' });

    expect(status).not.toContain('error');
});

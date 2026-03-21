// ============================================================
// Mesh Parameterization — Thin Client
// All computation on the server. Frontend is viewer only.
// ============================================================

import * as BABYLON from '@babylonjs/core';
import '@babylonjs/loaders/glTF';

const API = 'http://localhost:8080';

// --- State ---
const state = { inputGlb: null, resultGlb: null, fileName: '' };

// --- DOM ---
const $ = id => document.getElementById(id);
const canvas = $('renderCanvas');
const engine = new BABYLON.Engine(canvas, true);
let scene = null;
let currentMesh = null;

// --- Scene ---
function createScene() {
    if (scene) scene.dispose();
    scene = new BABYLON.Scene(engine);
    scene.clearColor = new BABYLON.Color4(0.1, 0.1, 0.12, 1);
    const cam = new BABYLON.ArcRotateCamera('cam', Math.PI/2, Math.PI/2, 300, BABYLON.Vector3.Zero(), scene);
    cam.attachControl(canvas, true);
    cam.wheelPrecision = 1; cam.minZ = 0.1; cam.maxZ = 10000;
    new BABYLON.HemisphericLight('h', new BABYLON.Vector3(0,1,0), scene).intensity = 0.9;
    new BABYLON.DirectionalLight('d', new BABYLON.Vector3(-1,-2,1), scene).intensity = 0.5;
    return scene;
}
engine.runRenderLoop(() => { if (scene) scene.render(); });
window.addEventListener('resize', () => engine.resize());
createScene();

// --- Helpers ---
function setStatus(msg, type = '') { $('statusBar').textContent = msg; $('statusBar').className = type; }
function setMetric(id, val) { $(id).textContent = val; }
function clearMetrics() {
    ['metMethod','metVerts','metTris','metStepTime','metParamTime',
     'metAngle','metAngleMax','metArea','metStretch','metScore'].forEach(id => setMetric(id, '-'));
    $('comparisonBody').innerHTML = '';
}

async function loadGlb(glbBuffer, applyChecker = false) {
    if (scene) scene.meshes.slice().forEach(m => m.dispose());
    const blob = new Blob([glbBuffer], { type: 'model/gltf-binary' });
    const url = URL.createObjectURL(blob);
    try {
        const container = await BABYLON.SceneLoader.LoadAssetContainerAsync(url, '', scene, undefined, '.glb');
        container.addAllToScene();
        if (container.meshes.length > 0) {
            let min = new BABYLON.Vector3(Infinity,Infinity,Infinity);
            let max = new BABYLON.Vector3(-Infinity,-Infinity,-Infinity);
            container.meshes.forEach(m => {
                if (m.getBoundingInfo) {
                    min = BABYLON.Vector3.Minimize(min, m.getBoundingInfo().boundingBox.minimumWorld);
                    max = BABYLON.Vector3.Maximize(max, m.getBoundingInfo().boundingBox.maximumWorld);
                }
            });
            const center = min.add(max).scale(0.5);
            const extent = max.subtract(min).length();
            scene.activeCamera.target = center;
            scene.activeCamera.radius = extent * 1.5;
            new BABYLON.AxesViewer(scene, extent > 10 ? 20 : extent * 0.2);
            currentMesh = container.meshes.find(m => m.getTotalVertices() > 0) || container.meshes[0];
            if (applyChecker && currentMesh?.isVerticesDataPresent(BABYLON.VertexBuffer.UVKind)) applyCheckerboard(currentMesh);
            let tv = 0, tt = 0;
            container.meshes.forEach(m => { tv += m.getTotalVertices(); tt += m.getTotalIndices()/3; });
            setMetric('metVerts', tv.toLocaleString());
            setMetric('metTris', Math.round(tt).toLocaleString());
        }
    } finally { URL.revokeObjectURL(url); }
}

function applyCheckerboard(mesh) {
    const mat = new BABYLON.StandardMaterial('checker', scene);
    const tex = new BABYLON.Texture('textures/checker.png', scene, false, true, BABYLON.Texture.NEAREST_SAMPLINGMODE);
    tex.wrapU = BABYLON.Texture.MIRROR_ADDRESSMODE;
    tex.wrapV = BABYLON.Texture.MIRROR_ADDRESSMODE;
    tex.uScale = 2.0; tex.vScale = 2.0;
    mat.diffuseTexture = tex;
    mat.specularColor = new BABYLON.Color3(0.1, 0.1, 0.1);
    mesh.material = mat;
}

// --- Server API calls ---
async function apiHealth() {
    const r = await fetch(`${API}/api/health`, { signal: AbortSignal.timeout(3000) });
    return r.ok;
}

async function apiTessellate(stepBuffer) {
    const r = await fetch(`${API}/api/tessellate/step`, {
        method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: stepBuffer,
    });
    if (!r.ok) { const e = await r.text(); throw new Error(e); }
    return await r.arrayBuffer();
}

async function apiParameterize(glbBuffer, method, viewWeighted) {
    const params = new URLSearchParams();
    if (method !== 'auto') params.set('method', method);
    if (viewWeighted) params.set('viewWeighted', 'true');
    const r = await fetch(`${API}/api/parameterize?${params}`, {
        method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: glbBuffer,
    });
    if (!r.ok) { const e = await r.text(); throw new Error(e); }
    return {
        glb: await r.arrayBuffer(),
        method: r.headers.get('X-Method'),
        metrics: r.headers.get('X-Metrics'),
        allMethods: r.headers.get('X-All-Methods'),
    };
}

// --- File input ---
$('fileInput').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    state.fileName = file.name;
    const ext = file.name.split('.').pop().toLowerCase();
    setStatus('Loading...', 'working');
    clearMetrics();

    try {
        const buffer = await file.arrayBuffer();
        if (ext === 'step' || ext === 'stp') {
            $('fileInfo').textContent = `${file.name} (${(file.size/1024).toFixed(1)} KB) - STEP`;
            setStatus('STEP → GLB (server-side OCC tessellation)...', 'working');
            const t0 = performance.now();
            state.inputGlb = await apiTessellate(buffer);
            setMetric('metStepTime', `${(performance.now()-t0).toFixed(0)} ms`);
            $('fileInfo').textContent = `${file.name} - STEP → GLB`;
        } else {
            $('fileInfo').textContent = `${file.name} (${(file.size/1024).toFixed(1)} KB) - GLB`;
            state.inputGlb = buffer;
        }
        state.resultGlb = null;
        await loadGlb(state.inputGlb, false);
        $('paramPanel').style.display = 'block';
        $('paramBtn').disabled = false;
        $('viewPanel').style.display = 'none';
        $('metricsPanel').style.display = 'block';
        $('comparisonPanel').style.display = 'none';
        $('exportPanel').style.display = 'none';
        setStatus('File loaded. Choose parameterization method.', '');
    } catch (err) {
        setStatus(`Error: ${err.message}`, 'error');
    }
});

// --- Parameterize ---
$('paramBtn').addEventListener('click', async () => {
    const method = $('methodSelect').value;
    const viewWeighted = $('viewWeighted')?.checked || false;
    if (!state.inputGlb) return;

    $('paramBtn').disabled = true;
    $('paramInfo').textContent = '';
    const label = method === 'auto' ? 'all methods (broker)' : method;
    setStatus(`Running ${label} on server...`, 'working');

    try {
        const t0 = performance.now();
        const result = await apiParameterize(state.inputGlb, method, viewWeighted);
        const elapsed = performance.now() - t0;

        state.resultGlb = result.glb;
        await loadGlb(state.resultGlb, true);

        // Show winner
        setMetric('metMethod', result.method || method);
        setMetric('metParamTime', `${elapsed.toFixed(0)} ms`);

        if (result.metrics) {
            try {
                const m = JSON.parse(result.metrics);
                if (m.angle_mean != null) setMetric('metAngle', m.angle_mean.toFixed(2) + '\u00b0');
                if (m.angle_max != null) setMetric('metAngleMax', m.angle_max.toFixed(2) + '\u00b0');
                if (m.area_mean != null) setMetric('metArea', m.area_mean.toFixed(3));
                if (m.stretch_mean != null) setMetric('metStretch', m.stretch_mean.toFixed(2));
                if (m.score != null) setMetric('metScore', m.score.toFixed(2));
            } catch (e) {}
        }

        // Show comparison table
        if (result.allMethods) {
            try {
                const all = JSON.parse(result.allMethods);
                const tbody = $('comparisonBody');
                tbody.innerHTML = '';
                all.sort((a, b) => (a.score || 1e18) - (b.score || 1e18));
                for (const m of all) {
                    const tr = document.createElement('tr');
                    const winner = m.method === result.method;
                    tr.style.fontWeight = winner ? 'bold' : 'normal';
                    tr.style.color = !m.success ? '#e94560' : winner ? '#4ecca3' : '#ccc';
                    tr.innerHTML = `<td>${m.method}${winner ? ' ★' : ''}</td>` +
                        `<td>${m.success ? m.angle_mean?.toFixed(1) + '°' : 'FAIL'}</td>` +
                        `<td>${m.success ? m.stretch_mean?.toFixed(1) : '-'}</td>` +
                        `<td>${m.success ? m.elapsed_ms?.toFixed(0) + 'ms' : '-'}</td>` +
                        `<td>${m.success ? m.score?.toFixed(1) : '-'}</td>`;
                    tbody.appendChild(tr);
                }
                $('comparisonPanel').style.display = 'block';
            } catch (e) {}
        }

        $('viewPanel').style.display = 'block';
        $('exportPanel').style.display = 'block';
        $('paramInfo').textContent = result.method || method;
        setStatus(`Done — winner: ${result.method} (${elapsed.toFixed(0)} ms total)`, '');
    } catch (err) {
        setStatus(`Error: ${err.message}`, 'error');
        $('paramInfo').textContent = err.message;
    } finally {
        $('paramBtn').disabled = false;
    }
});

// --- Display options ---
$('textureSelect').addEventListener('change', () => {
    if (!currentMesh) return;
    const mode = $('textureSelect').value;
    if (mode === 'checker') {
        applyCheckerboard(currentMesh);
        currentMesh.material.wireframe = $('showWireframe').checked;
    } else if (mode === 'uv') {
        BABYLON.Effect.ShadersStore['uvGradVertexShader'] = `
            precision highp float; attribute vec3 position; attribute vec2 uv;
            uniform mat4 worldViewProjection; varying vec2 vUV;
            void main() { gl_Position = worldViewProjection * vec4(position, 1.0); vUV = uv; }`;
        BABYLON.Effect.ShadersStore['uvGradFragmentShader'] = `
            precision highp float; varying vec2 vUV;
            void main() { gl_FragColor = vec4(fract(vUV.x), fract(vUV.y), 0.3, 1.0); }`;
        currentMesh.material = new BABYLON.ShaderMaterial('uvS', scene, 'uvGrad',
            { attributes: ['position', 'uv'], uniforms: ['worldViewProjection'] });
        currentMesh.material.wireframe = $('showWireframe').checked;
    } else {
        const mat = new BABYLON.StandardMaterial('wire', scene);
        mat.diffuseColor = new BABYLON.Color3(0.3, 0.3, 0.3);
        mat.wireframe = true;
        currentMesh.material = mat;
    }
});

$('showWireframe').addEventListener('change', () => {
    if (currentMesh?.material) currentMesh.material.wireframe = $('showWireframe').checked;
});

// --- Export ---
$('exportBtn').addEventListener('click', () => {
    if (!state.resultGlb) return;
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([state.resultGlb], { type: 'model/gltf-binary' }));
    a.download = state.fileName.replace(/\.\w+$/, '_parameterized.glb');
    a.click();
});

// --- Init ---
(async () => {
    try {
        await apiHealth();
        setStatus('Server connected. Load a file to begin.', '');
    } catch (e) {
        setStatus('Server not available at ' + API, 'error');
    }
})();

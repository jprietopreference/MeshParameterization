// ============================================================
// Mesh Parameterization — Web Frontend
// ============================================================

import * as BABYLON from '@babylonjs/core';
import '@babylonjs/loaders/glTF';
import { stepToGlb } from './occ_step.js';

// --- State ---
const state = {
    inputGlb: null,       // ArrayBuffer of input .glb
    remeshedGlb: null,    // ArrayBuffer after optional remeshing
    resultGlb: null,      // ArrayBuffer of parameterized .glb
    fileName: '',
    wasmModules: {},
};

// --- DOM refs ---
const $ = id => document.getElementById(id);
const fileInput      = $('fileInput');
const fileInfo       = $('fileInfo');
const remeshPanel    = $('remeshPanel');
const remeshBtn      = $('remeshBtn');
const remeshInfo     = $('remeshInfo');
const paramPanel     = $('paramPanel');
const paramBtn       = $('paramBtn');
const paramInfo      = $('paramInfo');
const methodSelect   = $('methodSelect');
const viewPanel      = $('viewPanel');
const textureSelect  = $('textureSelect');
const showWireframe  = $('showWireframe');
const metricsPanel   = $('metricsPanel');
const exportPanel    = $('exportPanel');
const exportBtn      = $('exportBtn');
const statusBar      = $('statusBar');
const canvas         = $('renderCanvas');

// --- BabylonJS setup ---
const engine = new BABYLON.Engine(canvas, true, { preserveDrawingBuffer: true });
let scene = null;
let currentMesh = null;

function createScene() {
    if (scene) scene.dispose();
    scene = new BABYLON.Scene(engine);
    scene.clearColor = new BABYLON.Color4(0.1, 0.1, 0.12, 1);

    // Camera looks from +Z toward origin (alpha=PI/2, beta=PI/2 = front view)
    const camera = new BABYLON.ArcRotateCamera(
        'cam', Math.PI / 2, Math.PI / 2, 300, BABYLON.Vector3.Zero(), scene
    );
    camera.attachControl(canvas, true);
    camera.wheelPrecision = 1;
    camera.minZ = 0.1;
    camera.maxZ = 10000;

    const light = new BABYLON.HemisphericLight('light', new BABYLON.Vector3(0, 1, 0), scene);
    light.intensity = 0.9;
    const light2 = new BABYLON.DirectionalLight('dir', new BABYLON.Vector3(-1, -2, 1), scene);
    light2.intensity = 0.5;

    return scene;
}

engine.runRenderLoop(() => { if (scene) scene.render(); });
window.addEventListener('resize', () => engine.resize());
createScene();

// --- Status helpers ---
function setStatus(msg, type = '') {
    statusBar.textContent = msg;
    statusBar.className = type;
}

function setMetric(id, value) {
    $(id).textContent = value;
}

function clearMetrics() {
    ['metVerts', 'metTris', 'metStepTime', 'metRemeshTime', 'metParamTime',
     'metAngle', 'metAngleMax', 'metArea', 'metStretch', 'metIso'].forEach(
        id => setMetric(id, '-')
    );
}

// --- Load GLB into BabylonJS ---
async function loadGlbIntoScene(glbBuffer, applyChecker = false) {
    if (scene) {
        scene.meshes.slice().forEach(m => m.dispose());
    } else {
        createScene();
    }

    const blob = new Blob([glbBuffer], { type: 'model/gltf-binary' });
    const url = URL.createObjectURL(blob);

    try {
        // Load glb via AppendAsync which handles blob URLs well, then get meshes
        const container = await BABYLON.SceneLoader.LoadAssetContainerAsync(url, '', scene, undefined, '.glb');
        container.addAllToScene();
        const result = { meshes: container.meshes };

        if (result.meshes.length > 0) {
            const root = result.meshes[0];
            root.computeWorldMatrix(true);

            let min = new BABYLON.Vector3(Infinity, Infinity, Infinity);
            let max = new BABYLON.Vector3(-Infinity, -Infinity, -Infinity);
            result.meshes.forEach(m => {
                if (m.getBoundingInfo) {
                    const bi = m.getBoundingInfo();
                    min = BABYLON.Vector3.Minimize(min, bi.boundingBox.minimumWorld);
                    max = BABYLON.Vector3.Maximize(max, bi.boundingBox.maximumWorld);
                }
            });
            const center = min.add(max).scale(0.5);
            const extent = max.subtract(min).length();

            const camera = scene.activeCamera;
            camera.target = center;
            camera.radius = extent * 1.5;

            // Axes gizmo (R=X, G=Y, B=Z) — 20mm for mm meshes, 20% of extent otherwise
            const gizmoSize = extent > 10 ? 20 : extent * 0.2;
            new BABYLON.AxesViewer(scene, gizmoSize);

            currentMesh = result.meshes.find(m => m.getTotalVertices() > 0) || result.meshes[0];

            if (applyChecker && currentMesh && currentMesh.isVerticesDataPresent(BABYLON.VertexBuffer.UVKind)) {
                applyCheckerboard(currentMesh);
            }

            let totalVerts = 0, totalTris = 0;
            result.meshes.forEach(m => {
                totalVerts += m.getTotalVertices();
                totalTris += m.getTotalIndices() / 3;
            });
            setMetric('metVerts', totalVerts.toLocaleString());
            setMetric('metTris', Math.round(totalTris).toLocaleString());
        }
    } finally {
        URL.revokeObjectURL(url);
    }
}

function applyCheckerboard(mesh) {
    const mat = new BABYLON.StandardMaterial('checker', scene);
    const tex = new BABYLON.Texture('textures/checker.png', scene, false, true,
        BABYLON.Texture.NEAREST_SAMPLINGMODE);
    tex.wrapU = BABYLON.Texture.MIRROR_ADDRESSMODE;
    tex.wrapV = BABYLON.Texture.MIRROR_ADDRESSMODE;
    tex.uScale = 2.0;
    tex.vScale = 2.0;
    mat.diffuseTexture = tex;
    mat.specularColor = new BABYLON.Color3(0.1, 0.1, 0.1);
    mesh.material = mat;
}

// --- Server-side API ---
const SERVER_URL = 'http://localhost:8080';
let serverAvailable = null; // null = unknown, true/false after check

async function checkServer() {
    // Re-check each time (server may start/stop between requests)
    try {
        console.log('[routing] Checking server at', SERVER_URL);
        const controller = new AbortController();
        const timeout = setTimeout(() => controller.abort(), 2000);
        const resp = await fetch(`${SERVER_URL}/api/health`, { signal: controller.signal });
        clearTimeout(timeout);
        serverAvailable = resp.ok;
    } catch (e) {
        serverAvailable = false;
    }
    return serverAvailable;
}

// Estimate compute time in WASM (ms) based on vertex count and method
function estimateWasmTime(vertexCount, method) {
    if (method === 'heat') {
        // O(n^2) geodesic + O(n^3) eigen in WASM
        // Measured: 272 verts = 47ms, 413 verts = ~500ms, 1537 verts = >120s
        // Conservative estimate: use n^2.5 to account for eigen scaling
        return Math.pow(vertexCount, 2.5) * 0.00005;
    }
    // CGAL: sparse solvers, roughly O(n)
    return vertexCount * 0.5;
}

const WASM_TIME_LIMIT = 5000; // 5 seconds

async function parameterizeViaServer(glbBuffer, method) {
    const isHeat = method === 'heat';
    const endpoint = isHeat ? '/api/parameterize/heat' : '/api/parameterize/cgal';
    const params = new URLSearchParams();
    if (isHeat) {
        const viewWeighted = $('viewWeighted')?.checked || false;
        if (viewWeighted) params.set('viewWeighted', 'true');
    } else {
        const methodMap = {
            'cgal_conformal': 'conformal', 'cgal_arap': 'arap',
            'cgal_authalic': 'authalic', 'cgal_mvc': 'mvc',
        };
        params.set('method', methodMap[method] || 'conformal');
    }

    const url = `${SERVER_URL}${endpoint}?${params}`;
    const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/octet-stream' },
        body: glbBuffer,
    });

    if (!resp.ok) {
        const err = await resp.text();
        throw new Error(`Server error ${resp.status}: ${err}`);
    }

    const metricsHeader = resp.headers.get('X-Metrics');
    const glb = await resp.arrayBuffer();
    return { glb, metrics: metricsHeader };
}

// --- Web Worker management ---
let paramWorker = null;
let remeshWorker = null;
let workerIdCounter = 0;
const workerCallbacks = {};

function getParamWorker() {
    if (!paramWorker) {
        paramWorker = new Worker('/worker-param.js');
        paramWorker.onmessage = (e) => {
            const cb = workerCallbacks[e.data.id];
            if (cb) { delete workerCallbacks[e.data.id]; cb(e.data); }
        };
    }
    return paramWorker;
}

function getRemeshWorker() {
    if (!remeshWorker) {
        remeshWorker = new Worker('/worker-remesh.js');
        remeshWorker.onmessage = (e) => {
            const cb = workerCallbacks[e.data.id];
            if (cb) { delete workerCallbacks[e.data.id]; cb(e.data); }
        };
    }
    return remeshWorker;
}

function callWorker(worker, action, data, timeoutMs = 120000) {
    return new Promise((resolve, reject) => {
        const id = ++workerIdCounter;
        const timer = setTimeout(() => {
            delete workerCallbacks[id];
            reject(new Error(`Worker timeout (${timeoutMs / 1000}s). Mesh may be too large for client-side processing.`));
        }, timeoutMs);
        workerCallbacks[id] = (result) => {
            clearTimeout(timer);
            if (result.ok) resolve(result);
            else reject(new Error(result.error));
        };
        const glbCopy = data.glb.slice(0);
        worker.postMessage({ id, action, data: { ...data, glb: glbCopy } }, [glbCopy]);
    });
}

// --- Parameterization (auto-routes between server and WASM worker) ---
async function parameterize(glbBuffer, method) {
    console.log('[param] Starting parameterization, method=' + method + ', size=' + glbBuffer.byteLength);
    // Estimate vertex count from GLB size (rough: ~12 bytes/vertex for positions)
    const roughVertCount = Math.round(glbBuffer.byteLength / 20);
    const estimatedMs = estimateWasmTime(roughVertCount, method);
    const hasServer = await checkServer();
    const useServer = hasServer && estimatedMs > WASM_TIME_LIMIT;

    const backend = useServer ? 'server' : 'WASM';
    console.log(`[param] ${method} on ~${roughVertCount} verts, est ${estimatedMs.toFixed(0)}ms WASM → using ${backend}`);

    const t0 = performance.now();
    let resultGlb, metricsStr;

    if (useServer) {
        setStatus(`Running ${method} on server...`, 'working');
        const result = await parameterizeViaServer(glbBuffer, method);
        resultGlb = result.glb;
        metricsStr = result.metrics;
    } else {
        const worker = getParamWorker();
        const isHeat = method === 'heat';

        let result;
        if (isHeat) {
            const viewWeighted = $('viewWeighted')?.checked || false;
            result = await callWorker(worker, 'parameterize_heat', {
                glb: glbBuffer,
                viewWeighted,
                viewDir: [0, 0, 1],
            });
        } else {
            const methodMap = {
                'cgal_conformal': 'conformal', 'cgal_arap': 'arap',
                'cgal_authalic': 'authalic', 'cgal_mvc': 'mvc',
            };
            result = await callWorker(worker, 'parameterize_cgal', {
                glb: glbBuffer,
                method: methodMap[method] || 'conformal',
            });
        }
        resultGlb = result.glb;
        metricsStr = result.metrics;
    }

    const elapsed = performance.now() - t0;
    setMetric('metParamTime', `${elapsed.toFixed(0)} ms (${backend})`);

    // Read metrics
    try {
        if (metricsStr) {
            const m = JSON.parse(metricsStr);
            if (m.angle_mean != null) setMetric('metAngle', m.angle_mean.toFixed(2) + '\u00b0');
            if (m.angle_max != null) setMetric('metAngleMax', m.angle_max.toFixed(2) + '\u00b0');
            if (m.area_mean != null) setMetric('metArea', m.area_mean.toFixed(3));
            if (m.stretch_mean != null) setMetric('metStretch', m.stretch_mean.toFixed(2));
            if (m.iso_rms != null) setMetric('metIso', m.iso_rms.toFixed(4));
        }
    } catch (e) {
        console.warn('Could not read metrics:', e);
    }

    return resultGlb;
}

// --- File input handler ---
fileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    state.fileName = file.name;
    const ext = file.name.split('.').pop().toLowerCase();

    setStatus('Loading file...', 'working');
    clearMetrics();

    try {
        const buffer = await file.arrayBuffer();

        if (ext === 'step' || ext === 'stp') {
            fileInfo.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB) - STEP`;
            setStatus('Loading OCCT WASM (~7 MB on first load)...', 'working');
            try {
                const t0 = performance.now();
                state.inputGlb = await stepToGlb(buffer, 1.0);
                const elapsed = performance.now() - t0;
                setMetric('metStepTime', `${elapsed.toFixed(0)} ms`);
                fileInfo.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB) - STEP → GLB`;
            } catch (err) {
                setStatus(`STEP import error: ${err.message}`, 'error');
                fileInfo.textContent = `${file.name} - STEP import failed`;
                console.error(err);
                return;
            }
        } else {
            fileInfo.textContent = `${file.name} (${(file.size / 1024).toFixed(1)} KB) - GLB`;
            state.inputGlb = buffer;
        }

        state.remeshedGlb = null;
        state.resultGlb = null;

        await loadGlbIntoScene(state.inputGlb, false);

        remeshPanel.style.display = 'block';
        paramPanel.style.display = 'block';
        paramBtn.disabled = false;
        remeshBtn.disabled = false;
        viewPanel.style.display = 'none';
        metricsPanel.style.display = 'block';
        exportPanel.style.display = 'none';

        setStatus('File loaded. Choose parameterization method.', '');
    } catch (err) {
        setStatus(`Error: ${err.message}`, 'error');
        console.error(err);
    }
});

// --- Parameterize button ---
paramBtn.addEventListener('click', async () => {
    const method = methodSelect.value;
    const glb = state.remeshedGlb || state.inputGlb;
    if (!glb) return;

    paramBtn.disabled = true;
    paramInfo.textContent = '';
    console.log('[click] glb=', glb, 'byteLength=', glb?.byteLength, 'method=', method);
    setStatus(`Running ${method} parameterization...`, 'working');

    try {
        state.resultGlb = await parameterize(glb, method);
        await loadGlbIntoScene(state.resultGlb, true);

        viewPanel.style.display = 'block';
        exportPanel.style.display = 'block';
        setStatus('Parameterization complete.', '');
        paramInfo.textContent = `Method: ${methodSelect.options[methodSelect.selectedIndex].text}`;
    } catch (err) {
        setStatus(`Parameterization error: ${err.message}`, 'error');
        paramInfo.textContent = err.message;
        console.error(err);
    } finally {
        paramBtn.disabled = false;
    }
});

// --- Remesh button ---
remeshBtn.addEventListener('click', async () => {
    if (!state.inputGlb) return;

    remeshBtn.disabled = true;
    setStatus('Remeshing (isotropic, via Web Worker)...', 'working');

    try {
        const worker = getRemeshWorker();
        const maxTris = parseInt($('targetTris').value) || 5000;

        const t0 = performance.now();
        const result = await callWorker(worker, 'remesh', {
            glb: state.inputGlb,
            maxTriangles: maxTris,
        });
        const elapsed = performance.now() - t0;

        state.remeshedGlb = result.glb;
        setMetric('metRemeshTime', `${elapsed.toFixed(0)} ms`);

        try {
            const m = JSON.parse(result.metrics);
            setMetric('metVerts', m.vertices?.toLocaleString() || '-');
            setMetric('metTris', m.faces?.toLocaleString() || '-');
            remeshInfo.textContent = `${m.input_faces} → ${m.faces} tris in ${elapsed.toFixed(0)}ms`;
        } catch (e) { /* ignore */ }

        await loadGlbIntoScene(state.remeshedGlb, false);
        setStatus('Remeshing complete. Choose parameterization method.', '');
    } catch (err) {
        setStatus(`Remesh error: ${err.message}`, 'error');
        remeshInfo.textContent = err.message;
        console.error(err);
    } finally {
        remeshBtn.disabled = false;
    }
});

// --- Display options ---
textureSelect.addEventListener('change', () => {
    if (!currentMesh) return;

    const mode = textureSelect.value;
    if (mode === 'checker') {
        applyCheckerboard(currentMesh);
        currentMesh.material.wireframe = showWireframe.checked;
    } else if (mode === 'uv') {
        BABYLON.Effect.ShadersStore['uvGradVertexShader'] = `
            precision highp float;
            attribute vec3 position;
            attribute vec2 uv;
            uniform mat4 worldViewProjection;
            varying vec2 vUV;
            void main() {
                gl_Position = worldViewProjection * vec4(position, 1.0);
                vUV = uv;
            }
        `;
        BABYLON.Effect.ShadersStore['uvGradFragmentShader'] = `
            precision highp float;
            varying vec2 vUV;
            void main() {
                gl_FragColor = vec4(fract(vUV.x), fract(vUV.y), 0.3, 1.0);
            }
        `;
        const shaderMat = new BABYLON.ShaderMaterial('uvShader', scene, 'uvGrad', {
            attributes: ['position', 'uv'],
            uniforms: ['worldViewProjection'],
        });
        currentMesh.material = shaderMat;
        currentMesh.material.wireframe = showWireframe.checked;
    } else {
        const mat = new BABYLON.StandardMaterial('wire', scene);
        mat.diffuseColor = new BABYLON.Color3(0.3, 0.3, 0.3);
        mat.wireframe = true;
        currentMesh.material = mat;
    }
});

showWireframe.addEventListener('change', () => {
    if (currentMesh && currentMesh.material) {
        currentMesh.material.wireframe = showWireframe.checked;
    }
});

// --- Export ---
exportBtn.addEventListener('click', () => {
    if (!state.resultGlb) return;

    const blob = new Blob([state.resultGlb], { type: 'model/gltf-binary' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = state.fileName.replace(/\.\w+$/, '_parameterized.glb');
    a.click();
    URL.revokeObjectURL(url);
});

// --- Init: check for WASM modules ---
(async function init() {
    setStatus('Checking WASM modules...', 'working');

    const modules = ['meshparam', 'cgalparam', 'gmsh_remesh'];
    const available = [];

    for (const name of modules) {
        try {
            const resp = await fetch(`wasm/${name}.js`, { method: 'HEAD' });
            if (resp.ok) available.push(name);
        } catch (e) { /* not available */ }
    }

    if (available.length > 0) {
        setStatus(`Ready. WASM: ${available.join(', ')}`, '');
    } else {
        setStatus('Ready. Load pre-computed .glb results or build WASM modules.', '');
    }
})();

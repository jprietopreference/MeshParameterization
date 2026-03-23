// ============================================================
// Mesh Parameterization — Thin Client
// All computation on the server. Frontend is viewer only.
// ============================================================

import * as BABYLON from '@babylonjs/core';
import '@babylonjs/loaders/glTF';

// Configure Draco decoder for KHR_draco_mesh_compression
if (BABYLON.DracoCompression) {
    BABYLON.DracoCompression.Configuration = {
        decoder: {
            wasmUrl: "https://cdn.babylonjs.com/draco_wasm_wrapper_gltf.js",
            wasmBinaryUrl: "https://cdn.babylonjs.com/draco_decoder_gltf.wasm",
            fallbackUrl: "https://cdn.babylonjs.com/draco_decoder_gltf.js",
        }
    };
}

const API = 'http://localhost:8080';

// --- State ---
const state = { inputGlb: null, resultGlb: null, fileName: '', artistGlb: null };

// --- DOM ---
const $ = id => document.getElementById(id);
const canvas = $('renderCanvas');
const engine = new BABYLON.Engine(canvas, true);
let scene = null;
let currentMesh = null;
// Expose for debugging/testing
window.__mesh = () => currentMesh;
window.__scene = () => scene;

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
    ['metMethod','metBBox','metVerts','metTris','metStepTime','metParamTime',
     'metEquilateral','metMinAngle','metMaxAspect','metEdgeLen',
     'metSymDir','metFlipped','metL2Area','metLinfArea',
     'metAngle','metStretch','metScore'].forEach(id => setMetric(id, '-'));
    $('comparisonBody').innerHTML = '';
}

let lastGlbBuffer = null; // keep for custom attr injection
async function loadGlb(glbBuffer, applyChecker = false) {
    lastGlbBuffer = glbBuffer;
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
            // Inject custom attributes (_SEAM, _FACE_ID) from raw GLB
            if (currentMesh && lastGlbBuffer) injectCustomAttrs(currentMesh, lastGlbBuffer);
            if (applyChecker && currentMesh?.isVerticesDataPresent(BABYLON.VertexBuffer.UVKind)) applyCheckerboard(currentMesh);
            if (currentMesh) showMeshQuality(currentMesh);
        }
    } finally { URL.revokeObjectURL(url); }
}

function computeMeshQuality(mesh) {
    const positions = mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
    const indices = mesh.getIndices();
    if (!positions || !indices) return null;

    const ntri = indices.length / 3;
    let ratioSum = 0, ratioMax = 0;
    let minAngleMin = 180, minAngleSum = 0;
    let edgeMin = Infinity, edgeMax = 0, edgeSum = 0, edgeCount = 0;
    let equilateral = 0; // ratio < 1.5

    for (let t = 0; t < indices.length; t += 3) {
        const i0 = indices[t], i1 = indices[t+1], i2 = indices[t+2];
        const x0 = positions[i0*3], y0 = positions[i0*3+1], z0 = positions[i0*3+2];
        const x1 = positions[i1*3], y1 = positions[i1*3+1], z1 = positions[i1*3+2];
        const x2 = positions[i2*3], y2 = positions[i2*3+1], z2 = positions[i2*3+2];

        const e0 = Math.sqrt((x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2);
        const e1 = Math.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2);
        const e2 = Math.sqrt((x0-x2)**2 + (y0-y2)**2 + (z0-z2)**2);

        const longest = Math.max(e0, e1, e2);
        const shortest = Math.min(e0, e1, e2);
        const ratio = shortest > 1e-10 ? longest / shortest : 999;
        ratioSum += ratio;
        if (ratio > ratioMax) ratioMax = ratio;
        if (ratio < 1.5) equilateral++;

        // Min angle via law of cosines
        const edges = [e0, e1, e2];
        const sides = [[e1,e2,e0],[e0,e2,e1],[e0,e1,e2]]; // opposite edge last
        let triMinAngle = 180;
        for (const [a, b, c] of sides) {
            const cosA = (a*a + b*b - c*c) / (2*a*b + 1e-15);
            const angle = Math.acos(Math.max(-1, Math.min(1, cosA))) * 180 / Math.PI;
            if (angle < triMinAngle) triMinAngle = angle;
        }
        minAngleSum += triMinAngle;
        if (triMinAngle < minAngleMin) minAngleMin = triMinAngle;

        for (const el of [e0, e1, e2]) {
            if (el < edgeMin) edgeMin = el;
            if (el > edgeMax) edgeMax = el;
            edgeSum += el;
            edgeCount++;
        }
    }

    return {
        triangles: ntri,
        vertices: positions.length / 3,
        aspectMean: ratioSum / ntri,
        aspectMax: ratioMax,
        equilateralPct: 100 * equilateral / ntri,
        minAngleMean: minAngleSum / ntri,
        minAngleMin: minAngleMin,
        edgeMin, edgeMax,
        edgeMean: edgeSum / edgeCount,
    };
}

function showMeshQuality(mesh) {
    const q = computeMeshQuality(mesh);
    if (!q) return;
    setMetric('metVerts', q.vertices.toLocaleString());
    setMetric('metTris', q.triangles.toLocaleString());
    setMetric('metEquilateral', `${q.equilateralPct.toFixed(1)}% (aspect < 1.5)`);
    setMetric('metMinAngle', `${q.minAngleMean.toFixed(1)}\u00b0 mean, ${q.minAngleMin.toFixed(1)}\u00b0 min`);
    setMetric('metMaxAspect', q.aspectMax.toFixed(1));
    setMetric('metEdgeLen', `${q.edgeMin.toFixed(3)} / ${q.edgeMean.toFixed(3)} / ${q.edgeMax.toFixed(3)}`);
    // Bounding box from mesh positions
    const positions = mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
    if (positions) {
        let bmin = [Infinity,Infinity,Infinity], bmax = [-Infinity,-Infinity,-Infinity];
        for (let i = 0; i < positions.length; i += 3) {
            for (let j = 0; j < 3; j++) { bmin[j] = Math.min(bmin[j], positions[i+j]); bmax[j] = Math.max(bmax[j], positions[i+j]); }
        }
        const ext = [bmax[0]-bmin[0], bmax[1]-bmin[1], bmax[2]-bmin[2]];
        setMetric('metBBox', `${ext[0].toFixed(1)} \u00d7 ${ext[1].toFixed(1)} \u00d7 ${ext[2].toFixed(1)} mm`);
    }
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
    const r = await fetch(`${API}/api/tessellate/step?tessellator=gmsh`, {
        method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: stepBuffer,
    });
    if (!r.ok) { const e = await r.text(); throw new Error(e); }
    return await r.arrayBuffer();
}

async function apiConvertObj(objBuffer) {
    const r = await fetch(`${API}/api/convert/obj`, {
        method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: objBuffer,
    });
    if (!r.ok) { const e = await r.text(); throw new Error(e); }
    return await r.arrayBuffer();
}

async function apiParameterize(glbBuffer, method, viewWeighted, forceHeal) {
    const params = new URLSearchParams();
    if (method !== 'auto') params.set('method', method);
    if (viewWeighted) params.set('viewWeighted', 'true');
    if (forceHeal) params.set('heal', 'true');
    const r = await fetch(`${API}/api/parameterize?${params}`, {
        method: 'POST', headers: { 'Content-Type': 'application/octet-stream' }, body: glbBuffer,
    });
    if (!r.ok) { const e = await r.text(); throw new Error(e); }
    return {
        glb: await r.arrayBuffer(),
        method: r.headers.get('X-Method'),
        metrics: r.headers.get('X-Metrics'),
        allMethods: r.headers.get('X-All-Methods'),
        healInfo: r.headers.get('X-Heal-Info'),
        session: r.headers.get('X-Session'),
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
            setStatus('STEP → GLB (Gmsh tessellation)...', 'working');
            const t0 = performance.now();
            state.inputGlb = await apiTessellate(buffer);
            setMetric('metStepTime', `${(performance.now()-t0).toFixed(0)} ms`);
            $('fileInfo').textContent = `${file.name} - STEP (Gmsh)`;
        } else if (ext === 'obj') {
            $('fileInfo').textContent = `${file.name} (${(file.size/1024).toFixed(1)} KB) - OBJ`;
            setStatus('OBJ → GLB (converting)...', 'working');
            const t0 = performance.now();
            state.inputGlb = await apiConvertObj(buffer);
            setMetric('metStepTime', `${(performance.now()-t0).toFixed(0)} ms`);
            $('fileInfo').textContent = `${file.name} - OBJ`;
        } else {
            $('fileInfo').textContent = `${file.name} (${(file.size/1024).toFixed(1)} KB) - GLB`;
            state.inputGlb = buffer;
        }
        state.resultGlb = null;
        await loadGlb(state.inputGlb, false);
        $('paramPanel').style.display = 'block';
        $('paramBtn').disabled = false;
        $('viewPanel').style.display = 'block';
        $('metricsPanel').style.display = 'block';
        $('comparisonPanel').style.display = 'none';
        $('exportPanel').style.display = 'none';
        // Show face edges by default on input mesh (B-Rep edges available before parameterization)
        if (currentMesh && $('showFaceEdges')) {
            $('showFaceEdges').checked = true;
            showFaceEdges(currentMesh);
        }
        // Show artist UV loader for OBJ files
        $('artistRow').style.display = (ext === 'obj') ? 'block' : 'none';
        state.artistGlb = null;
        $('artistInfo').textContent = '';
        $('artistMetricsRow').style.display = 'none';
        setStatus('File loaded. Choose parameterization method.', '');
    } catch (err) {
        setStatus(`Error: ${err.message}`, 'error');
    }
});

// --- Artist UV comparison ---
$('artistInput').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    try {
        setStatus('Loading artist UVs...', 'working');
        const buffer = await file.arrayBuffer();
        // Convert OBJ with UVs to GLB via server
        state.artistGlb = await apiConvertObj(buffer);
        $('artistInfo').textContent = `${file.name} loaded`;

        // Show artist mesh with checkerboard
        state.resultGlb = state.artistGlb;
        await loadGlb(state.artistGlb, true);
        $('viewPanel').style.display = 'block';
        $('exportPanel').style.display = 'block';
        setMetric('metMethod', 'Artist UVs');
        setStatus(`Artist UVs: ${file.name}`, '');
    } catch (err) {
        $('artistInfo').textContent = 'Failed';
        setStatus(`Error loading artist UVs: ${err.message}`, 'error');
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
        const result = await apiParameterize(state.inputGlb, method, viewWeighted, false);
        const elapsed = performance.now() - t0;

        state.resultGlb = result.glb;
        await loadGlb(state.resultGlb, true);

        // Show winner
        setMetric('metMethod', result.method || method);
        setMetric('metParamTime', `${elapsed.toFixed(0)} ms`);

        if (result.metrics) {
            try {
                const m = JSON.parse(result.metrics);
                // Stein et al. paper metrics
                if (m.sym_dirichlet != null && m.sym_dirichlet > 0) setMetric('metSymDir', m.sym_dirichlet.toFixed(4));
                if (m.flipped_tris != null && m.flipped_tris >= 0) setMetric('metFlipped', m.flipped_tris);
                if (m.l2_area != null && m.l2_area >= 0) setMetric('metL2Area', m.l2_area.toFixed(4));
                if (m.linf_area != null && m.linf_area >= 0) setMetric('metLinfArea', m.linf_area.toFixed(4));
                // Other metrics
                if (m.angle_mean != null) setMetric('metAngle', m.angle_mean.toFixed(2) + '\u00b0 / ' + (m.angle_max?.toFixed(2) || '-') + '\u00b0');
                if (m.stretch_mean != null) setMetric('metStretch', m.stretch_mean.toFixed(2));
                if (m.score != null) setMetric('metScore', m.score.toFixed(2));
            } catch (e) {}
        }

        // Show comparison table (clickable rows to switch method)
        const sessionId = result.session;
        if (result.allMethods) {
            try {
                const all = JSON.parse(result.allMethods);
                const tbody = $('comparisonBody');
                tbody.innerHTML = '';
                all.sort((a, b) => (a.score || 1e18) - (b.score || 1e18));
                const fmtSD = v => v > 1e6 ? v.toExponential(1) : v > 100 ? v.toFixed(0) : v.toFixed(2);
                for (const m of all) {
                    const tr = document.createElement('tr');
                    const winner = m.method === result.method;
                    tr.style.fontWeight = winner ? 'bold' : 'normal';
                    tr.style.color = !m.success ? '#e94560' : winner ? '#4ecca3' : '#ccc';
                    if (m.success) {
                        tr.style.cursor = 'pointer';
                        tr.addEventListener('click', () => loadMethodResult(sessionId, m.method));
                    }
                    const sd = m.sym_dirichlet > 0 ? fmtSD(m.sym_dirichlet) : '-';
                    const flips = m.flipped_tris >= 0 ? m.flipped_tris : '-';
                    tr.innerHTML = `<td>${m.method}${winner ? ' \u2605' : ''}</td>` +
                        `<td>${m.success ? sd : 'FAIL'}</td>` +
                        `<td>${m.success ? flips : '-'}</td>` +
                        `<td>${m.success ? m.elapsed_ms?.toFixed(0) + 'ms' : '-'}</td>` +
                        `<td>${m.success ? fmtSD(m.score) : '-'}</td>`;
                    tbody.appendChild(tr);
                }
                $('comparisonPanel').style.display = 'block';
            } catch (e) {}
        }

        // Show healing info
        if (result.healInfo) {
            try {
                const h = JSON.parse(result.healInfo);
                const parts = [];
                if (h.removed > 0) parts.push(`${h.removed} removed`);
                if (h.perturbed > 0) parts.push(`${h.perturbed} perturbed`);
                if (parts.length > 0) {
                    $('paramInfo').textContent = `Healed: ${parts.join(', ')}${h.forced ? ' [forced]' : ''}`;
                }
            } catch (e) {}
        }

        // Build seam lines from UV discontinuities (light blue)
        if (currentMesh && $('showSeams')?.checked) {
            showSeamLines(currentMesh);
        }

        $('viewPanel').style.display = 'block';
        $('exportPanel').style.display = 'block';
        const healNote = result.healInfo ? (() => { try { const h=JSON.parse(result.healInfo); return h.removed+h.perturbed > 0 ? `, healed ${h.removed+h.perturbed} tris` : ''; } catch(e){ return ''; }})() : '';
        setStatus(`Done — winner: ${result.method} (${elapsed.toFixed(0)} ms${healNote})`, '');
    } catch (err) {
        setStatus(`Error: ${err.message}`, 'error');
        $('paramInfo').textContent = err.message;
    } finally {
        $('paramBtn').disabled = false;
    }
});

// --- Load a specific method result from session ---
async function loadMethodResult(session, methodName) {
    setStatus(`Loading ${methodName}...`, 'working');
    try {
        const r = await fetch(`${API}/api/result/${session}/${encodeURIComponent(methodName)}`);
        if (!r.ok) throw new Error(`Failed to load ${methodName}`);
        const glb = await r.arrayBuffer();
        const metricsStr = r.headers.get('X-Metrics');
        state.resultGlb = glb;
        await loadGlb(glb, true);

        if (metricsStr) {
            try {
                const m = JSON.parse(metricsStr);
                setMetric('metMethod', methodName);
                if (m.sym_dirichlet != null && m.sym_dirichlet > 0) setMetric('metSymDir', m.sym_dirichlet.toFixed(4));
                if (m.flipped_tris != null && m.flipped_tris >= 0) setMetric('metFlipped', m.flipped_tris);
                if (m.l2_area != null && m.l2_area >= 0) setMetric('metL2Area', m.l2_area.toFixed(4));
                if (m.linf_area != null && m.linf_area >= 0) setMetric('metLinfArea', m.linf_area.toFixed(4));
                if (m.angle_mean != null) setMetric('metAngle', m.angle_mean.toFixed(2) + '\u00b0 / ' + (m.angle_max?.toFixed(2) || '-') + '\u00b0');
                if (m.stretch_mean != null) setMetric('metStretch', m.stretch_mean.toFixed(2));
                if (m.score != null) setMetric('metScore', m.score.toFixed(2));
            } catch (e) {}
        }

        if ($('showSeams')?.checked && currentMesh) showSeamLines(currentMesh);
        setStatus(`Viewing: ${methodName}`, '');
    } catch (err) {
        setStatus(`Error: ${err.message}`, 'error');
    }
}

// --- Edge visualization from GLB attributes ---
function showEdgeOverlay(mesh, attrName, color, overlayName) {
    scene.meshes.filter(m => m.name === overlayName).forEach(m => m.dispose());
    if (!mesh) return;

    const positions = mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
    const indices = mesh.getIndices();
    const attrData = mesh.getVerticesData(attrName);
    if (!positions || !indices) return;

    const linePoints = [];

    if (attrData && attrName === '_SEAM') {
        // _SEAM: draw edges where both endpoints are seam vertices (value > 0.5)
        for (let t = 0; t < indices.length; t += 3) {
            for (let e = 0; e < 3; e++) {
                const a = indices[t + e], b = indices[t + (e + 1) % 3];
                if (attrData[a] > 0.5 && attrData[b] > 0.5) {
                    linePoints.push(
                        new BABYLON.Vector3(positions[a*3], positions[a*3+1], positions[a*3+2]),
                        new BABYLON.Vector3(positions[b*3], positions[b*3+1], positions[b*3+2]),
                    );
                }
            }
        }
    } else if (attrData && attrName === '_FACE_ID') {
        // _FACE_ID: draw edges between triangles with DIFFERENT face IDs
        // Build edge map: geometric edge → list of {face_id, vtx_a, vtx_b}
        const q = v => Math.round(v * 1e4);
        const pk = i => `${q(positions[i*3])}_${q(positions[i*3+1])}_${q(positions[i*3+2])}`;
        const edgeMap = new Map();
        for (let t = 0; t < indices.length; t += 3) {
            // Use face ID from first vertex of triangle (all 3 have same face ID)
            const faceId = attrData[indices[t]];
            for (let e = 0; e < 3; e++) {
                const a = indices[t + e], b = indices[t + (e + 1) % 3];
                const ka = pk(a), kb = pk(b);
                const key = ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
                if (!edgeMap.has(key)) edgeMap.set(key, []);
                edgeMap.get(key).push({ faceId, a, b });
            }
        }
        // Edges shared by two triangles with different face IDs = B-Rep boundary
        for (const [, entries] of edgeMap) {
            if (entries.length < 2) continue;
            // Check if any two entries have different face IDs
            const fids = new Set(entries.map(e => e.faceId));
            if (fids.size > 1) {
                const { a, b } = entries[0];
                linePoints.push(
                    new BABYLON.Vector3(positions[a*3], positions[a*3+1], positions[a*3+2]),
                    new BABYLON.Vector3(positions[b*3], positions[b*3+1], positions[b*3+2]),
                );
            }
        }
    } else if (attrName === '_FACE_ID') {
        // Fallback: detect face edges from position-based edge matching
        const q = v => Math.round(v * 1e4);
        const pk = i => `${q(positions[i*3])}_${q(positions[i*3+1])}_${q(positions[i*3+2])}`;
        const edgeMap = new Map();
        for (let t = 0; t < indices.length; t += 3) {
            for (let e = 0; e < 3; e++) {
                const a = indices[t + e], b = indices[t + (e + 1) % 3];
                const ka = pk(a), kb = pk(b);
                const key = ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
                if (!edgeMap.has(key)) edgeMap.set(key, 0);
                edgeMap.set(key, edgeMap.get(key) + 1);
            }
        }
        // Edges appearing only once = face boundary (split vertices)
        for (let t = 0; t < indices.length; t += 3) {
            for (let e = 0; e < 3; e++) {
                const a = indices[t + e], b = indices[t + (e + 1) % 3];
                const ka = pk(a), kb = pk(b);
                const key = ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
                if (edgeMap.get(key) === 1) {
                    linePoints.push(
                        new BABYLON.Vector3(positions[a*3], positions[a*3+1], positions[a*3+2]),
                        new BABYLON.Vector3(positions[b*3], positions[b*3+1], positions[b*3+2]),
                    );
                }
            }
        }
    } else if (attrName === '_SEAM') {
        // Fallback: detect seams from UV discontinuity (position-based)
        const uvs = mesh.getVerticesData(BABYLON.VertexBuffer.UVKind);
        if (!uvs) return;
        const q = v => Math.round(v * 1e4);
        const pk = i => `${q(positions[i*3])}_${q(positions[i*3+1])}_${q(positions[i*3+2])}`;
        const edgeMap = new Map();
        for (let t = 0; t < indices.length; t += 3) {
            for (let e = 0; e < 3; e++) {
                const a = indices[t + e], b = indices[t + (e + 1) % 3];
                const ka = pk(a), kb = pk(b);
                const key = ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
                const uA = ka < kb ? [uvs[a*2],uvs[a*2+1]] : [uvs[b*2],uvs[b*2+1]];
                const uB = ka < kb ? [uvs[b*2],uvs[b*2+1]] : [uvs[a*2],uvs[a*2+1]];
                if (!edgeMap.has(key)) edgeMap.set(key, []);
                edgeMap.get(key).push({uA, uB, a, b});
            }
        }
        for (const [, entries] of edgeMap) {
            if (entries.length < 2) continue;
            const d0 = Math.abs(entries[0].uA[0]-entries[1].uA[0]) + Math.abs(entries[0].uA[1]-entries[1].uA[1]);
            const d1 = Math.abs(entries[0].uB[0]-entries[1].uB[0]) + Math.abs(entries[0].uB[1]-entries[1].uB[1]);
            if (d0 > 0.001 || d1 > 0.001) {
                const {a, b} = entries[0];
                linePoints.push(
                    new BABYLON.Vector3(positions[a*3], positions[a*3+1], positions[a*3+2]),
                    new BABYLON.Vector3(positions[b*3], positions[b*3+1], positions[b*3+2]),
                );
            }
        }
    }

    console.log(`[edges] ${overlayName}: ${linePoints.length / 2} edges found (attr=${attrName}, hasData=${!!attrData})`);

    if (linePoints.length > 0) {
        const lines = BABYLON.MeshBuilder.CreateLineSystem(overlayName, {
            lines: Array.from({length: linePoints.length / 2}, (_, i) =>
                [linePoints[i * 2], linePoints[i * 2 + 1]]),
        }, scene);
        lines.color = color;
        lines.isPickable = false;
    }
}

// Inject custom attributes from raw GLB into BabylonJS mesh
// (BabylonJS glTF loader ignores underscore-prefixed attributes)
function injectCustomAttrs(mesh, glbBuffer) {
    try {
        const data = new Uint8Array(glbBuffer);
        const jl = new DataView(data.buffer).getUint32(12, true);
        const json = JSON.parse(new TextDecoder().decode(data.slice(20, 20 + jl)));
        const prim = json.meshes?.[0]?.primitives?.[0];
        if (!prim) return;

        let binStart = 20 + jl;
        while (binStart % 4) binStart++;
        binStart += 8;

        for (const attrName of ['_SEAM', '_FACE_ID']) {
            const accIdx = prim.attributes[attrName];
            if (accIdx == null) continue;
            const acc = json.accessors[accIdx];
            const bv = json.bufferViews[acc.bufferView];
            const offset = binStart + (bv.byteOffset || 0) + (acc.byteOffset || 0);
            const count = acc.count;
            const attrData = new Float32Array(data.buffer, offset, count);

            // Register as custom vertex buffer
            const buffer = new BABYLON.Buffer(mesh.getEngine(), attrData, false, 1);
            mesh.setVerticesBuffer(buffer.createVertexBuffer(attrName, 0, 1));
            console.log(`[inject] ${attrName}: ${count} values injected`);
        }
    } catch (e) {
        console.warn('[inject] Failed to inject custom attrs:', e);
    }
}

function showSeamLines(mesh) {
    showEdgeOverlay(mesh, '_SEAM', new BABYLON.Color3(0.3, 0.7, 1.0), '_seam_lines');
}

function showFaceEdges(mesh) {
    showEdgeOverlay(mesh, '_FACE_ID', new BABYLON.Color3(1.0, 0.6, 0.2), '_face_edges');
}

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

$('showSeams')?.addEventListener('change', () => {
    if ($('showSeams').checked) {
        $('showFaceEdges').checked = false;
        scene?.meshes.filter(m => m.name === '_face_edges').forEach(m => m.dispose());
        if (currentMesh) showSeamLines(currentMesh);
    } else {
        scene?.meshes.filter(m => m.name === '_seam_lines').forEach(m => m.dispose());
    }
});

$('showFaceEdges')?.addEventListener('change', () => {
    if ($('showFaceEdges').checked) {
        $('showSeams').checked = false;
        scene?.meshes.filter(m => m.name === '_seam_lines').forEach(m => m.dispose());
        if (currentMesh) showFaceEdges(currentMesh);
    } else {
        scene?.meshes.filter(m => m.name === '_face_edges').forEach(m => m.dispose());
    }
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

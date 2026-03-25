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
const state = { inputGlb: null, resultGlb: null, fileName: '' };

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
    // Uniform lighting: hemispheric (ambient fill) + 3 directional from orthogonal axes
    // This minimizes shading gradients on flat faces while preserving curvature visibility
    const hemi = new BABYLON.HemisphericLight('h', new BABYLON.Vector3(0, 1, 0), scene);
    hemi.intensity = 0.7;
    hemi.groundColor = new BABYLON.Color3(0.4, 0.4, 0.45); // strong ground fill
    const d1 = new BABYLON.DirectionalLight('d1', new BABYLON.Vector3(0, 0, -1), scene);
    d1.intensity = 0.3;
    const d2 = new BABYLON.DirectionalLight('d2', new BABYLON.Vector3(-1, 0, 0), scene);
    d2.intensity = 0.2;
    const d3 = new BABYLON.DirectionalLight('d3', new BABYLON.Vector3(0, -1, 0), scene);
    d3.intensity = 0.15;
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
let lastEdgeLines = null; // cached B-Rep edge lines from pipeline
let lastGlTFRoot = null;  // cached glTF root node for edge line parenting
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
            // Tighten near/far clip to scene bounds for better depth precision (reduces Z-fighting)
            scene.activeCamera.minZ = extent * 0.01;  // 1% of extent
            scene.activeCamera.maxZ = extent * 10;    // 10x extent
            new BABYLON.AxesViewer(scene, extent > 10 ? 20 : extent * 0.2);
            // Find all renderable meshes
            const renderMeshes = container.meshes.filter(m => m.getTotalVertices() > 0);
            currentMesh = renderMeshes[0] || container.meshes[0];

            // Inject custom attributes and disable backface culling
            for (const mesh of renderMeshes) {
                if (lastGlbBuffer) injectCustomAttrs(mesh, lastGlbBuffer);
                if (mesh.material) mesh.material.backFaceCulling = false;
            }

            // Draw B-Rep edge lines from glTF node extras
            // Parent edge lines to the glTF root node so they inherit its transform (Z-flip for LH)
            const glTFRoot = renderMeshes[0]?.parent || null;
            drawEdgeLinesFromExtras(container, glTFRoot);

            console.log(`[loadGlb] applyChecker=${applyChecker}, renderMeshes=${renderMeshes.length}`,
                renderMeshes.map(m => `${m.name}: ${m.getTotalVertices()}v ${m.getTotalIndices()/3}f vis=${m.isVisible}`));

            if (applyChecker) {
                // After parameterization: apply checkerboard and enable the checkbox
                applyCheckerToAll(renderMeshes);
                if ($('showParam')) { $('showParam').checked = true; $('showParam').disabled = false; }
            } else if (renderMeshes.length >= 2) {
                // Before parameterization: color front/back differently
                const frontMat = new BABYLON.StandardMaterial('frontMat', scene);
                frontMat.diffuseColor = new BABYLON.Color3(1.0, 1.0, 0.7);
                frontMat.emissiveColor = new BABYLON.Color3(0.3, 0.3, 0.2);
                frontMat.backFaceCulling = false;
                renderMeshes[0].material = frontMat;

                const backMat = new BABYLON.StandardMaterial('backMat', scene);
                backMat.diffuseColor = new BABYLON.Color3(0.7, 1.0, 0.7);
                backMat.emissiveColor = new BABYLON.Color3(0.2, 0.3, 0.2);
                backMat.backFaceCulling = false;
                renderMeshes[1].material = backMat;
            }

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

function applyCheckerboard(mesh, isBack = false) {
    const mat = new BABYLON.StandardMaterial(isBack ? 'checkerBack' : 'checker', scene);
    const texPath = isBack ? 'textures/checker_grey.png' : 'textures/checker.png';
    const tex = new BABYLON.Texture(texPath, scene, false, true, BABYLON.Texture.NEAREST_SAMPLINGMODE);
    tex.wrapU = BABYLON.Texture.MIRROR_ADDRESSMODE;
    tex.wrapV = BABYLON.Texture.MIRROR_ADDRESSMODE;
    tex.uScale = 2.0; tex.vScale = 2.0;
    mat.diffuseTexture = tex;
    mat.emissiveTexture = tex;  // self-lit: eliminates shading gradients from directional lights
    mat.emissiveColor = new BABYLON.Color3(0.3, 0.3, 0.3);
    mat.specularColor = new BABYLON.Color3(0.05, 0.05, 0.05);
    mat.backFaceCulling = false;
    mesh.material = mat;
}

// Apply/remove checkerboard on all render meshes
function applyCheckerToAll(meshes) {
    if (!meshes) meshes = scene?.meshes.filter(m =>
        m.getTotalVertices() > 0 && m.getVerticesData(BABYLON.VertexBuffer.NormalKind)) || [];
    for (let mi = 0; mi < meshes.length; mi++) {
        applyCheckerboard(meshes[mi], mi > 0);
        meshes[mi].isVisible = true;
    }
}

function applyPlainToAll(meshes) {
    if (!meshes) meshes = scene?.meshes.filter(m =>
        m.getTotalVertices() > 0 && m.getVerticesData(BABYLON.VertexBuffer.NormalKind)) || [];
    for (let mi = 0; mi < meshes.length; mi++) {
        const mat = new BABYLON.StandardMaterial(mi === 0 ? 'frontMat' : 'backMat', scene);
        mat.diffuseColor = mi === 0 ? new BABYLON.Color3(1.0, 1.0, 0.7) : new BABYLON.Color3(0.7, 1.0, 0.7);
        // Strong emissive base to reduce directional light gradients (same approach as checkerboard)
        mat.emissiveColor = mi === 0 ? new BABYLON.Color3(0.55, 0.55, 0.4) : new BABYLON.Color3(0.4, 0.55, 0.4);
        mat.specularColor = new BABYLON.Color3(0.05, 0.05, 0.05);
        mat.backFaceCulling = false;
        meshes[mi].material = mat;
        meshes[mi].isVisible = true;
    }
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

async function apiParameterize(glbBuffer, method) {
    const params = new URLSearchParams();
    if (method !== 'auto') params.set('method', method);
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
        // Face edges are now auto-shown via line primitives in loadGlb
        if ($('showFaceEdges')) {
            $('showFaceEdges').checked = currentMesh?.getVerticesData('_FACE_ID') != null;
        }
        setStatus('File loaded. Choose parameterization method.', '');
    } catch (err) {
        setStatus(`Error: ${err.message}`, 'error');
    }
});

// --- Parameterize ---
$('paramBtn').addEventListener('click', async () => {
    const method = $('methodSelect').value;
    if (!state.inputGlb) return;

    $('paramBtn').disabled = true;
    $('paramInfo').textContent = '';
    const label = method === 'auto' ? 'all methods (broker)' : method;
    setStatus(`Running ${label} on server...`, 'working');

    try {
        const t0 = performance.now();
        const result = await apiParameterize(state.inputGlb, method);
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
            showSeamLines();
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

        if ($('showSeams')?.checked && currentMesh) showSeamLines();
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
        // Parent to mesh so edge overlay follows the mesh's world transform
        lines.parent = mesh.parent || mesh;
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

function showSeamLines() {
    // Seam = thick transparent blue, using edgeLines.seam from cached extras
    scene.meshes.filter(m => m.name === '_seam_lines').forEach(m => m.dispose());
    if (!lastEdgeLines || !lastEdgeLines.seam) return;

    const pts = lastEdgeLines.seam;
    const lines = [];
    for (let i = 0; i < pts.length; i += 6) {
        lines.push([
            new BABYLON.Vector3(pts[i], pts[i+1], pts[i+2]),
            new BABYLON.Vector3(pts[i+3], pts[i+4], pts[i+5]),
        ]);
    }
    if (lines.length > 0) {
        const lineSys = BABYLON.MeshBuilder.CreateLineSystem('_seam_lines', { lines }, scene);
        lineSys.color = new BABYLON.Color3(0.3, 0.6, 1.0);
        lineSys.alpha = 0.6;
        lineSys.isPickable = false;
        // No zOffset needed — edges are offset 0.05mm along surface normal in the pipeline
        if (lastGlTFRoot) lineSys.parent = lastGlTFRoot;
        lineSys.enableEdgesRendering();
        lineSys.edgesWidth = 4.0;
        lineSys.edgesColor = new BABYLON.Color4(0.3, 0.6, 1.0, 0.6);
    }
}

// Draw B-Rep edge lines from glTF node extras (edgeLines: {seam, zperp, other})
function drawEdgeLinesFromExtras(container, parentNode) {
    // Dispose old edge overlays
    for (const n of ['_edges_seam', '_edges_zperp', '_edges_other'])
        scene.meshes.filter(m => m.name === n).forEach(m => m.dispose());

    // Find edgeLines in node extras (BabylonJS stores them on transformNodes or meshes)
    let edgeLines = null;
    // Check all transform nodes
    for (const tn of (container.transformNodes || [])) {
        edgeLines = tn.metadata?.gltf?.extras?.edgeLines;
        if (edgeLines) break;
    }
    // Fallback: check meshes
    if (!edgeLines) {
        for (const m of (container.meshes || [])) {
            edgeLines = m.metadata?.gltf?.extras?.edgeLines;
            if (edgeLines) break;
        }
    }
    // Fallback: parse from raw GLB JSON
    if (!edgeLines && lastGlbBuffer) {
        try {
            const dv = new DataView(lastGlbBuffer.buffer || lastGlbBuffer);
            const jsonLen = dv.getUint32(12, true);
            const jsonStr = new TextDecoder().decode(new Uint8Array(lastGlbBuffer.buffer || lastGlbBuffer, 20, jsonLen));
            const gltf = JSON.parse(jsonStr);
            edgeLines = gltf.nodes?.[0]?.extras?.edgeLines;
        } catch(e) { console.warn('[edges] Failed to parse GLB JSON:', e); }
    }
    if (edgeLines) {
        // Cache for reuse after parameterization
        lastEdgeLines = edgeLines;
        if (parentNode) lastGlTFRoot = parentNode;
    } else if (lastEdgeLines) {
        // Reuse cached edge lines (e.g. after parameterization)
        edgeLines = lastEdgeLines;
    } else {
        console.log('[edges] No edgeLines found');
        return;
    }

    const colorMap = {
        seam:   new BABYLON.Color3(1.0, 0.1, 0.1),  // red
        zperp:  new BABYLON.Color3(1.0, 0.6, 0.2),  // orange
        other:  new BABYLON.Color3(1.0, 0.9, 0.2),  // yellow
    };

    for (const [key, pts] of Object.entries(edgeLines)) {
        if (!pts || pts.length < 6) continue;
        const lines = [];
        for (let i = 0; i < pts.length; i += 6) {
            lines.push([
                new BABYLON.Vector3(pts[i], pts[i+1], pts[i+2]),
                new BABYLON.Vector3(pts[i+3], pts[i+4], pts[i+5]),
            ]);
        }
        const lineSys = BABYLON.MeshBuilder.CreateLineSystem(`_edges_${key}`, { lines }, scene);
        lineSys.color = colorMap[key] || new BABYLON.Color3(1, 1, 1);
        lineSys.isPickable = false;
        // No zOffset needed — edges are offset 0.05mm along surface normal in the pipeline
        // Parent to glTF root so edge lines inherit the same transform (LH Z-flip)
        if (parentNode) lineSys.parent = parentNode;
        console.log(`[edges] ${key}: ${lines.length} segments`);
    }
}

function showFaceEdges() {
    const show = $('showFaceEdges')?.checked ?? true;
    for (const n of ['_edges_seam', '_edges_zperp', '_edges_other'])
        scene?.meshes.filter(m => m.name === n).forEach(m => { m.isVisible = show; });
}

function showVertexNormals() {
    scene?.meshes.filter(m => m.name === '_vertex_normals').forEach(m => m.dispose());
    if (!currentMesh) return;

    // Collect positions and normals from all render meshes (front + back primitives)
    const renderMeshes = scene.meshes.filter(m =>
        m.getTotalVertices() > 0 && m.getVerticesData(BABYLON.VertexBuffer.NormalKind));

    // Compute normal length: 2% of bounding box diagonal
    const bb = currentMesh.getBoundingInfo().boundingBox;
    const diag = bb.maximumWorld.subtract(bb.minimumWorld).length();
    const normalLen = diag * 0.02;

    const allLines = [];
    for (const mesh of renderMeshes) {
        const positions = mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
        const normals = mesh.getVerticesData(BABYLON.VertexBuffer.NormalKind);
        if (!positions || !normals) continue;

        const nv = positions.length / 3;
        for (let i = 0; i < nv; i++) {
            const px = positions[i * 3], py = positions[i * 3 + 1], pz = positions[i * 3 + 2];
            const nx = normals[i * 3], ny = normals[i * 3 + 1], nz = normals[i * 3 + 2];
            allLines.push([
                new BABYLON.Vector3(px, py, pz),
                new BABYLON.Vector3(px + nx * normalLen, py + ny * normalLen, pz + nz * normalLen),
            ]);
        }
    }

    if (allLines.length === 0) return;
    const lineSys = BABYLON.MeshBuilder.CreateLineSystem('_vertex_normals', { lines: allLines }, scene);
    lineSys.color = new BABYLON.Color3(1, 1, 1);
    lineSys.isPickable = false;
    if (lastGlTFRoot) lineSys.parent = lastGlTFRoot;
    console.log(`[normals] ${allLines.length} vertex normals, length=${normalLen.toFixed(2)}mm`);
}

// Legacy Z-loop function (replaced by color-coded edges)
function showZPerpendicularLoops(mesh) {
    const overlayName = '_z_loops';
    scene.meshes.filter(m => m.name === overlayName).forEach(m => m.dispose());
    if (!mesh) return;

    const positions = mesh.getVerticesData(BABYLON.VertexBuffer.PositionKind);
    const indices = mesh.getIndices();
    const faceIdData = mesh.getVerticesData('_FACE_ID');
    if (!positions || !indices || !faceIdData) return;

    // 1. Find all B-Rep boundary edges (different face IDs on adjacent triangles)
    const q = v => Math.round(v * 1e4);
    const pk = i => `${q(positions[i*3])}_${q(positions[i*3+1])}_${q(positions[i*3+2])}`;
    const edgeMap = new Map();
    for (let t = 0; t < indices.length; t += 3) {
        const fid = faceIdData[indices[t]];
        for (let e = 0; e < 3; e++) {
            const a = indices[t + e], b = indices[t + (e + 1) % 3];
            const ka = pk(a), kb = pk(b);
            const key = ka < kb ? `${ka}|${kb}` : `${kb}|${ka}`;
            if (!edgeMap.has(key)) edgeMap.set(key, []);
            edgeMap.get(key).push({ fid, a, b });
        }
    }

    // Extract boundary edges (different face IDs)
    const boundaryEdges = []; // [{a, b}] with vertex indices
    for (const [, entries] of edgeMap) {
        if (entries.length < 2) continue;
        const fids = new Set(entries.map(e => e.fid));
        if (fids.size > 1) {
            boundaryEdges.push({ a: entries[0].a, b: entries[0].b });
        }
    }

    // 2. Build adjacency graph from boundary edges using position keys
    const posKey = i => pk(i);
    const adj = new Map(); // posKey -> [{ posKey, vtxIdx }]
    for (const { a, b } of boundaryEdges) {
        const ka = posKey(a), kb = posKey(b);
        if (!adj.has(ka)) adj.set(ka, []);
        if (!adj.has(kb)) adj.set(kb, []);
        adj.get(ka).push({ key: kb, idx: b });
        adj.get(kb).push({ key: ka, idx: a });
    }

    // 3. Trace edge loops
    const visited = new Set();
    const loops = [];
    for (const [startKey] of adj) {
        if (visited.has(startKey)) continue;
        // Follow chain
        const loop = [];
        let curKey = startKey;
        let prevKey = null;
        let stuck = false;
        for (let step = 0; step < 100000; step++) {
            if (visited.has(curKey) && loop.length > 2) {
                // Closed loop
                if (curKey === startKey) loops.push([...loop]);
                break;
            }
            visited.add(curKey);
            const neighbors = adj.get(curKey) || [];
            // Pick first unvisited neighbor (or back to start for closing)
            let next = null;
            for (const n of neighbors) {
                if (n.key === prevKey) continue;
                if (n.key === startKey && loop.length > 2) {
                    loop.push(n.idx);
                    next = { key: startKey };
                    break;
                }
                if (!visited.has(n.key)) {
                    loop.push(n.idx);
                    next = n;
                    break;
                }
            }
            if (!next) break;
            prevKey = curKey;
            curKey = next.key;
        }
    }

    // 4. Filter: keep only loops where all vertices fit in a Z-perpendicular plane
    const zTolerance = 0.5; // mm — max Z variation within a loop
    const linePoints = [];
    let loopCount = 0;

    for (const loop of loops) {
        if (loop.length < 3) continue;
        // Check Z spread
        let zMin = Infinity, zMax = -Infinity;
        for (const idx of loop) {
            const z = positions[idx * 3 + 2];
            if (z < zMin) zMin = z;
            if (z > zMax) zMax = z;
        }
        if (zMax - zMin > zTolerance) continue;

        // This loop lies in a Z-perpendicular plane — draw it
        loopCount++;
        for (let i = 0; i < loop.length; i++) {
            const a = loop[i], b = loop[(i + 1) % loop.length];
            linePoints.push(
                new BABYLON.Vector3(positions[a*3], positions[a*3+1], positions[a*3+2]),
                new BABYLON.Vector3(positions[b*3], positions[b*3+1], positions[b*3+2]),
            );
        }
    }

    console.log(`[z-loops] Found ${loops.length} loops, ${loopCount} Z-perpendicular (tol=${zTolerance}mm)`);

    if (linePoints.length > 0) {
        const lines = BABYLON.MeshBuilder.CreateLineSystem(overlayName, {
            lines: Array.from({length: linePoints.length / 2}, (_, i) =>
                [linePoints[i * 2], linePoints[i * 2 + 1]]),
        }, scene);
        lines.color = new BABYLON.Color3(1.0, 0.15, 0.15); // red
        lines.isPickable = false;
        lines.parent = mesh.parent || mesh;
    }
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
        mat.backFaceCulling = false;
        currentMesh.material = mat;
    }
});

$('showWireframe').addEventListener('change', () => {
    const checked = $('showWireframe').checked;
    scene?.meshes.forEach(m => {
        if (m.getTotalVertices() > 0 && m.material && m.getVerticesData(BABYLON.VertexBuffer.NormalKind))
            m.material.wireframe = checked;
    });
});

$('showParam')?.addEventListener('change', async () => {
    if ($('showParam').checked && state.resultGlb) {
        // Reload parameterized mesh with checkerboard
        await loadGlb(new Uint8Array(state.resultGlb), true);
    } else if (state.inputGlb) {
        // Reload original tessellated mesh (with correct OCC normals)
        await loadGlb(new Uint8Array(state.inputGlb), false);
    }
});

$('showSeams')?.addEventListener('change', () => {
    if ($('showSeams').checked) {
        if (currentMesh) showSeamLines();
    } else {
        scene?.meshes.filter(m => m.name === '_seam_lines').forEach(m => m.dispose());
    }
});

$('showFaceEdges')?.addEventListener('change', () => {
    showFaceEdges(currentMesh);
});

$('showNormals')?.addEventListener('change', () => {
    if ($('showNormals').checked) {
        showVertexNormals();
    } else {
        scene?.meshes.filter(m => m.name === '_vertex_normals').forEach(m => m.dispose());
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

// ============================================================
// ViewCube — BabylonJS on a separate overlay canvas
// ============================================================

class ViewCube {
    constructor(mainScene) {
        this.mainScene = mainScene;
        this.camera = mainScene.activeCamera;
        this.animating = false;
        this.isOrtho = false;

        // Create a separate engine on the overlay canvas
        const vcCanvas = document.getElementById('vcCanvas');
        this.vcEngine = new BABYLON.Engine(vcCanvas, true, { alpha: true });
        this.vcScene = new BABYLON.Scene(this.vcEngine);
        this.vcScene.clearColor = new BABYLON.Color4(0, 0, 0, 0); // transparent

        // Orthographic camera
        this.vcCamera = new BABYLON.ArcRotateCamera('vcCam', 0, 0, 5, BABYLON.Vector3.Zero(), this.vcScene);
        this.vcCamera.mode = BABYLON.Camera.ORTHOGRAPHIC_CAMERA;
        this.vcCamera.orthoLeft = -1.5;
        this.vcCamera.orthoRight = 1.5;
        this.vcCamera.orthoTop = 1.5;
        this.vcCamera.orthoBottom = -1.5;
        this.vcCamera.inputs.clear();

        // Light
        const light = new BABYLON.HemisphericLight('vcLight', new BABYLON.Vector3(0.5, 1, 0.8), this.vcScene);
        light.intensity = 1.0;
        light.groundColor = new BABYLON.Color3(0.2, 0.25, 0.3);

        this._buildCube();

        // Render loop for the VC engine
        this.vcEngine.runRenderLoop(() => {
            this._syncCamera();
            this.vcScene.render();
        });

        // Click handling directly on the VC canvas — no viewport mapping needed
        vcCanvas.addEventListener('pointerdown', (evt) => this._handleClick(evt));

        // ORTHO/PERSP label
        document.getElementById('vcModeLabel')?.addEventListener('click', () => this.toggleProjection());

        // Camera angle definitions
        // ArcRotateCamera: alpha=0 → camera on +X axis, alpha=PI/2 → camera on +Z axis
        // Front = looking at the Z+ face of the model → camera at +Z → alpha = PI/2
        const PI = Math.PI;
        this.views = {
            // Faces
            front:  { alpha: PI / 2,   beta: PI / 2 },   // camera at +Z
            back:   { alpha: PI * 1.5, beta: PI / 2 },   // camera at -Z
            top:    { alpha: PI / 2,   beta: 0.001 },     // camera above
            bottom: { alpha: PI / 2,   beta: PI - 0.001 },// camera below
            right:  { alpha: 0,        beta: PI / 2 },    // camera at +X
            left:   { alpha: PI,       beta: PI / 2 },    // camera at -X
            // Edges
            'front-top':     { alpha: PI / 2,   beta: PI / 4 },
            'front-bottom':  { alpha: PI / 2,   beta: PI * 3 / 4 },
            'front-right':   { alpha: PI * 0.25, beta: PI / 2 },
            'front-left':    { alpha: PI * 0.75, beta: PI / 2 },
            'back-top':      { alpha: PI * 1.5, beta: PI / 4 },
            'back-bottom':   { alpha: PI * 1.5, beta: PI * 3 / 4 },
            'back-right':    { alpha: PI * 1.75, beta: PI / 2 },
            'back-left':     { alpha: PI * 1.25, beta: PI / 2 },
            'top-right':     { alpha: 0,         beta: PI / 4 },
            'top-left':      { alpha: PI,        beta: PI / 4 },
            'bottom-right':  { alpha: 0,         beta: PI * 3 / 4 },
            'bottom-left':   { alpha: PI,        beta: PI * 3 / 4 },
            // Corners (front = camera at +Z side)
            'front-top-right':    { alpha: PI * 0.25, beta: PI / 4 },
            'front-top-left':     { alpha: PI * 0.75, beta: PI / 4 },
            'front-bottom-right': { alpha: PI * 0.25, beta: PI * 3 / 4 },
            'front-bottom-left':  { alpha: PI * 0.75, beta: PI * 3 / 4 },
            'back-top-right':     { alpha: PI * 1.75, beta: PI / 4 },
            'back-top-left':      { alpha: PI * 1.25, beta: PI / 4 },
            'back-bottom-right':  { alpha: PI * 1.75, beta: PI * 3 / 4 },
            'back-bottom-left':   { alpha: PI * 1.25, beta: PI * 3 / 4 },
        };
    }

    _buildCube() {
        const S = 0.5; // half size

        // Create 6 face planes using CreatePlane facing +Z, then rotate into position
        // For faces where rotation mirrors the texture, flip scaling.x
        // Build a single box and create 6 dynamic textures for its faces
        // faceUV maps each box face to a specific region of a texture atlas
        // Box face order in BabylonJS: 0=back(Z-), 1=front(Z+), 2=right(X+), 3=left(X-), 4=top(Y+), 5=bottom(Y-)
        // Labels are SWAPPED: the Z- face of the cube shows "FRONT" because when
        // the camera looks from +Z (front view), you see the Z- side of the cube.
        const faceLabels = ['FRONT', 'BACK', 'RIGHT', 'LEFT', 'TOP', 'BOTTOM'];
        const faceNames =  ['front', 'back', 'right', 'left', 'top', 'bottom'];

        // Create a texture atlas: 6 labels in a 3x2 grid, each 128x128 → 384x256
        const atlasW = 384, atlasH = 256, cellW = 128, cellH = 128;
        const atlasTex = new BABYLON.DynamicTexture('vcAtlas', { width: atlasW, height: atlasH }, this.vcScene, false);
        const ctx = atlasTex.getContext();

        for (let i = 0; i < 6; i++) {
            const col = i % 3, row = Math.floor(i / 3);
            const ox = col * cellW, oy = row * cellH;
            ctx.fillStyle = 'rgba(40, 70, 120, 0.9)';
            ctx.fillRect(ox, oy, cellW, cellH);
            ctx.strokeStyle = 'rgba(100, 160, 220, 0.8)';
            ctx.lineWidth = 3;
            ctx.strokeRect(ox + 2, oy + 2, cellW - 4, cellH - 4);
            ctx.font = 'bold 22px Segoe UI, sans-serif';
            ctx.fillStyle = '#c0d8f0';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillText(faceLabels[i], ox + cellW/2, oy + cellH/2);
        }
        atlasTex.update();

        // Map each face to its atlas cell
        const faceUV = [];
        for (let i = 0; i < 6; i++) {
            const col = i % 3, row = Math.floor(i / 3);
            const u0 = col * cellW / atlasW, u1 = (col + 1) * cellW / atlasW;
            const v0 = 1 - (row + 1) * cellH / atlasH, v1 = 1 - row * cellH / atlasH;
            faceUV.push(new BABYLON.Vector4(u0, v0, u1, v1));
        }

        const cube = BABYLON.MeshBuilder.CreateBox('vcCube', { size: 1, faceUV, wrap: true }, this.vcScene);
        const cubeMat = new BABYLON.StandardMaterial('vcCubeMat', this.vcScene);
        cubeMat.diffuseTexture = atlasTex;
        cubeMat.emissiveColor = new BABYLON.Color3(0.15, 0.25, 0.4);
        cubeMat.specularColor = new BABYLON.Color3(0.1, 0.1, 0.1);
        cube.material = cubeMat;
        cube.metadata = { viewName: '__cube__' };
        this.cubeMesh = cube;

        // For picking individual faces, use faceId from the pick result
        // BabylonJS box faceId: each face has 2 triangles, so faceId/2 gives the face index
        // Face index: 0=back, 1=front, 2=right, 3=left, 4=top, 5=bottom
        this.faceIndexToName = faceNames;

        // Edge strips (thin boxes along cube edges)
        this.edgeMeshes = {};
        const ES = 0.08;
        const edgeDefs = [
            { name: 'front-top',    pos: [0, S, S],   scale: [1+ES, ES, ES] },
            { name: 'front-bottom', pos: [0, -S, S],  scale: [1+ES, ES, ES] },
            { name: 'front-left',   pos: [-S, 0, S],  scale: [ES, 1+ES, ES] },
            { name: 'front-right',  pos: [S, 0, S],   scale: [ES, 1+ES, ES] },
            { name: 'back-top',     pos: [0, S, -S],  scale: [1+ES, ES, ES] },
            { name: 'back-bottom',  pos: [0, -S, -S], scale: [1+ES, ES, ES] },
            { name: 'back-left',    pos: [-S, 0, -S], scale: [ES, 1+ES, ES] },
            { name: 'back-right',   pos: [S, 0, -S],  scale: [ES, 1+ES, ES] },
            { name: 'top-right',    pos: [S, S, 0],   scale: [ES, ES, 1+ES] },
            { name: 'top-left',     pos: [-S, S, 0],  scale: [ES, ES, 1+ES] },
            { name: 'bottom-right', pos: [S, -S, 0],  scale: [ES, ES, 1+ES] },
            { name: 'bottom-left',  pos: [-S, -S, 0], scale: [ES, ES, 1+ES] },
        ];

        const edgeMat = new BABYLON.StandardMaterial('vcEdgeMat', this.vcScene);
        edgeMat.diffuseColor = new BABYLON.Color3(0.3, 0.5, 0.7);
        edgeMat.emissiveColor = new BABYLON.Color3(0.15, 0.3, 0.5);
        edgeMat.specularColor = BABYLON.Color3.Black();

        for (const ed of edgeDefs) {
            const box = BABYLON.MeshBuilder.CreateBox(`vcE_${ed.name}`, { size: 1 }, this.vcScene);
            box.position = new BABYLON.Vector3(...ed.pos);
            box.scaling = new BABYLON.Vector3(...ed.scale);
            box.material = edgeMat;
            box.metadata = { viewName: ed.name };
            this.edgeMeshes[ed.name] = box;
        }

        // Corner spheres
        this.cornerMeshes = {};
        const CS = 0.12;
        const cornerMat = new BABYLON.StandardMaterial('vcCornerMat', this.vcScene);
        cornerMat.diffuseColor = new BABYLON.Color3(0.35, 0.55, 0.75);
        cornerMat.emissiveColor = new BABYLON.Color3(0.2, 0.35, 0.55);
        cornerMat.specularColor = BABYLON.Color3.Black();

        for (const sx of [-1, 1]) {
            for (const sy of [-1, 1]) {
                for (const sz of [-1, 1]) {
                    const fname = (sz > 0 ? 'front' : 'back') + '-' +
                                  (sy > 0 ? 'top' : 'bottom') + '-' +
                                  (sx > 0 ? 'right' : 'left');
                    const sph = BABYLON.MeshBuilder.CreateSphere(`vcC_${fname}`, { diameter: CS * 2, segments: 6 }, this.vcScene);
                    sph.position = new BABYLON.Vector3(sx * S, sy * S, sz * S);
                    sph.material = cornerMat;
                    sph.metadata = { viewName: fname };
                    this.cornerMeshes[fname] = sph;
                }
            }
        }
    }

    _syncCamera() {
        if (!this.camera) this.camera = this.mainScene.activeCamera;
        if (!this.camera) return;
        this.vcCamera.alpha = this.camera.alpha;
        this.vcCamera.beta = this.camera.beta;
        if (this.isOrtho) this._updateOrthoFrustum();
    }

    _handleClick(evt) {
        // Coordinates are local to the vcCanvas
        const pick = this.vcScene.pick(evt.offsetX, evt.offsetY);
        if (!pick.hit || !pick.pickedMesh) return;

        let viewName = null;
        if (pick.pickedMesh === this.cubeMesh && pick.faceId != null) {
            // Box face: faceId is the triangle index, /2 gives the face index
            const faceIdx = Math.floor(pick.faceId / 2);
            viewName = this.faceIndexToName[faceIdx];
        } else if (pick.pickedMesh.metadata?.viewName) {
            // Edge or corner mesh
            viewName = pick.pickedMesh.metadata.viewName;
        }
        if (viewName && this.views[viewName]) {
            this._animateTo(this.views[viewName].alpha, this.views[viewName].beta, viewName);
        }
    }

    _animateTo(targetAlpha, targetBeta, viewName) {
        if (this.animating) return;
        this.animating = true;

        const cam = this.camera;
        if (!cam) { this.animating = false; return; }

        if (!this.isOrtho) this._setOrthoProjection(true);

        const startAlpha = cam.alpha;
        const startBeta = cam.beta;
        const duration = 500;
        const startTime = performance.now();

        let da = targetAlpha - startAlpha;
        while (da > Math.PI) da -= Math.PI * 2;
        while (da < -Math.PI) da += Math.PI * 2;
        const resolvedTargetAlpha = startAlpha + da;

        const easeInOut = t => t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

        const animate = () => {
            const elapsed = performance.now() - startTime;
            const t = Math.min(elapsed / duration, 1);
            const et = easeInOut(t);
            cam.alpha = startAlpha + (resolvedTargetAlpha - startAlpha) * et;
            cam.beta = startBeta + (targetBeta - startBeta) * et;
            if (this.isOrtho) this._updateOrthoFrustum();
            if (t < 1) {
                requestAnimationFrame(animate);
            } else {
                cam.alpha = ((resolvedTargetAlpha % (Math.PI * 2)) + Math.PI * 2) % (Math.PI * 2);
                cam.beta = targetBeta;
                this.animating = false;
            }
        };
        requestAnimationFrame(animate);
    }

    toggleProjection() {
        this._setOrthoProjection(!this.isOrtho);
    }

    _setOrthoProjection(ortho) {
        const cam = this.camera;
        if (!cam) return;
        if (ortho) {
            cam.mode = BABYLON.Camera.ORTHOGRAPHIC_CAMERA;
            this._updateOrthoFrustum();
            this.isOrtho = true;
        } else {
            cam.mode = BABYLON.Camera.PERSPECTIVE_CAMERA;
            this.isOrtho = false;
        }
        const label = document.getElementById('vcModeLabel');
        if (label) label.textContent = this.isOrtho ? 'ORTHO' : 'PERSP';
    }

    _updateOrthoFrustum() {
        const cam = this.camera;
        if (!cam || cam.mode !== BABYLON.Camera.ORTHOGRAPHIC_CAMERA) return;
        const canvas = this.mainScene.getEngine().getRenderingCanvas();
        if (!canvas) return;
        const aspect = canvas.width / canvas.height;
        const halfHeight = cam.radius * 0.5;
        cam.orthoLeft = -halfHeight * aspect;
        cam.orthoRight = halfHeight * aspect;
        cam.orthoTop = halfHeight;
        cam.orthoBottom = -halfHeight;
    }

    onCameraChange() {
        if (this.isOrtho) this._updateOrthoFrustum();
    }
}

// Global ViewCube instance
let viewCube = null;

function initViewCube() {
    if (!scene || !scene.activeCamera) return;
    viewCube = new ViewCube(scene);

    // Also update ortho frustum on zoom (radius change)
    scene.registerBeforeRender(() => {
        if (viewCube) viewCube.onCameraChange();
    });
}

// --- Init ---
(async () => {
    try {
        await apiHealth();
        setStatus('Server connected. Load a file to begin.', '');
    } catch (e) {
        setStatus('Server not available at ' + API, 'error');
    }
    // Initialize ViewCube after scene is ready
    initViewCube();
})();

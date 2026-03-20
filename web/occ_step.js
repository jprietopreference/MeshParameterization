// STEP → GLB converter using occt-import-js
// Lightweight OCC WASM (7.3 MB) with built-in tessellation.

let occPromise = null;

async function initOCCT() {
    if (occPromise) return occPromise;
    occPromise = (async () => {
        // occt-import-js exports a factory function as CommonJS module
        const occtModule = await import('occt-import-js');
        const factory = occtModule.default || occtModule;
        const occt = await factory({
            locateFile(path) {
                if (path.endsWith('.wasm')) return '/wasm/occt-import-js.wasm';
                return path;
            }
        });
        return occt;
    })();
    return occPromise;
}

/**
 * Convert a STEP file (ArrayBuffer) to GLB (ArrayBuffer).
 * @param {ArrayBuffer} stepBuffer - The STEP file data
 * @param {number} deflection - Linear deflection for tessellation (default 0.001 = ~1mm for meter-scale)
 * @returns {ArrayBuffer} GLB file data
 */
export async function stepToGlb(stepBuffer, deflection = 1.0) {
    const occt = await initOCCT();

    const fileBuffer = new Uint8Array(stepBuffer);
    console.log(`[occ] Reading STEP file (${fileBuffer.length} bytes)...`);

    const params = {
        linearUnit: 'millimeter',
        linearDeflectionType: 'absolute_value',
        linearDeflection: deflection,
        angularDeflection: 0.5,
    };

    const result = occt.ReadStepFile(fileBuffer, params);

    if (!result.success) {
        throw new Error('STEP import failed');
    }

    console.log(`[occ] STEP imported: ${result.meshes.length} mesh(es)`);

    // Merge all meshes into one
    const allVerts = [];
    const allTris = [];
    let vertexOffset = 0;

    for (const mesh of result.meshes) {
        const pos = mesh.attributes.position.array;
        const idx = mesh.index.array;
        const nv = pos.length / 3;

        for (let i = 0; i < pos.length; i++) {
            allVerts.push(pos[i]);
        }

        for (let i = 0; i < idx.length; i++) {
            allTris.push(idx[i] + vertexOffset);
        }

        vertexOffset += nv;
    }

    const nv = allVerts.length / 3;
    const nf = allTris.length / 3;
    console.log(`[occ] Merged: ${nv} vertices, ${nf} triangles`);

    return buildGlb(allVerts, nv, allTris, nf);
}

function buildGlb(verts, nv, tris, nf) {
    const posSize = nv * 12;
    const idxSize = nf * 12;
    let bufSize = posSize + idxSize;
    while (bufSize % 4) bufSize++;

    const buf = new ArrayBuffer(bufSize);
    const posView = new Float32Array(buf, 0, nv * 3);
    for (let i = 0; i < nv * 3; i++) posView[i] = verts[i];

    const idxView = new Uint32Array(buf, posSize, nf * 3);
    for (let i = 0; i < nf * 3; i++) idxView[i] = tris[i];

    const min = [Infinity, Infinity, Infinity];
    const max = [-Infinity, -Infinity, -Infinity];
    for (let i = 0; i < nv; i++) {
        for (let k = 0; k < 3; k++) {
            min[k] = Math.min(min[k], verts[i * 3 + k]);
            max[k] = Math.max(max[k], verts[i * 3 + k]);
        }
    }

    const json = JSON.stringify({
        asset: { version: '2.0', generator: 'OCCT-Import-JS' },
        scene: 0, scenes: [{ nodes: [0] }], nodes: [{ mesh: 0 }],
        meshes: [{ primitives: [{ attributes: { POSITION: 0 }, indices: 1, mode: 4 }] }],
        accessors: [
            { bufferView: 0, componentType: 5126, count: nv, type: 'VEC3', min, max },
            { bufferView: 1, componentType: 5125, count: nf * 3, type: 'SCALAR' },
        ],
        bufferViews: [
            { buffer: 0, byteOffset: 0, byteLength: posSize, target: 34962 },
            { buffer: 0, byteOffset: posSize, byteLength: idxSize, target: 34963 },
        ],
        buffers: [{ byteLength: bufSize }],
    });

    let jsonStr = json;
    while (jsonStr.length % 4) jsonStr += ' ';
    const jsonBytes = new TextEncoder().encode(jsonStr);

    const totalLen = 12 + 8 + jsonBytes.length + 8 + bufSize;
    const glb = new ArrayBuffer(totalLen);
    const view = new DataView(glb);
    let offset = 0;

    view.setUint32(offset, 0x46546C67, true); offset += 4;
    view.setUint32(offset, 2, true); offset += 4;
    view.setUint32(offset, totalLen, true); offset += 4;

    view.setUint32(offset, jsonBytes.length, true); offset += 4;
    view.setUint32(offset, 0x4E4F534A, true); offset += 4;
    new Uint8Array(glb, offset, jsonBytes.length).set(jsonBytes); offset += jsonBytes.length;

    view.setUint32(offset, bufSize, true); offset += 4;
    view.setUint32(offset, 0x004E4942, true); offset += 4;
    new Uint8Array(glb, offset, bufSize).set(new Uint8Array(buf));

    return glb;
}

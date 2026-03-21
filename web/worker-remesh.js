// Web Worker for Gmsh remeshing
// Runs the 5.5 MB Gmsh WASM off the main thread.

let gmshMod = null;

async function loadGmsh() {
    importScripts('/wasm/gmsh_remesh.js');
    const factory = self.createGmshModule;
    if (!factory) throw new Error('createGmshModule not found');
    return await factory({
        locateFile(path) {
            if (path.endsWith('.wasm')) return '/wasm/gmsh_remesh.wasm';
            return path;
        }
    });
}

self.onmessage = async function(e) {
    const { id, action, data } = e.data;

    try {
        if (action === 'remesh') {
            if (!gmshMod) gmshMod = await loadGmsh();
            const input = new Uint8Array(data.glb);
            const result = gmshMod.remeshGlb(input, data.maxTriangles || 5000);
            if (!result || result.length === 0) {
                const metrics = gmshMod.getMetrics();
                throw new Error('Remeshing failed: ' + metrics);
            }
            const metrics = gmshMod.getMetrics();
            const output = new Uint8Array(result.length);
            for (let i = 0; i < result.length; i++) output[i] = result[i];
            self.postMessage({ id, ok: true, glb: output.buffer, metrics }, [output.buffer]);
        } else {
            throw new Error(`Unknown action: ${action}`);
        }
    } catch (err) {
        self.postMessage({ id, ok: false, error: err.message || String(err) });
    }
};

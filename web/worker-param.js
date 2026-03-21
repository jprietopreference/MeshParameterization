// Web Worker for mesh parameterization (heat + CGAL)
// Runs WASM modules off the main thread to avoid UI freezes and crashes.

let meshparamMod = null;
let cgalparamMod = null;

async function loadModule(name) {
    const jsUrl = `/wasm/${name}.js`;
    importScripts(jsUrl);

    const factoryName = name === 'meshparam' ? 'createMeshparamModule' : 'createCgalparamModule';
    const factory = self[factoryName];
    if (!factory) throw new Error(`Factory ${factoryName} not found`);
    return await factory({
        locateFile(path) {
            if (path.endsWith('.wasm')) return `/wasm/${name}.wasm`;
            return path;
        }
    });
}

self.onmessage = async function(e) {
    const { id, action, data } = e.data;

    try {
        if (action === 'parameterize_heat') {
            if (!meshparamMod) meshparamMod = await loadModule('meshparam');
            const input = new Uint8Array(data.glb);
            const result = meshparamMod.parameterizeGltf(
                input, false,
                data.viewWeighted || false,
                data.viewDir?.[0] || 0,
                data.viewDir?.[1] || 0,
                data.viewDir?.[2] || 1
            );
            const metrics = meshparamMod.getMetrics();
            // Copy result to transferable buffer
            const output = new Uint8Array(result.length);
            for (let i = 0; i < result.length; i++) output[i] = result[i];
            self.postMessage({ id, ok: true, glb: output.buffer, metrics }, [output.buffer]);

        } else if (action === 'parameterize_cgal') {
            if (!cgalparamMod) cgalparamMod = await loadModule('cgalparam');
            const input = new Uint8Array(data.glb);
            const result = cgalparamMod.parameterizeGltf(input, data.method || 'conformal');
            const metrics = cgalparamMod.getMetrics();
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

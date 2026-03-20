import { defineConfig } from 'vite';

export default defineConfig({
    root: '.',
    server: {
        port: 5199,
        strictPort: true,
    },
    optimizeDeps: {
        exclude: ['opencascade.js'],
    },
    build: {
        // Don't inline WASM files
        assetsInlineLimit: 0,
    },
});

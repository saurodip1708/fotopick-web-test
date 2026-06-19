import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import fs from 'fs';
import path from 'path';

export default defineConfig({
  plugins: [
    react(),
    {
      name: 'serve-wasm-helpers',
      configureServer(server) {
        server.middlewares.use((req, res, next) => {
          const url = new URL(req.url!, 'http://localhost');
          const pathname = url.pathname;
          if (pathname.startsWith('/ort-wasm-simd-threaded')) {
            const filePath = path.join(process.cwd(), 'public', pathname);
            if (fs.existsSync(filePath)) {
              const ext = path.extname(pathname);
              res.setHeader('Content-Type', ext === '.wasm' ? 'application/wasm' : 'application/javascript');
              res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
              res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
              res.end(fs.readFileSync(filePath));
              return;
            }
          }
          next();
        });
      },
    },
  ],
  server: {
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'require-corp',
    },
  },
  worker: {
    format: 'es',
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web'],
  },
});

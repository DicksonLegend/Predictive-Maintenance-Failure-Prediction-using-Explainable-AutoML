import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      '/predict': 'http://localhost:8000',
      '/explain': 'http://localhost:8000',
      '/what-if': 'http://localhost:8000',
      '/text-explain': 'http://localhost:8000',
      '/samples': 'http://localhost:8000',
      '/global-explanations': 'http://localhost:8000',
      '/image-demo': 'http://localhost:8000',
      '/model-card': 'http://localhost:8000',
      '/model': 'http://localhost:8000',
      '/health': 'http://localhost:8000',
      '/version': 'http://localhost:8000',
      '/artifacts': 'http://localhost:8000',
    }
  },
  build: {
    outDir: 'dist',
    emptyOutDir: true,
  }
});

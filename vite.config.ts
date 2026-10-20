import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
export default defineConfig({ plugins: [react()], build:{emptyOutDir:false}, server: { watch: { ignored: ['**/.local/**'] }, proxy: { '/api': 'http://127.0.0.1:47821', '/examples': 'http://127.0.0.1:47821' } } });

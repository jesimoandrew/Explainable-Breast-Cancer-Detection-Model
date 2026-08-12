import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// /api/* is proxied to the FastAPI inference server (app/server.py) so the
// browser only ever talks to one origin.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8000',
        changeOrigin: true,
      },
    },
  },
})

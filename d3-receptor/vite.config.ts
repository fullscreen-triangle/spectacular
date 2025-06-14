import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  build: {
    minify: 'terser', // Use Terser for better minification
    terserOptions: {
      compress: {
        drop_console: true, // Remove console.log in production
        drop_debugger: true, // Remove debugger statements
      },
      output: {
        comments: false // Remove comments
      }
    },
    rollupOptions: {
      output: {
        manualChunks: {
          vendor: ['react', 'react-dom'],
          services: [
            './src/services/models/ModelService.ts',
            './src/services/models/AnthropicService.ts',
            './src/services/models/OpenAIService.ts',
            './src/services/models/HuggingFaceService.ts',
            './src/services/models/ModelRegistry.ts'
          ]
        },
        chunkFileNames: 'assets/[name]-[hash].js',
      }
    },
    chunkSizeWarningLimit: 1000, // Increase warning limit
    sourcemap: false, // Disable sourcemaps in production
    reportCompressedSize: true, // Report compressed size
  },
  optimizeDeps: {
    include: ['react', 'react-dom'] // Pre-bundle these dependencies
  },
  server: {
    open: true, // Open browser on server start
    port: 3000 // Use port 3000
  }
}) 
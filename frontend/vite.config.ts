import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
const target = process.env.API_PROXY_TARGET || "http://127.0.0.1:8000";
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target, changeOrigin: true },
      "/ws": { target, ws: true },
    },
  },
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          candles: ["lightweight-charts"],
          metrics: ["recharts"],
          motion: ["framer-motion"],
        },
      },
    },
  },
});

import { defineConfig, devices } from "@playwright/test";
import { existsSync } from "node:fs";

const hasSystemChrome = [
  "/usr/bin/google-chrome",
  "/usr/bin/google-chrome-stable",
  "/opt/google/chrome/chrome"
].some((path) => existsSync(path));

const chromeChannel = hasSystemChrome || process.env.PLAYWRIGHT_USE_SYSTEM_CHROME === "1" ? "chrome" : undefined;

export default defineConfig({
  testDir: "./tests",
  timeout: 30_000,
  expect: {
    timeout: 5_000
  },
  use: {
    baseURL: "http://127.0.0.1:5173",
    trace: "on-first-retry"
  },
  webServer: {
    command: "npm run dev -- --host 127.0.0.1",
    url: "http://127.0.0.1:5173",
    reuseExistingServer: !process.env.CI,
    timeout: 120_000
  },
  projects: [
    {
      name: "chromium",
      use: {
        ...devices["Desktop Chrome"],
        ...(chromeChannel ? { channel: chromeChannel } : {})
      }
    }
  ]
});

// Serve the real files under a project prefix, matching GitHub Pages.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const { chromium } = require("playwright");

test("GitHub Pages deployment layouts", { timeout: 60000 }, async (t) => {
  const directory = fs.mkdtempSync(path.join(os.tmpdir(), "mathsphere-pages-"));
  fs.symlinkSync(
    path.resolve(__dirname, "../static"),
    path.join(directory, "mathsphere-web"),
    "dir",
  );
  fs.symlinkSync(
    path.resolve(__dirname, ".."),
    path.join(directory, "branch-site"),
    "dir",
  );
  t.after(() => fs.rmSync(directory, { recursive: true, force: true }));
  const server = spawn(
    "python3",
    [
      "-u",
      "-m",
      "http.server",
      "0",
      "--bind",
      "127.0.0.1",
      "--directory",
      directory,
    ],
    { stdio: ["ignore", "pipe", "ignore"] },
  );
  t.after(() => server.kill());
  const port = await new Promise((resolve, reject) => {
    const timeout = setTimeout(
      () => reject(new Error("Static server did not start")),
      10000,
    );
    server.once("error", (error) => {
      clearTimeout(timeout);
      reject(error);
    });
    server.stdout.on("data", (chunk) => {
      const match = chunk.toString().match(/port (\d+)/);
      if (match) {
        clearTimeout(timeout);
        resolve(match[1]);
      }
    });
  });
  const base = `http://127.0.0.1:${port}`;
  const browser = await chromium.launch({
    executablePath: process.env.BROWSER_EXECUTABLE || undefined,
    args: ["--no-sandbox"],
  });
  t.after(() => browser.close());
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1000 },
    reducedMotion: "reduce",
  });
  const page = await context.newPage();
  const errors = [],
    failedAssets = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("response", (response) => {
    if (response.status() >= 400 && !response.url().includes("/api/"))
      failedAssets.push(response.url());
  });

  await t.test(
    "artifact loads styles, scripts, fonts, and math under a repository prefix",
    async () => {
      await page.goto(base + "/mathsphere-web/", { waitUntil: "networkidle" });
      await page
        .getByRole("heading", { name: /A good day to be curious/ })
        .waitFor();
      assert.equal(await page.locator(".tool-card").count(), 4);
      assert.equal(
        await page.evaluate(() => typeof renderMathInElement),
        "function",
      );
      assert.equal(
        await page.evaluate(async () => {
          await document.fonts.ready;
          return document.fonts.check('400 14px "DM Sans"');
        }),
        true,
      );
      assert.equal(
        await page
          .locator(".sidebar")
          .evaluate((el) => getComputedStyle(el).position),
        "fixed",
      );
      await page
        .getByRole("link", { name: "AI Tutor AI", exact: true })
        .click();
      await page
        .getByRole("heading", { name: "What’s on your mind?" })
        .waitFor();
      await page.reload({ waitUntil: "networkidle" });
      await page
        .getByRole("heading", { name: "What’s on your mind?" })
        .waitFor();
    },
  );

  await t.test(
    "branch-based publishing has an entry page at the repository root",
    async () => {
      await page.goto(base + "/branch-site/", { waitUntil: "networkidle" });
      await page.waitForURL(base + "/branch-site/static/");
      await page
        .getByRole("heading", { name: /A good day to be curious/ })
        .waitFor();
    },
  );

  await t.test(
    "configured external backend receives API requests instead of Pages",
    async () => {
      const requests = [];
      await page.route("**/config.js", (route) =>
        route.fulfill({
          contentType: "application/javascript",
          body: 'window.MATHSPHERE_CONFIG = {apiBaseUrl: "https://backend.example.test/"};',
        }),
      );
      await page.route("https://backend.example.test/api/**", (route) => {
        const headers = {
          "access-control-allow-origin": "*",
          "access-control-allow-methods": "GET, POST, OPTIONS",
          "access-control-allow-headers": "Content-Type",
        };
        if (route.request().method() === "OPTIONS")
          return route.fulfill({ status: 204, headers });
        requests.push(route.request().url());
        const json = route.request().url().endsWith("/health")
          ? { services: { groq: true, gemini: true } }
          : { success: true, count: 0, questions: [] };
        return route.fulfill({ headers, json });
      });
      await page.goto(base + "/mathsphere-web/#pyqs", {
        waitUntil: "networkidle",
      });
      await page.locator('#tool-form [type="submit"]').click();
      await page
        .getByRole("heading", { name: "A little quiet here, for now." })
        .waitFor();
      assert.ok(requests.includes("https://backend.example.test/api/health"));
      assert.ok(requests.includes("https://backend.example.test/api/pyq/load"));
    },
  );
  assert.deepEqual(errors, []);
  assert.deepEqual(failedAssets, []);
});

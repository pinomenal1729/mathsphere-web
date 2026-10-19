// Browser regressions use deterministic API fixtures; they do not validate live AI.
const { test } = require("node:test");
const assert = require("node:assert/strict");
const path = require("node:path");
const { chromium } = require("playwright");

const base = process.env.MATHSPHERE_URL || "http://127.0.0.1:5000";

test("MathSphere learning workspace", { timeout: 60000 }, async (t) => {
  const browser = await chromium.launch({
    executablePath: process.env.BROWSER_EXECUTABLE || undefined,
    headless: true,
    args: ["--no-sandbox"],
  });
  t.after(() => browser.close());
  const context = await browser.newContext({
    viewport: { width: 1440, height: 1000 },
    reducedMotion: "reduce",
  });
  const page = await context.newPage();
  const errors = [];
  const failedAssets = [];
  let chatPayload;
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("response", (response) => {
    if (response.status() >= 400 && !response.url().includes("/api/"))
      failedAssets.push(response.url());
  });
  await page.route("**/api/health", (route) =>
    route.fulfill({
      json: {
        services: { groq: true, gemini: true, sympy: true, numpy: true },
      },
    }),
  );
  await page.route("**/api/formula", (route) =>
    route.fulfill({
      json: {
        answer:
          '## A useful idea\n\\[\n\\int_0^1 x^2 \\, dx = \\frac{1}{3}\n\\]\n<img src=x onerror="window.unsafe=true">',
      },
    }),
  );
  await page.route("**/api/chat", (route) => {
    chatPayload = route.request().postDataJSON();
    return route.fulfill({
      json: { answer: "## Key insight\n\\(2 + 2 = 4\\)" },
    });
  });
  await page.route("**/api/pyq/load", (route) =>
    route.fulfill({
      json: { success: true, questions: [], count: 0 },
    }),
  );

  await t.test("dashboard, search, and daily challenge persist", async () => {
    await page.goto(base, { waitUntil: "networkidle" });
    assert.equal(await page.locator(".tool-card").count(), 4);
    await page
      .getByRole("button", { name: "Linear Algebra", exact: true })
      .click();
    assert.equal(await page.locator(".learning-card").count(), 2);
    await page.locator('[data-answer="1"]').click();
    await page.locator(".challenge-feedback").waitFor();
    await page.reload();
    assert.equal(await page.locator(".challenge-feedback").count(), 1);
    await page.keyboard.press("Control+k");
    await page.locator("#global-search").fill("formula");
    await page.locator(".search-result").first().click();
    await page
      .getByRole("heading", { name: "Formula Sheets", exact: true })
      .waitFor();
  });

  await t.test(
    "multiline math renders and response HTML is escaped",
    async () => {
      await page.locator('#tool-form [type="submit"]').click();
      await page.locator("#tool-results .katex-display").waitFor();
      assert.equal(await page.locator("#tool-results img").count(), 0);
      assert.equal(await page.evaluate(() => window.unsafe), undefined);
      await page.getByRole("button", { name: "Save to library" }).click();
      await page
        .getByRole("link", { name: "Saved Library", exact: true })
        .click();
      await page.locator(".saved-card").waitFor();
      await page.reload();
      assert.equal(await page.locator(".saved-card").count(), 1);
      await page.getByRole("button", { name: "Revisit this idea" }).click();
      await page.locator(".katex-display").waitFor();
    },
  );

  await t.test(
    "image chat sends context and persists text without image data",
    async () => {
      await page.goto(base + "/#ask");
      await page
        .locator("#image-input")
        .setInputFiles(path.join(__dirname, "../static/logo.png"));
      await page.locator(".attachment-preview img").waitFor();
      await page.locator("#chat-input").fill("What is 2 + 2?");
      await page.locator("#chat-send").click();
      await page.locator(".message.assistant .katex").waitFor();
      assert.equal(chatPayload.messages.at(-1).content, "What is 2 + 2?");
      assert.equal(chatPayload.image_type, "image/png");
      assert.ok(chatPayload.image_b64.length > 0);
      const stored = await page.evaluate(() =>
        JSON.parse(localStorage.getItem("mathsphere.chats")),
      );
      assert.equal(stored[0].messages.length, 2);
      assert.ok(stored[0].messages.every((message) => !message.image));
    },
  );

  await t.test(
    "empty question bank and rate limits show useful feedback",
    async () => {
      await page.goto(base + "/#pyqs");
      await page.locator('#tool-form [type="submit"]').click();
      await page
        .getByRole("heading", { name: "A little quiet here, for now." })
        .waitFor();
      await page.route("**/api/formula", (route) =>
        route.fulfill({
          status: 429,
          json: { error: "Rate limit exceeded" },
        }),
      );
      await page.goto(base + "/#formula");
      await page.locator('#tool-form [type="submit"]').click();
      await page.locator(".error-panel").waitFor();
      assert.match(
        await page.locator(".error-panel").textContent(),
        /wait a moment/,
      );
      assert.equal(
        await page.locator('#tool-form [type="submit"]').isEnabled(),
        true,
      );
    },
  );

  await t.test("mobile drawer, focus, and layout remain usable", async () => {
    await page.setViewportSize({ width: 390, height: 844 });
    await page.goto(base + "/#dashboard");
    assert.equal(
      await page.locator("#sidebar").evaluate((el) => el.inert),
      true,
    );
    await page.locator("#menu-button").click();
    assert.equal(
      await page.locator("#sidebar").evaluate((el) => el.inert),
      false,
    );
    await page
      .getByRole("link", { name: "Graph Plotter", exact: true })
      .click();
    await page
      .getByRole("heading", { name: "Graph Plotter", exact: true })
      .waitFor();
    assert.equal(
      await page.locator("#sidebar").evaluate((el) => el.inert),
      true,
    );
    assert.equal(
      await page.evaluate(
        () => document.documentElement.scrollWidth > innerWidth,
      ),
      false,
    );
    assert.equal(await page.evaluate(() => document.body.style.overflow), "");
  });
  assert.deepEqual(errors, []);
  assert.deepEqual(failedAssets, []);
});

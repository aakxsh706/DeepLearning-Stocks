import { test, expect, type Page } from '@playwright/test';

async function generate(page: Page, symbol = 'AAPL') {
  await page.goto('/');
  await expect(page.getByLabel('Select stock').locator('option')).toHaveCount(2);
  await page.getByLabel('Select stock').selectOption(symbol);
  await expect(page.locator('#model-select')).toBeEnabled();
  await expect(page.locator('#model-select option').first()).toContainText('LSTM');
  await page.getByLabel('Forecast input cutoff').fill('2017-02-14');
  const response = page.waitForResponse(r => r.url().endsWith('/api/forecasts') && r.request().method() === 'POST');
  await page.getByRole('button', { name: 'Generate forecast', exact: true }).click();
  const result = await response;
  expect(result.status()).toBe(201);
  const forecast = await result.json();
  expect(forecast.mode).toBe('Replay');
  expect(forecast.telemetry.output.prices.median).toBe(forecast.median);
  await expect(page.locator('.forecast-number')).toContainText(new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(forecast.median));
  return forecast;
}

async function nav(page: Page, name: string) {
  const menu = page.getByRole('button', { name: 'Open navigation' });
  if (await menu.isVisible()) await menu.click();
  await page.getByRole('navigation', { name: 'Main navigation' }).getByRole('button', { name, exact: true }).click();
}

test('stock → forecast → real computation → output → stored replay', async ({ page }) => {
  const errors: string[] = [];
  page.on('pageerror', error => errors.push(error.message));
  const forecast = await generate(page, 'MSFT');
  await expect(page.getByRole('heading', { name: 'Microsoft Corporation' })).toBeVisible();
  await page.getByRole('button', { name: 'Explain this prediction' }).click();
  await expect(page.getByLabel('Replay timestep')).toHaveValue('0');
  await page.getByRole('button', { name: 'Step forward', exact: true }).click();
  await expect(page.getByLabel('Replay timestep')).toHaveValue('1');
  await expect(page.locator('.activation-readout strong')).toHaveText(forecast.telemetry.hidden[1][0].toFixed(4));
  await page.getByLabel('Hidden unit', { exact: true }).selectOption('3');
  await expect(page.locator('.activation-readout strong')).toHaveText(forecast.telemetry.hidden[1][3].toFixed(4));
  await page.getByLabel('Playback speed').selectOption('4');
  await page.getByRole('button', { name: 'Play replay', exact: true }).click();
  await expect.poll(async () => Number(await page.getByLabel('Replay timestep').inputValue())).toBeGreaterThan(1);
  await page.getByRole('button', { name: 'Pause replay', exact: true }).click();
  await page.getByRole('button', { name: 'Technical', exact: true }).click();
  await expect(page.locator('.input-tensors')).toContainText('return_1d');
  await expect(page.locator('.network-diagram')).toContainText(new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(forecast.median));
  await expect(page.locator('.chart-footer')).toContainText('Inspecting');
  await nav(page, 'Prediction archive');
  await page.getByRole('button', { name: `Replay prediction for ${forecast.target_time}` }).first().click();
  await expect(page.locator('#replay-select')).toHaveValue(forecast.id);
  await expect(page.getByLabel('Replay timestep')).toHaveValue('0');
  await expect(page.locator('.activation-readout strong')).toHaveText(forecast.telemetry.hidden[0][0].toFixed(4));
  expect(errors).toEqual([]);
});

test('what-if preserves original forecast and evaluation reports a baseline', async ({ page, request }) => {
  const forecast = await generate(page);
  const response = page.waitForResponse(r => r.url().endsWith('/api/what-if') && r.request().method() === 'POST');
  await page.getByRole('button', { name: 'Run sensitivity analysis' }).click();
  const scenario = await (await response).json();
  expect(scenario.sensitivity_only).toBe(true);
  expect(scenario.changes.length).toBeGreaterThan(0);
  await expect(page.locator('.sensitivity-results')).toContainText('Perturbed median');
  const saved = await request.get(`/api/forecasts/${forecast.id}`);
  expect((await saved.json()).median).toBe(forecast.median);
  await nav(page, 'Evaluation center');
  await expect(page.getByText('Persistence', { exact: false }).first()).toBeVisible();
  await expect(page.getByText('Historical test results only.', { exact: false })).toBeVisible();
  await expect(page.getByRole('heading', { name: 'Walk-forward windows' })).toBeVisible();
});

test('real bounded training job streams progress and saves a checkpoint', async ({ page, request }) => {
  await page.goto('/');
  await nav(page, 'Training studio');
  await page.getByLabel('Training architecture').selectOption('gru');
  await page.getByLabel('Training lookback').fill('10');
  await page.getByLabel('Training epochs').fill('2');
  await page.getByLabel('Training samples').fill('240');
  const response = page.waitForResponse(r => r.url().endsWith('/api/training') && r.request().method() === 'POST');
  await page.getByRole('button', { name: 'Train a demo model', exact: true }).click();
  const created = await response;
  expect(created.status()).toBe(202);
  const job = await created.json();
  await expect(page.locator('.checkpoint')).toContainText('Checkpoint saved', { timeout: 45_000 });
  const complete = await request.get(`/api/training/${job.id}`);
  const record = await complete.json();
  expect(record.status).toBe('completed');
  expect(record.history).toHaveLength(2);
  expect(record.history[1].gradient_norm).toBeGreaterThanOrEqual(0);
  expect(record.model_id).toMatch(/^gru-AAPL-/);
  await expect(page.locator('.connection-status')).toContainText('connected');
});

test('mobile and reduced-motion layout supports forecast and keyboard replay', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.emulateMedia({ reducedMotion: 'reduce' });
  await generate(page);
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
  await page.getByRole('button', { name: 'Explain this prediction' }).click();
  const step = page.getByRole('button', { name: 'Step forward', exact: true });
  await step.focus();
  await page.keyboard.press('Enter');
  await expect(page.getByLabel('Replay timestep')).toHaveValue('1');
  await expect(page.locator('.network-scroll')).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
  await nav(page, 'Prediction archive');
  await expect(page.getByRole('heading', { name: 'Prediction archive', exact: true }).last()).toBeVisible();
  expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth + 1)).toBe(true);
});

test('missing external credentials are explicit and replay remains usable', async ({ page, request }) => {
  const status = await request.get('/api/provider/status');
  test.skip((await status.json()).external.configured, 'This check targets the no-credentials replay workflow.');
  await page.goto('/');
  await page.getByText('Data sources & optional external quotes', { exact: true }).click();
  await expect(page.getByRole('region', { name: 'Historical dataset provenance' })).toContainText('Historical replay');
  await page.getByRole('button', { name: 'Request external quote for AAPL' }).click();
  await expect(page.locator('.data-source-panel [role="alert"]')).toContainText('Historical replay is available');
  await expect(page.getByRole('button', { name: 'Generate forecast', exact: true })).toBeEnabled();
});

test('learning glossary keeps keyboard focus and restores the opener', async ({ page }) => {
  await page.goto('/');
  const opener = page.getByRole('button', { name: 'Open learning resources', exact: true });
  await opener.click();
  const dialog = page.getByRole('dialog', { name: 'A field guide to neural networks' });
  await expect(dialog).toBeVisible();
  const close = dialog.getByRole('button', { name: 'Close dialog' });
  await expect(close).toBeFocused();
  await page.keyboard.press('Tab');
  await expect(close).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(dialog).not.toBeVisible();
  await expect(opener).toBeFocused();
});

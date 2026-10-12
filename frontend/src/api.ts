export async function api<T>(
  path: string,
  body?: unknown,
  method?: string,
): Promise<T> {
  const res = await fetch(`/api${path}`, {
    method: method ?? (body === undefined ? "GET" : "POST"),
    headers:
      body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) {
    let reason = `${res.status} ${res.statusText}`;
    try {
      const data = await res.json();
      reason =
        typeof data.detail === "string"
          ? data.detail
          : JSON.stringify(data.detail ?? data);
    } catch {
      /* Keep HTTP error. */
    }
    throw new Error(reason);
  }
  return res.json();
}
export const money = (value: number | undefined | null) =>
  value == null || !Number.isFinite(value)
    ? "—"
    : new Intl.NumberFormat("en-US", {
        style: "currency",
        currency: "USD",
        minimumFractionDigits: 2,
        maximumFractionDigits: 2,
      }).format(value);
export const num = (value: number | undefined | null, digits = 3) =>
  value == null || !Number.isFinite(value) ? "—" : value.toFixed(digits);
export const pct = (value: number | undefined | null) =>
  value == null || !Number.isFinite(value)
    ? "—"
    : `${(value * 100).toFixed(2)}%`;
export const shortDate = (value: string | undefined) =>
  !value
    ? "—"
    : new Date(
        value.length === 10 ? value + "T12:00:00Z" : value,
      ).toLocaleDateString("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
        timeZone: "UTC",
      });

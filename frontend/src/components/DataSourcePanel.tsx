import { useEffect, useRef, useState } from "react";
import {
  Database,
  ExternalLink,
  LoaderCircle,
  Radio,
  RefreshCw,
} from "lucide-react";
import { api, money, shortDate } from "../api";
import type { Market } from "../types";
import { Notice } from "./Common";

type SourceMetadata = {
  name?: string;
  url?: string;
  price_convention?: string;
  rows?: number;
  start?: string;
  end?: string;
};
type ProviderStatus = {
  external: {
    configured: boolean;
    connected: boolean;
    mode: string;
    message?: string;
  };
};
type ExternalQuote = {
  symbol: string;
  price: number;
  as_of: string;
  source: string;
  mode: string;
  delay_seconds: number | null;
  frequency: string;
  cached: boolean;
  received_at: string;
};

function readableError(error: unknown) {
  const message =
    error instanceof Error
      ? error.message
      : "The external provider is unavailable.";
  try {
    const parsed: unknown = JSON.parse(message);
    if (
      parsed &&
      typeof parsed === "object" &&
      "message" in parsed &&
      typeof parsed.message === "string"
    )
      return parsed.message;
  } catch {
    /* The API may return a plain-text error. */
  }
  return message;
}
function sourceLink(value?: string) {
  if (!value) return null;
  try {
    const url = new URL(value);
    return url.protocol === "https:" ? url.href : null;
  } catch {
    return null;
  }
}

export function DataSourcePanel({
  symbol,
  market,
}: {
  symbol: string;
  market: Market | null;
}) {
  const [status, setStatus] = useState<ProviderStatus | null>(null);
  const [statusError, setStatusError] = useState("");
  const [quote, setQuote] = useState<ExternalQuote | null>(null);
  const [quoteError, setQuoteError] = useState("");
  const [busy, setBusy] = useState(false);
  const requestVersion = useRef(0);
  const metadata = (
    market as (Market & { source_metadata?: SourceMetadata }) | null
  )?.source_metadata;
  const url = sourceLink(metadata?.url);

  useEffect(() => {
    let current = true;
    requestVersion.current += 1;
    setQuote(null);
    setQuoteError("");
    setBusy(false);
    setStatusError("");
    void api<ProviderStatus>("/provider/status")
      .then((result) => {
        if (current) setStatus(result);
      })
      .catch((error) => {
        if (current) setStatusError(readableError(error));
      });
    return () => {
      current = false;
      requestVersion.current += 1;
    };
  }, [symbol]);

  async function requestQuote() {
    const version = ++requestVersion.current;
    setBusy(true);
    setQuote(null);
    setQuoteError("");
    try {
      const result = await api<ExternalQuote>(
        `/provider/quote/${encodeURIComponent(symbol)}`,
      );
      if (version !== requestVersion.current) return;
      setQuote(result);
      setStatus({
        external: { configured: true, connected: true, mode: result.mode },
      });
    } catch (error) {
      if (version !== requestVersion.current) return;
      setQuoteError(readableError(error));
      setStatus((previous) =>
        previous
          ? {
              external: {
                ...previous.external,
                connected: false,
                mode: "Disconnected",
              },
            }
          : null,
      );
    } finally {
      if (version === requestVersion.current) setBusy(false);
    }
  }

  const providerLabel = quote
    ? "Delayed · end-of-day"
    : statusError
      ? "Status unavailable"
      : status?.external.connected
        ? "Last request connected"
        : status
          ? "Disconnected"
          : "Checking configuration…";

  return (
    <details className="panel data-source-panel">
      <summary>
        <Database size={16} aria-hidden="true" />
        <span>Data sources & optional external quotes</span>
        <span className="tag amber">Replay inputs</span>
      </summary>
      <div className="data-source-content">
        <section aria-label="Historical dataset provenance">
          <h3>Historical model inputs</h3>
          <p className="section-intro">
            {metadata?.name ||
              market?.source ||
              "Waiting for historical source details."}
          </p>
          <dl>
            <dt>Mode</dt>
            <dd>Historical replay · completed daily candles</dd>
            <dt>Observation range</dt>
            <dd>
              {metadata?.start && metadata.end
                ? `${shortDate(metadata.start)} – ${shortDate(metadata.end)}`
                : market
                  ? `Available through ${shortDate(market.as_of)}`
                  : "—"}
              {metadata?.rows ? ` · ${metadata.rows} sessions` : ""}
            </dd>
            <dt>Price convention</dt>
            <dd>
              {metadata?.price_convention ||
                "Source details are unavailable until market observations load."}
            </dd>
            <dt>Exchange timezone</dt>
            <dd>{market?.timezone || "America/New_York"}</dd>
          </dl>
          {url && (
            <a
              href={url}
              target="_blank"
              rel="noreferrer"
              className="text-button"
            >
              Open original source dataset{" "}
              <ExternalLink size={13} aria-hidden="true" />
            </a>
          )}
        </section>
        <section aria-label="Optional external market provider">
          <h3>
            <Radio size={15} aria-hidden="true" /> Optional external quote
          </h3>
          <p className="section-intro">
            Alpha Vantage end-of-day quotes are displayed separately from the
            historical chart and model inputs.
          </p>
          <p className="muted" role="status">
            {providerLabel}
            {status && !status.external.configured
              ? " · A server API key is required."
              : ""}
          </p>
          <button
            className="button secondary"
            onClick={() => void requestQuote()}
            disabled={busy || !symbol}
          >
            {busy ? (
              <LoaderCircle size={15} className="spin" aria-hidden="true" />
            ) : (
              <RefreshCw size={15} aria-hidden="true" />
            )}
            {busy
              ? "Checking external provider…"
              : `Request external quote for ${symbol}`}
          </button>
          {statusError && !quoteError && (
            <p className="negative" role="alert">
              Provider status could not be checked. Historical replay remains
              available.
            </p>
          )}
          {quoteError && (
            <p className="negative" role="alert">
              {quoteError}
            </p>
          )}
          {quote && (
            <div className="external-quote-result" aria-live="polite">
              <p>
                <strong>
                  {quote.symbol} {money(quote.price)}
                </strong>{" "}
                <span className="tag amber">Delayed · end-of-day</span>
              </p>
              <dl>
                <dt>Quote observation date</dt>
                <dd>{shortDate(quote.as_of)}</dd>
                <dt>Source</dt>
                <dd>{quote.source}</dd>
                <dt>Delay</dt>
                <dd>
                  {quote.delay_seconds === null
                    ? "Exact delay not supplied"
                    : `${quote.delay_seconds} seconds`}
                </dd>
                <dt>Retrieval</dt>
                <dd>
                  {quote.cached
                    ? "Cached provider response"
                    : "Provider response"}{" "}
                  · {new Date(quote.received_at).toLocaleString()}
                </dd>
              </dl>
            </div>
          )}
          <Notice>
            Historical replay stays selected. External quotes do not change the
            chart, forecast inputs, or recorded predictions.
          </Notice>
        </section>
      </div>
    </details>
  );
}

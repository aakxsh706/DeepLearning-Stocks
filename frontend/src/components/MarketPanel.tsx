import { useEffect, useRef, useState } from "react";
import {
  createChart,
  CandlestickSeries,
  HistogramSeries,
  ColorType,
  CrosshairMode,
  type IChartApi,
  type ISeriesApi,
  type Time,
} from "lightweight-charts";
import {
  ArrowUpRight,
  ArrowDownRight,
  CandlestickChart,
  Clock3,
  Star,
} from "lucide-react";
import type { Market, Instrument } from "../types";
import { money, shortDate } from "../api";
import { Empty, Spinner } from "./Common";
export function MarketPanel({
  market,
  instrument,
  selectedDate,
  loading,
  watched,
  onWatch,
}: {
  market: Market | null;
  instrument?: Instrument;
  selectedDate?: string;
  loading: boolean;
  watched: boolean;
  onWatch: () => void;
}) {
  const mount = useRef<HTMLDivElement>(null);
  const chart = useRef<IChartApi | null>(null);
  const series = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const [range, setRange] = useState(90);
  useEffect(() => {
    if (!mount.current || !market?.candles.length) return;
    const c = createChart(mount.current, {
      autoSize: true,
      height: 285,
      layout: {
        background: { type: ColorType.Solid, color: "#101724" },
        textColor: "#7c89a2",
        fontFamily: "Inter, ui-sans-serif, system-ui",
        fontSize: 11,
      },
      grid: {
        vertLines: { color: "#1a2333" },
        horzLines: { color: "#1a2333" },
      },
      rightPriceScale: {
        borderVisible: false,
        scaleMargins: { top: 0.15, bottom: 0.24 },
      },
      timeScale: {
        borderVisible: false,
        timeVisible: false,
        rightOffset: 4,
        barSpacing: 7,
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: "#a897ff", labelBackgroundColor: "#6e58bf" },
        horzLine: { color: "#62718a", labelBackgroundColor: "#354159" },
      },
      handleScroll: { vertTouchDrag: false },
      handleScale: { axisPressedMouseMove: true },
    });
    const candles = c.addSeries(CandlestickSeries, {
      upColor: "#50cdbb",
      downColor: "#dd7189",
      wickUpColor: "#50cdbb",
      wickDownColor: "#dd7189",
      borderVisible: false,
      lastValueVisible: true,
      priceLineVisible: true,
    });
    candles.setData(
      market.candles.map((d) => ({ ...d, time: d.time as Time })),
    );
    const volume = c.addSeries(HistogramSeries, {
      priceFormat: { type: "volume" },
      priceScaleId: "volume",
      lastValueVisible: false,
      priceLineVisible: false,
    });
    volume
      .priceScale()
      .applyOptions({ scaleMargins: { top: 0.83, bottom: 0 } });
    volume.setData(
      market.candles.map((d) => ({
        time: d.time as Time,
        value: d.volume,
        color: d.close >= d.open ? "#23473f" : "#482f3c",
      })),
    );
    chart.current = c;
    series.current = candles;
    c.timeScale().setVisibleLogicalRange({
      from: Math.max(0, market.candles.length - range),
      to: market.candles.length + 3,
    });
    return () => {
      c.remove();
      chart.current = null;
      series.current = null;
    };
  }, [market, range]);
  useEffect(() => {
    if (!market || !series.current || !chart.current) return;
    series.current.setData(
      market.candles.map((d) => ({
        ...d,
        time: d.time as Time,
        ...(d.time === selectedDate
          ? { color: "#b39aff", wickColor: "#b39aff", borderColor: "#b39aff" }
          : {}),
      })),
    );
    if (selectedDate) {
      const candle = market.candles.find((d) => d.time === selectedDate);
      if (candle) {
        const index = market.candles.indexOf(candle);
        const visible = chart.current.timeScale().getVisibleLogicalRange();
        if (visible && (index < visible.from || index > visible.to))
          chart.current
            .timeScale()
            .setVisibleLogicalRange({
              from: Math.max(0, index - Math.floor(range * 0.7)),
              to: index + Math.ceil(range * 0.3),
            });
        chart.current.setCrosshairPosition(
          candle.close,
          selectedDate as Time,
          series.current,
        );
      }
    } else chart.current.clearCrosshairPosition();
  }, [selectedDate, market, range]);
  const last = market?.candles.at(-1),
    prev = market?.candles.at(-2);
  const change = last && prev ? (last.close - prev.close) / prev.close : 0;
  return (
    <section className="panel market-panel">
      <div className="market-top">
        <div className="stock-identity">
          <div className="stock-logo">{market?.symbol?.slice(0, 1) || "N"}</div>
          <div>
            <div className="stock-title">
              <h2>{instrument?.name || market?.symbol || "Market overview"}</h2>
              <button
                className={`icon-button watch-star ${watched ? "active" : ""}`}
                aria-label={
                  watched ? "Remove from watchlist" : "Add to watchlist"
                }
                onClick={onWatch}
              >
                <Star size={15} fill={watched ? "currentColor" : "none"} />
              </button>
            </div>
            <p>
              {market?.symbol || "—"} <span>·</span>{" "}
              {instrument?.exchange || "US equities"} <span>·</span> USD
            </p>
          </div>
        </div>
        <span className="tag amber">
          <span className="status-dot" />
          Historical replay
        </span>
      </div>
      <div className="market-price-row">
        <div className="market-price">
          {money(last?.close)}
          <span className={change >= 0 ? "positive" : "negative"}>
            {change >= 0 ? (
              <ArrowUpRight size={16} />
            ) : (
              <ArrowDownRight size={16} />
            )}{" "}
            {change >= 0 ? "+" : ""}
            {(change * 100).toFixed(2)}% <small>last session</small>
          </span>
        </div>
        <div className="segmented compact" aria-label="Chart range">
          {[
            [30, "1M"],
            [90, "3M"],
            [180, "6M"],
            [260, "1Y"],
          ].map(([v, label]) => (
            <button
              key={v}
              className={range === v ? "selected" : ""}
              onClick={() => setRange(Number(v))}
            >
              {label}
            </button>
          ))}
        </div>
      </div>
      <div
        className="candle-chart"
        ref={mount}
        aria-label="Historical daily candlestick chart"
      >
        {loading ? (
          <Spinner />
        ) : !market?.candles.length ? (
          <Empty title="Waiting for market observations">
            Historical data will appear when the API is available.
          </Empty>
        ) : null}
      </div>
      <div className="chart-footer">
        <span>
          <CandlestickChart size={13} /> 1D candles{" "}
          <span className="separator">/</span> Vol{" "}
          {last
            ? new Intl.NumberFormat("en-US", { notation: "compact" }).format(
                last.volume,
              )
            : "—"}
        </span>
        <span>
          <span className="legend-dot violet" />
          {selectedDate
            ? `Inspecting ${shortDate(selectedDate)}`
            : "Select a replay timestep to inspect"}
        </span>
      </div>
      <div className="market-source">
        <span>
          <Clock3 size={12} />
          {shortDate(market?.as_of)} · {market?.timezone || "America/New_York"}
        </span>
        <span title={market?.source}>
          {market?.source || "Waiting for source"} ·{" "}
          {market?.session_status || "Historical replay"}
          {market?.delay_seconds != null
            ? ` · ${market.delay_seconds}s delay`
            : ""}
        </span>
      </div>
    </section>
  );
}

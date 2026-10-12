import { LoaderCircle, ArrowUpRight, Inbox, Info, X } from "lucide-react";
import { useEffect, useRef, type ReactNode } from "react";
export function Empty({
  title,
  children,
  action,
}: {
  title: string;
  children?: ReactNode;
  action?: ReactNode;
}) {
  return (
    <div className="empty">
      <Inbox size={27} />
      <h3>{title}</h3>
      <p>{children}</p>
      {action}
    </div>
  );
}
export function Spinner({
  label = "Loading observations…",
}: {
  label?: string;
}) {
  return (
    <div className="loading">
      <LoaderCircle className="spin" size={19} />
      {label}
    </div>
  );
}
export function PanelHeading({
  eyebrow,
  title,
  action,
}: {
  eyebrow?: string;
  title: string;
  action?: ReactNode;
}) {
  return (
    <div className="panel-heading">
      <div>
        {eyebrow && <div className="eyebrow">{eyebrow}</div>}
        <h2>{title}</h2>
      </div>
      {action}
    </div>
  );
}
export function Metric({
  label,
  value,
  detail,
  color = "cyan",
}: {
  label: string;
  value: ReactNode;
  detail?: ReactNode;
  color?: string;
}) {
  return (
    <div className={`metric ${color}`}>
      <div className="metric-label">
        {label}
        <ArrowUpRight size={14} />
      </div>
      <div className="metric-value">{value}</div>
      {detail && <div className="metric-detail">{detail}</div>}
    </div>
  );
}
export function Notice({ children }: { children: ReactNode }) {
  return (
    <div className="notice">
      <Info size={15} />
      <span>{children}</span>
    </div>
  );
}
export function Modal({
  title,
  onClose,
  children,
}: {
  title: string;
  onClose: () => void;
  children: ReactNode;
}) {
  const dialog = useRef<HTMLElement>(null);
  const returnFocus = useRef(document.activeElement as HTMLElement | null);
  useEffect(() => {
    dialog.current?.querySelector<HTMLButtonElement>("button")?.focus();
    return () => returnFocus.current?.focus();
  }, []);
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <section
        ref={dialog}
        className="modal"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        onClick={(e) => e.stopPropagation()}
        onKeyDown={(e) => {
          if (e.key === "Escape") {
            e.preventDefault();
            onClose();
          }
          if (e.key === "Tab") {
            const controls = Array.from(
              dialog.current?.querySelectorAll<HTMLElement>(
                'button:not(:disabled), a[href], input:not(:disabled), select:not(:disabled), [tabindex="0"]',
              ) ?? [],
            );
            const first = controls[0];
            const last = controls.at(-1);
            if (e.shiftKey && document.activeElement === first) {
              e.preventDefault();
              last?.focus();
            } else if (!e.shiftKey && document.activeElement === last) {
              e.preventDefault();
              first?.focus();
            }
          }
        }}
      >
        <header>
          <h2>{title}</h2>
          <button
            className="icon-button"
            onClick={onClose}
            aria-label="Close dialog"
            autoFocus
          >
            <X size={20} />
          </button>
        </header>
        {children}
      </section>
    </div>
  );
}
export const chartTooltip = {
  contentStyle: {
    background: "#151d2d",
    border: "1px solid #2b354b",
    borderRadius: 8,
    color: "#e9eef9",
    fontSize: 12,
  },
  labelStyle: { color: "#a5b1c7" },
};

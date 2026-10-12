import { useEffect, useRef, useState } from "react";
type EventMessage = {
  id: string;
  type: string;
  timestamp: string;
  payload: unknown;
};
export function useEvents(onEvent: (event: EventMessage) => void) {
  const [connected, setConnected] = useState(false);
  const handler = useRef(onEvent);
  handler.current = onEvent;
  useEffect(() => {
    let disposed = false;
    let socket: WebSocket | undefined;
    let timer: ReturnType<typeof setTimeout>;
    let attempt = 0;
    let last = "";
    const seen = new Set<string>();
    function connect() {
      if (disposed) return;
      socket = new WebSocket(
        `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws${last ? "?after=" + encodeURIComponent(last) : ""}`,
      );
      socket.onopen = () => {
        setConnected(true);
        attempt = 0;
      };
      socket.onmessage = (e) => {
        try {
          const event = JSON.parse(e.data) as EventMessage;
          if (event.type === "connected") {
            const payload = event.payload as Record<string, unknown>;
            if (payload?.cursor_reset) {
              last = "";
              seen.clear();
            }
            handler.current(event);
            return;
          }
          if (seen.has(event.id)) return;
          seen.add(event.id);
          if (/^\d+$/.test(event.id)) last = event.id;
          if (seen.size > 1000) seen.delete(seen.values().next().value!);
          handler.current(event);
        } catch {
          /* Ignore malformed transport frames. */
        }
      };
      socket.onclose = () => {
        setConnected(false);
        if (!disposed)
          timer = setTimeout(connect, Math.min(15000, 1000 * 2 ** attempt++));
      };
      socket.onerror = () => socket?.close();
    }
    connect();
    return () => {
      disposed = true;
      clearTimeout(timer);
      socket?.close();
    };
  }, []);
  return connected;
}

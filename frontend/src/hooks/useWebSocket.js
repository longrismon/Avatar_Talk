import { useEffect, useRef, useState, useCallback } from "react";

const WS_URL = `ws://${window.location.hostname}:9600/ws`;

export function useWebSocket(token) {
  const wsRef = useRef(null);
  const [readyState, setReadyState] = useState(WebSocket.CONNECTING);
  const [lastMessage, setLastMessage] = useState(null);

  useEffect(() => {
    const url = token ? `${WS_URL}?token=${encodeURIComponent(token)}` : WS_URL;
    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => setReadyState(WebSocket.OPEN);
    ws.onclose = () => setReadyState(WebSocket.CLOSED);
    ws.onerror = () => setReadyState(WebSocket.CLOSED);
    ws.onmessage = (e) => {
      try {
        setLastMessage(JSON.parse(e.data));
      } catch {
        // ignore non-JSON frames
      }
    };

    const ping = setInterval(() => {
      if (ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify({ type: "ping" }));
      }
    }, 20000);

    return () => {
      clearInterval(ping);
      ws.close();
    };
  }, [token]);

  const sendMessage = useCallback((msg) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  }, []);

  return { sendMessage, lastMessage, readyState };
}

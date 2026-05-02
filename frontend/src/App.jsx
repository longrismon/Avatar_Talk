import React, { useEffect, useReducer, useRef } from "react";
import { useWebSocket } from "./hooks/useWebSocket.js";
import ReviewPanel from "./components/ReviewPanel.jsx";
import StatusBar from "./components/StatusBar.jsx";

const TOKEN = new URLSearchParams(window.location.search).get("token") || "change-me-in-production";

const initialState = {
  agentState: "idle",
  options: [],
  timeoutSeconds: 5,
  callStartTime: null,
  lastSpoken: null,
  errorMessage: null,
};

function reducer(state, action) {
  switch (action.type) {
    case "state_changed": {
      const s = action.to.toLowerCase();
      return {
        ...state,
        agentState: s,
        callStartTime: s === "listening" && !state.callStartTime ? Date.now() : state.callStartTime,
        errorMessage: null,
      };
    }
    case "options":
      return { ...state, agentState: "generating", options: action.options };
    case "review_started":
      return {
        ...state,
        agentState: "reviewing",
        options: action.options,
        timeoutSeconds: action.timeout_seconds ?? state.timeoutSeconds,
      };
    case "response_selected":
      return { ...state, lastSpoken: action.text };
    case "call_ended":
      return { ...initialState };
    case "error":
      return { ...state, agentState: "error", errorMessage: action.message };
    default:
      return state;
  }
}

export default function App() {
  const { sendMessage, lastMessage, readyState } = useWebSocket(TOKEN);
  const [state, dispatch] = useReducer(reducer, initialState);

  useEffect(() => {
    if (!lastMessage) return;
    dispatch(lastMessage);
  }, [lastMessage]);

  const handleSelect = (id) => {
    sendMessage({ type: "selection", option_id: id });
  };

  const handleTakeover = () => {
    sendMessage({ type: "takeover" });
  };

  const wsStatus = readyState === WebSocket.OPEN ? "connected" : "disconnected";

  return (
    <div style={styles.root}>
      <StatusBar agentState={state.agentState} callStartTime={state.callStartTime} />

      <div style={styles.body}>
        {state.agentState === "reviewing" && state.options.length > 0 ? (
          <ReviewPanel
            options={state.options}
            timeoutSeconds={state.timeoutSeconds}
            onSelect={handleSelect}
            onTakeover={handleTakeover}
          />
        ) : (
          <div style={styles.idle}>
            <div style={styles.stateLabel}>{state.agentState.replace("_", " ").toUpperCase()}</div>
            {state.lastSpoken && (
              <div style={styles.spoken}>
                <span style={styles.spokenLabel}>Last spoken</span>
                <p style={styles.spokenText}>{state.lastSpoken}</p>
              </div>
            )}
            {state.errorMessage && (
              <div style={styles.errorBox}>{state.errorMessage}</div>
            )}
          </div>
        )}
      </div>

      <div style={styles.footer}>
        <span style={{ ...styles.wsDot, background: wsStatus === "connected" ? "var(--green)" : "var(--red)" }} />
        <span style={styles.wsLabel}>WebSocket {wsStatus}</span>
      </div>
    </div>
  );
}

const styles = {
  root: {
    display: "flex",
    flexDirection: "column",
    minHeight: "100vh",
  },
  body: {
    flex: 1,
    overflow: "auto",
  },
  idle: {
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    minHeight: 320,
    gap: 20,
    padding: 32,
  },
  stateLabel: {
    fontSize: 24,
    fontWeight: 700,
    color: "var(--text-muted)",
    letterSpacing: 2,
  },
  spoken: {
    background: "var(--surface)",
    border: "1px solid var(--border)",
    borderRadius: "var(--radius)",
    padding: "16px 20px",
    maxWidth: 560,
    width: "100%",
  },
  spokenLabel: {
    fontSize: 11,
    color: "var(--text-muted)",
    textTransform: "uppercase",
    letterSpacing: 1,
    display: "block",
    marginBottom: 6,
  },
  spokenText: {
    fontSize: 15,
    lineHeight: 1.6,
  },
  errorBox: {
    background: "#2d1a1a",
    border: "1px solid var(--red)",
    borderRadius: "var(--radius)",
    padding: "12px 18px",
    color: "var(--red)",
    maxWidth: 560,
    width: "100%",
    fontSize: 14,
  },
  footer: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "8px 20px",
    borderTop: "1px solid var(--border)",
  },
  wsDot: {
    width: 8,
    height: 8,
    borderRadius: "50%",
  },
  wsLabel: {
    fontSize: 12,
    color: "var(--text-muted)",
  },
};

import React, { useEffect, useReducer } from "react";
import { useWebSocket } from "./hooks/useWebSocket.js";
import ReviewPanel from "./components/ReviewPanel.jsx";
import StatusBar from "./components/StatusBar.jsx";
import TranscriptPanel from "./components/TranscriptPanel.jsx";

const TOKEN = new URLSearchParams(window.location.search).get("token") || "change-me-in-production";

const initialState = {
  agentState: "idle",
  options: [],
  timeoutSeconds: 5,
  callStartTime: null,
  lastSpoken: null,
  errorMessage: null,
  transcript: [],
  partialText: "",
  nextId: 0,
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
    case "partial_transcript":
      return { ...state, partialText: action.text };
    case "utterance_complete":
      return {
        ...state,
        partialText: "",
        transcript: [
          ...state.transcript,
          {
            id: state.nextId,
            speaker: "other",
            text: action.text,
            turn: action.turn,
            interrupted: false,
          },
        ],
        nextId: state.nextId + 1,
      };
    case "speaking":
      return {
        ...state,
        transcript: [
          ...state.transcript,
          {
            id: state.nextId,
            speaker: "agent",
            text: action.text,
            turn: state.transcript.length > 0
              ? state.transcript[state.transcript.length - 1].turn
              : 0,
            interrupted: false,
          },
        ],
        nextId: state.nextId + 1,
      };
    case "barge_in": {
      const last = state.transcript[state.transcript.length - 1];
      if (!last || last.speaker !== "agent") return state;
      return {
        ...state,
        transcript: [
          ...state.transcript.slice(0, -1),
          { ...last, interrupted: true },
        ],
      };
    }
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
    <div className="app-layout">
      <StatusBar agentState={state.agentState} callStartTime={state.callStartTime} />

      <TranscriptPanel transcript={state.transcript} partialText={state.partialText} />

      {state.agentState === "reviewing" && state.options.length > 0 && (
        <ReviewPanel
          options={state.options}
          timeoutSeconds={state.timeoutSeconds}
          onSelect={handleSelect}
          onTakeover={handleTakeover}
        />
      )}

      {state.agentState !== "reviewing" && (state.lastSpoken || state.errorMessage) && (
        <div style={styles.statusArea}>
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

      <div style={styles.footer}>
        <span style={{ ...styles.wsDot, background: wsStatus === "connected" ? "var(--green)" : "var(--red)" }} />
        <span style={styles.wsLabel}>WebSocket {wsStatus}</span>
      </div>
    </div>
  );
}

const styles = {
  statusArea: {
    padding: "0 20px 16px",
    display: "flex",
    flexDirection: "column",
    gap: 10,
    alignItems: "center",
  },
  spoken: {
    background: "var(--surface)",
    border: "1px solid var(--border)",
    borderRadius: "var(--radius)",
    padding: "12px 16px",
    maxWidth: 560,
    width: "100%",
  },
  spokenLabel: {
    fontSize: 11,
    color: "var(--text-muted)",
    textTransform: "uppercase",
    letterSpacing: 1,
    display: "block",
    marginBottom: 4,
  },
  spokenText: {
    fontSize: 14,
    lineHeight: 1.5,
  },
  errorBox: {
    background: "#2d1a1a",
    border: "1px solid var(--red)",
    borderRadius: "var(--radius)",
    padding: "10px 16px",
    color: "var(--red)",
    maxWidth: 560,
    width: "100%",
    fontSize: 13,
  },
  footer: {
    display: "flex",
    alignItems: "center",
    gap: 8,
    padding: "8px 20px",
    borderTop: "1px solid var(--border)",
    flexShrink: 0,
  },
  wsDot: {
    width: 8,
    height: 8,
    borderRadius: "50%",
    flexShrink: 0,
  },
  wsLabel: {
    fontSize: 12,
    color: "var(--text-muted)",
  },
};

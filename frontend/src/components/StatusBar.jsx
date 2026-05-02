import React, { useEffect, useState } from "react";

const STATE_LABELS = {
  idle: "Idle — ready for instructions",
  planning: "Planning...",
  browser_action: "Automating browser...",
  awaiting_call: "Waiting for call to connect...",
  listening: "Listening...",
  generating: "Generating responses...",
  reviewing: "Review required",
  speaking: "Speaking...",
  manual_override: "Manual override active",
  call_ended: "Call ended",
  error: "Error",
};

const STATE_COLORS = {
  listening: "var(--green)",
  generating: "var(--yellow)",
  speaking: "var(--accent)",
  error: "var(--red)",
};

export default function StatusBar({ agentState, callStartTime }) {
  const [elapsed, setElapsed] = useState(0);

  useEffect(() => {
    if (!callStartTime) return;
    const id = setInterval(() => {
      setElapsed(Math.floor((Date.now() - callStartTime) / 1000));
    }, 1000);
    return () => clearInterval(id);
  }, [callStartTime]);

  const fmt = (s) => {
    const m = Math.floor(s / 60).toString().padStart(2, "0");
    const sec = (s % 60).toString().padStart(2, "0");
    return `${m}:${sec}`;
  };

  const color = STATE_COLORS[agentState] || "var(--text-muted)";
  const label = STATE_LABELS[agentState] || agentState;

  return (
    <div style={styles.bar}>
      <span style={{ ...styles.dot, background: color }} />
      <span style={styles.label}>{label}</span>
      {callStartTime && (
        <span style={styles.timer}>{fmt(elapsed)}</span>
      )}
    </div>
  );
}

const styles = {
  bar: {
    display: "flex",
    alignItems: "center",
    gap: 10,
    padding: "12px 20px",
    background: "var(--surface)",
    borderBottom: "1px solid var(--border)",
  },
  dot: {
    width: 10,
    height: 10,
    borderRadius: "50%",
    flexShrink: 0,
    transition: "background 0.3s",
  },
  label: {
    color: "var(--text-muted)",
    fontSize: 13,
    flex: 1,
  },
  timer: {
    fontVariantNumeric: "tabular-nums",
    color: "var(--text-muted)",
    fontSize: 13,
  },
};

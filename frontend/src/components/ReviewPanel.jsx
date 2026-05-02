import React, { useEffect, useRef, useState } from "react";

const TONE_COLORS = {
  professional: "#5b6ef5",
  empathetic: "#3ecf8e",
  direct: "#f5c542",
  light: "#c084fc",
};

export default function ReviewPanel({ options, timeoutSeconds, onSelect, onTakeover }) {
  const [timeLeft, setTimeLeft] = useState(timeoutSeconds * 1000);
  const [active, setActive] = useState(true);
  const intervalRef = useRef(null);

  useEffect(() => {
    intervalRef.current = setInterval(() => {
      setTimeLeft((t) => {
        if (t <= 100) {
          clearInterval(intervalRef.current);
          setActive(false);
          const rec = options.find((o) => o.recommended) || options[0];
          if (rec) onSelect(rec.id);
          return 0;
        }
        return t - 100;
      });
    }, 100);
    return () => clearInterval(intervalRef.current);
  }, [options, onSelect]);

  useEffect(() => {
    const handler = (e) => {
      if (!active) return;
      const key = e.key;
      if (key >= "1" && key <= "4") {
        const idx = parseInt(key, 10) - 1;
        if (options[idx]) {
          setActive(false);
          clearInterval(intervalRef.current);
          onSelect(options[idx].id);
        }
      } else if (key === "t" || key === "Escape") {
        setActive(false);
        clearInterval(intervalRef.current);
        onTakeover();
      }
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [active, options, onSelect, onTakeover]);

  const barWidth = `${(timeLeft / (timeoutSeconds * 1000)) * 100}%`;

  return (
    <div style={styles.container}>
      <div style={styles.header}>
        <span style={styles.title}>Choose a response</span>
        <span style={styles.hint}>Keys: 1–4 to select · T / Esc to take over</span>
      </div>

      {/* Countdown bar */}
      <div style={styles.barTrack}>
        <div
          style={{
            ...styles.barFill,
            width: barWidth,
            background: timeLeft < timeoutSeconds * 300 ? "var(--red)" : "var(--accent)",
          }}
        />
      </div>

      <div style={styles.grid}>
        {options.map((opt, i) => (
          <button
            key={opt.id}
            disabled={!active}
            onClick={() => {
              if (!active) return;
              setActive(false);
              clearInterval(intervalRef.current);
              onSelect(opt.id);
            }}
            style={{
              ...styles.option,
              borderColor: opt.recommended ? TONE_COLORS[opt.tone] : "var(--border)",
              opacity: active ? 1 : 0.5,
            }}
          >
            <div style={styles.optionTop}>
              <span style={styles.keyHint}>{i + 1}</span>
              <span
                style={{
                  ...styles.tone,
                  color: TONE_COLORS[opt.tone] || "var(--text-muted)",
                }}
              >
                {opt.tone}
                {opt.recommended && " ★"}
              </span>
            </div>
            <p style={styles.text}>{opt.text}</p>
          </button>
        ))}
      </div>

      <button style={styles.takeoverBtn} onClick={() => { setActive(false); clearInterval(intervalRef.current); onTakeover(); }}>
        Take over manually (T)
      </button>
    </div>
  );
}

const styles = {
  container: {
    padding: "20px 24px",
    display: "flex",
    flexDirection: "column",
    gap: 16,
  },
  header: {
    display: "flex",
    justifyContent: "space-between",
    alignItems: "center",
  },
  title: {
    fontWeight: 600,
    fontSize: 16,
  },
  hint: {
    color: "var(--text-muted)",
    fontSize: 12,
  },
  barTrack: {
    height: 4,
    background: "var(--border)",
    borderRadius: 2,
    overflow: "hidden",
  },
  barFill: {
    height: "100%",
    borderRadius: 2,
    transition: "width 0.1s linear, background 0.3s",
  },
  grid: {
    display: "grid",
    gridTemplateColumns: "1fr 1fr",
    gap: 12,
  },
  option: {
    background: "var(--surface)",
    border: "1.5px solid var(--border)",
    borderRadius: "var(--radius)",
    padding: "14px 16px",
    textAlign: "left",
    color: "var(--text)",
    transition: "border-color 0.2s, background 0.2s",
    cursor: "pointer",
  },
  optionTop: {
    display: "flex",
    justifyContent: "space-between",
    marginBottom: 6,
  },
  keyHint: {
    background: "var(--surface-2)",
    borderRadius: 4,
    padding: "1px 7px",
    fontSize: 12,
    color: "var(--text-muted)",
    fontWeight: 600,
  },
  tone: {
    fontSize: 12,
    fontWeight: 500,
  },
  text: {
    fontSize: 14,
    lineHeight: 1.55,
    color: "var(--text)",
  },
  takeoverBtn: {
    alignSelf: "flex-end",
    background: "transparent",
    border: "1px solid var(--border)",
    color: "var(--text-muted)",
    borderRadius: "var(--radius)",
    padding: "8px 16px",
    fontSize: 13,
    transition: "border-color 0.2s, color 0.2s",
  },
};

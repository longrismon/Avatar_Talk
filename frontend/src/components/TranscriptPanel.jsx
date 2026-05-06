import { useEffect, useRef } from "react";

export default function TranscriptPanel({ transcript, partialText }) {
  const bottomRef = useRef(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [transcript, partialText]);

  if (transcript.length === 0 && !partialText) {
    return (
      <div className="transcript-panel transcript-empty">
        Waiting for conversation…
      </div>
    );
  }

  return (
    <div className="transcript-panel">
      {transcript.map((entry) => (
        <div
          key={entry.id}
          className={`transcript-entry transcript-${entry.speaker}${
            entry.interrupted ? " transcript-interrupted" : ""
          }`}
        >
          <span className="transcript-label">
            {entry.speaker === "other" ? "Them" : "Agent"}
            <span className="transcript-turn">#{entry.turn}</span>
          </span>
          <span className="transcript-text">
            {entry.text}
            {entry.interrupted && (
              <span className="transcript-barge-tag"> [interrupted]</span>
            )}
          </span>
        </div>
      ))}

      {partialText && (
        <div className="transcript-entry transcript-other transcript-partial">
          <span className="transcript-label">Them</span>
          <span className="transcript-text">
            {partialText}
            <span className="transcript-cursor" />
          </span>
        </div>
      )}

      <div ref={bottomRef} />
    </div>
  );
}

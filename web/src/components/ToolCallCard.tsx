import { useEffect, useState } from 'react';
import { ToolCall } from '../types';
import './ToolCallCard.css';

interface ToolCallCardProps {
  toolCall: ToolCall;
}

/** One-line summary shown in the collapsed header so results are visible at a glance */
function CollapsedSummary({ toolCall }: { toolCall: ToolCall }) {
  const r = toolCall.result;
  if (!r) return null;

  if (toolCall.tool === 'analyze_image') {
    const name = r.full_name ?? r.diagnosis;
    const pct = r.confidence != null ? `${Math.round(r.confidence * 100)}%` : null;
    if (name) return (
      <span className="tool-header-summary">
        {name}{pct ? ` — ${pct}` : ''}
      </span>
    );
  }

  if (toolCall.tool === 'compare_images') {
    const key = r.status_label ?? 'STABLE';
    const cfg = STATUS_CONFIG[key] ?? { emoji: '⚪', label: key };
    return (
      <span className="tool-header-summary">
        {cfg.emoji} {cfg.label}
      </span>
    );
  }

  return null;
}

export function ToolCallCard({ toolCall }: ToolCallCardProps) {
  // Auto-expand when the tool completes so results are immediately visible.
  // User can collapse manually afterwards.
  const [expanded, setExpanded] = useState(false);

  useEffect(() => {
    if (toolCall.status === 'complete') setExpanded(true);
  }, [toolCall.status]);

  const isLoading = toolCall.status === 'calling';
  const isError = toolCall.status === 'error';

  const icon = toolCall.tool === 'compare_images' ? '🔄' : '🔬';
  const label = toolCall.tool.replace(/_/g, ' ');

  return (
    <div className={`tool-card ${isLoading ? 'loading' : ''} ${isError ? 'error' : ''}`}>
      <button
        className="tool-card-header"
        onClick={() => !isLoading && setExpanded(e => !e)}
        disabled={isLoading}
      >
        <span className="tool-icon">{icon}</span>
        <span className="tool-label">{label}</span>
        {isLoading ? (
          <span className="tool-status calling">
            <span className="spinner" /> running…
          </span>
        ) : isError ? (
          <span className="tool-status error-text">error</span>
        ) : (
          <>
            <span className="tool-status done">✓</span>
            {!expanded && <CollapsedSummary toolCall={toolCall} />}
          </>
        )}
        {!isLoading && (
          <span className="tool-chevron">{expanded ? '▲' : '▼'}</span>
        )}
      </button>

      {expanded && !isLoading && toolCall.result && (
        <div className="tool-card-body">
          {toolCall.tool === 'analyze_image' && (
            <AnalyzeImageResult result={toolCall.result} />
          )}
          {toolCall.tool === 'compare_images' && (
            <CompareImagesResult result={toolCall.result} />
          )}
          {toolCall.tool !== 'analyze_image' && toolCall.tool !== 'compare_images' && (
            <GenericResult result={toolCall.result} />
          )}
        </div>
      )}
    </div>
  );
}

/* ─── analyze_image renderer ─────────────────────────────────────────────── */

function AnalyzeImageResult({ result }: { result: ToolCall['result'] }) {
  if (!result) return null;

  const hasClassifier = result.diagnosis != null;
  const topPrediction = result.all_predictions?.[0];
  const otherPredictions = result.all_predictions?.slice(1) ?? [];
  const confidence = result.confidence ?? topPrediction?.probability ?? 0;
  const pct = Math.round(confidence * 100);
  const statusColor = pct >= 70 ? '#ef4444' : pct >= 40 ? '#f59e0b' : '#22c55e';

  return (
    <div className="analyze-result">
      <div className="analyze-top">
        {result.image_url && (
          <img
            src={result.image_url}
            alt="Analyzed lesion"
            className="analyze-thumb"
          />
        )}
        <div className="analyze-info">
          {hasClassifier ? (
            <>
              <p className="diagnosis-name">{result.full_name ?? result.diagnosis}</p>
              <p className="confidence-label" style={{ color: statusColor }}>
                Confidence: {pct}%
              </p>
              <div className="confidence-bar-track">
                <div
                  className="confidence-bar-fill"
                  style={{ width: `${pct}%`, background: statusColor }}
                />
              </div>
            </>
          ) : (
            <p className="diagnosis-name" style={{ color: 'var(--gray-500)', fontWeight: 400, fontSize: '0.875rem' }}>
              Visual assessment complete — classifier unavailable
            </p>
          )}
        </div>
      </div>

      {hasClassifier && otherPredictions.length > 0 && (
        <ul className="other-predictions">
          {otherPredictions.map(p => (
            <li key={p.class} className="prediction-row">
              <span className="pred-name">{p.full_name ?? p.class}</span>
              <span className="pred-pct">{Math.round(p.probability * 100)}%</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/* ─── compare_images renderer ────────────────────────────────────────────── */

const STATUS_CONFIG: Record<string, { label: string; color: string; emoji: string }> = {
  STABLE: { label: 'Stable', color: '#22c55e', emoji: '🟢' },
  MINOR_CHANGE: { label: 'Minor Change', color: '#f59e0b', emoji: '🟡' },
  SIGNIFICANT_CHANGE: { label: 'Significant Change', color: '#ef4444', emoji: '🔴' },
  IMPROVED: { label: 'Improved', color: '#3b82f6', emoji: '🔵' },
};

function CompareImagesResult({ result }: { result: ToolCall['result'] }) {
  if (!result) return null;

  const statusKey = result.status_label ?? 'STABLE';
  const status = STATUS_CONFIG[statusKey] ?? { label: statusKey, color: '#6b7280', emoji: '⚪' };
  const featureChanges = Object.entries(result.feature_changes ?? {});

  return (
    <div className="compare-result">
      <div className="compare-status" style={{ color: status.color }}>
        <strong>Status: {status.label} {status.emoji}</strong>
      </div>

      {featureChanges.length > 0 && (
        <ul className="feature-changes">
          {featureChanges.map(([name, vals]) => {
            const delta = vals.curr - vals.prev;
            const sign = delta > 0 ? '+' : '';
            return (
              <li key={name} className="feature-row">
                <span className="feature-name">{name}</span>
                <span className="feature-delta" style={{ color: Math.abs(delta) > 0.1 ? '#f59e0b' : '#6b7280' }}>
                  {sign}{(delta * 100).toFixed(1)}%
                </span>
              </li>
            );
          })}
        </ul>
      )}

      {result.summary && (
        <p className="compare-summary">{result.summary}</p>
      )}
    </div>
  );
}

/* ─── Generic (unknown tool) renderer ───────────────────────────────────── */

function GenericResult({ result }: { result: ToolCall['result'] }) {
  return (
    <pre className="generic-result">
      {JSON.stringify(result, null, 2)}
    </pre>
  );
}

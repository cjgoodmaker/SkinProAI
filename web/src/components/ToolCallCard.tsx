import { ToolCall } from '../types';
import './ToolCallCard.css';

interface ToolCallCardProps {
  toolCall: ToolCall;
}

const TOOL_LABELS: Record<string, string> = {
  analyze_image: 'Analyse image',
  compare_images: 'Compare images',
  load_model: 'Loading analysis model',
  visual_exam: 'MedGemma visual examination',
  classify: 'Running classifier',
  gradcam: 'Generating attention map',
  guidelines: 'Searching clinical guidelines',
  search_guidelines: 'Searching clinical guidelines',
};

const STATUS_CONFIG: Record<string, { label: string; dot: string }> = {
  STABLE: { label: 'Stable', dot: 'dot-green' },
  MINOR_CHANGE: { label: 'Minor Change', dot: 'dot-amber' },
  SIGNIFICANT_CHANGE: { label: 'Significant Change', dot: 'dot-red' },
  IMPROVED: { label: 'Improved', dot: 'dot-blue' },
};

export function ToolCallCard({ toolCall }: ToolCallCardProps) {
  const isLoading = toolCall.status === 'calling';
  const isError = toolCall.status === 'error';
  const label = TOOL_LABELS[toolCall.tool] ?? toolCall.tool.replace(/_/g, ' ');

  // Build summary text for completed tools
  let summary = '';
  if (toolCall.status === 'complete' && toolCall.result) {
    const r = toolCall.result;
    if (toolCall.tool === 'analyze_image' && (r.full_name || r.diagnosis)) {
      const pct = r.confidence != null ? ` (${Math.round(r.confidence * 100)}%)` : '';
      summary = `${r.full_name ?? r.diagnosis}${pct}`;
    } else if (toolCall.tool === 'compare_images' && r.status_label) {
      const cfg = STATUS_CONFIG[r.status_label];
      summary = cfg?.label ?? r.status_label;
    }
  }

  return (
    <div className="tool-line">
      <span className="tool-line-indicator">
        {isLoading ? (
          <span className="tool-spinner" />
        ) : isError ? (
          <span className="tool-dot dot-red" />
        ) : (
          <span className="tool-dot dot-green" />
        )}
      </span>
      <span className="tool-line-label">{label}</span>
      {isLoading && <span className="tool-line-status">running</span>}
      {summary && <span className="tool-line-summary">{summary}</span>}
      {isError && <span className="tool-line-error">failed</span>}
    </div>
  );
}

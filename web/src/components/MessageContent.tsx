import ReactMarkdown from 'react-markdown';
import './MessageContent.css';

// Serve any temp visualization image (GradCAM, comparison) through the API
const TEMP_IMG_URL = (path: string) =>
  `/api/patients/gradcam?path=${encodeURIComponent(path)}`;

// ─── Types ─────────────────────────────────────────────────────────────────

type Segment =
  | { type: 'stage'; label: string }
  | { type: 'thinking'; content: string }
  | { type: 'response'; content: string }
  | { type: 'tool_output'; label: string; content: string }
  | { type: 'gradcam'; path: string }
  | { type: 'comparison'; path: string }
  | { type: 'gradcam_compare'; path1: string; path2: string }
  | { type: 'result'; content: string }
  | { type: 'error'; content: string }
  | { type: 'complete'; content: string }
  | { type: 'references'; content: string }
  | { type: 'observation'; content: string }
  | { type: 'text'; content: string };

// ─── Parser ────────────────────────────────────────────────────────────────

// Splits raw text by all known complete tag patterns (capturing group preserves them)
const TAG_SPLIT_RE = new RegExp(
  '(' +
    [
      '\\[STAGE:[^\\]]*\\][\\s\\S]*?\\[\\/STAGE\\]',
      '\\[THINKING\\][\\s\\S]*?\\[\\/THINKING\\]',
      '\\[RESPONSE\\][\\s\\S]*?\\[\\/RESPONSE\\]',
      '\\[TOOL_OUTPUT:[^\\]]*\\][\\s\\S]*?\\[\\/TOOL_OUTPUT\\]',
      '\\[GRADCAM_IMAGE:[^\\]]+\\]',
      '\\[COMPARISON_IMAGE:[^\\]]+\\]',
      '\\[GRADCAM_COMPARE:[^:\\]]+:[^\\]]+\\]',
      '\\[RESULT\\][\\s\\S]*?\\[\\/RESULT\\]',
      '\\[ERROR\\][\\s\\S]*?\\[\\/ERROR\\]',
      '\\[COMPLETE\\][\\s\\S]*?\\[\\/COMPLETE\\]',
      '\\[REFERENCES\\][\\s\\S]*?\\[\\/REFERENCES\\]',
      '\\[OBSERVATION\\][\\s\\S]*?\\[\\/OBSERVATION\\]',
      '\\[CONFIRM:[^\\]]*\\][\\s\\S]*?\\[\\/CONFIRM\\]',
    ].join('|') +
    ')',
  'g',
);

// Strips known opening tags that haven't yet been closed (mid-stream partial content)
function cleanStreamingText(text: string): string {
  return text.replace(
    /\[(STAGE:[^\]]*|THINKING|RESPONSE|TOOL_OUTPUT:[^\]]*|RESULT|ERROR|COMPLETE|REFERENCES|OBSERVATION|CONFIRM:[^\]]*)\]/g,
    '',
  );
}

function parseContent(raw: string): Segment[] {
  const segments: Segment[] = [];

  for (const part of raw.split(TAG_SPLIT_RE)) {
    if (!part) continue;

    let m: RegExpMatchArray | null;

    if ((m = part.match(/^\[STAGE:([^\]]*)\]([\s\S]*)\[\/STAGE\]$/))) {
      const label = m[2].trim();
      if (label) segments.push({ type: 'stage', label });

    } else if ((m = part.match(/^\[THINKING\]([\s\S]*)\[\/THINKING\]$/))) {
      const c = m[1].trim();
      if (c) segments.push({ type: 'thinking', content: c });

    } else if ((m = part.match(/^\[RESPONSE\]([\s\S]*)\[\/RESPONSE\]$/))) {
      const c = m[1].trim();
      if (c) segments.push({ type: 'response', content: c });

    } else if ((m = part.match(/^\[TOOL_OUTPUT:([^\]]*)\]([\s\S]*)\[\/TOOL_OUTPUT\]$/))) {
      segments.push({ type: 'tool_output', label: m[1], content: m[2] });

    } else if ((m = part.match(/^\[GRADCAM_IMAGE:([^\]]+)\]$/))) {
      segments.push({ type: 'gradcam', path: m[1] });

    } else if ((m = part.match(/^\[COMPARISON_IMAGE:([^\]]+)\]$/))) {
      segments.push({ type: 'comparison', path: m[1] });

    } else if ((m = part.match(/^\[GRADCAM_COMPARE:([^:\]]+):([^\]]+)\]$/))) {
      segments.push({ type: 'gradcam_compare', path1: m[1], path2: m[2] });

    } else if ((m = part.match(/^\[RESULT\]([\s\S]*)\[\/RESULT\]$/))) {
      const c = m[1].trim();
      if (c) segments.push({ type: 'result', content: c });

    } else if ((m = part.match(/^\[ERROR\]([\s\S]*)\[\/ERROR\]$/))) {
      const c = m[1].trim();
      if (c) segments.push({ type: 'error', content: c });

    } else if ((m = part.match(/^\[COMPLETE\]([\s\S]*)\[\/COMPLETE\]$/))) {
      const c = m[1].trim();
      if (c) segments.push({ type: 'complete', content: c });

    } else if ((m = part.match(/^\[REFERENCES\]([\s\S]*)\[\/REFERENCES\]$/))) {
      segments.push({ type: 'references', content: m[1].trim() });

    } else if ((m = part.match(/^\[OBSERVATION\]([\s\S]*)\[\/OBSERVATION\]$/))) {
      const c = m[1].trim();
      if (c) segments.push({ type: 'observation', content: c });

    } else if ((m = part.match(/^\[CONFIRM:[^\]]*\]([\s\S]*)\[\/CONFIRM\]$/))) {
      const c = m[1].trim();
      if (c) segments.push({ type: 'result', content: c });

    } else {
      // Plain text (may be mid-stream with incomplete opening tags)
      const cleaned = cleanStreamingText(part);
      if (cleaned.trim()) segments.push({ type: 'text', content: cleaned });
    }
  }

  return segments;
}

// ─── References renderer ───────────────────────────────────────────────────

function References({ content }: { content: string }) {
  const refs = content.match(/\[REF:[^\]]+\]/g) ?? [];
  if (!refs.length) return null;

  return (
    <div className="mc-references">
      <div className="mc-references-title">References</div>
      {refs.map((ref, i) => {
        // [REF:id:source:page:file:superscript]
        const parts = ref.slice(1, -1).split(':');
        const source = parts[2] ?? '';
        const page = parts[3] ?? '';
        const sup = parts[5] ?? `[${i + 1}]`;
        return (
          <div key={i} className="mc-ref-item">
            <span className="mc-ref-sup">{sup}</span>
            <span className="mc-ref-source">{source}</span>
            {page && <span className="mc-ref-page">, p.{page}</span>}
          </div>
        );
      })}
    </div>
  );
}

// ─── Main component ────────────────────────────────────────────────────────

export function MessageContent({ text }: { text: string }) {
  const segments = parseContent(text);

  return (
    <div className="mc-root">
      {segments.map((seg, i) => {
        switch (seg.type) {
          case 'stage':
            return <div key={i} className="mc-stage">{seg.label}</div>;

          case 'thinking': {
            // Spinner only on the last thinking segment (earlier ones are done)
            const isLast = !segments.slice(i + 1).some(s => s.type !== 'text' || s.content.trim());
            return (
              <div key={i} className="mc-thinking">
                {isLast
                  ? <span className="mc-thinking-spinner" />
                  : <span className="mc-thinking-done" />}
                {seg.content}
              </div>
            );
          }

          case 'response':
            return (
              <div key={i} className="mc-response">
                <ReactMarkdown>{seg.content}</ReactMarkdown>
              </div>
            );

          case 'tool_output':
            return (
              <div key={i} className="mc-tool-output">
                {seg.label && <div className="mc-tool-output-label">{seg.label}</div>}
                <pre>{seg.content}</pre>
              </div>
            );

          case 'gradcam':
            return (
              <div key={i} className="mc-image-block">
                <div className="mc-image-label">Grad-CAM Attention Map</div>
                <img
                  src={TEMP_IMG_URL(seg.path)}
                  className="mc-gradcam-img"
                  alt="Grad-CAM attention map"
                />
              </div>
            );

          case 'comparison':
            return (
              <div key={i} className="mc-image-block">
                <div className="mc-image-label">Lesion Comparison</div>
                <img
                  src={TEMP_IMG_URL(seg.path)}
                  className="mc-comparison-img"
                  alt="Side-by-side lesion comparison"
                />
              </div>
            );

          case 'gradcam_compare':
            return (
              <div key={i} className="mc-image-block">
                <div className="mc-image-label">Grad-CAM Comparison</div>
                <div className="mc-gradcam-compare">
                  <div className="mc-gradcam-compare-item">
                    <div className="mc-gradcam-compare-title">Previous</div>
                    <img
                      src={TEMP_IMG_URL(seg.path1)}
                      className="mc-gradcam-compare-img"
                      alt="Previous GradCAM"
                    />
                  </div>
                  <div className="mc-gradcam-compare-item">
                    <div className="mc-gradcam-compare-title">Current</div>
                    <img
                      src={TEMP_IMG_URL(seg.path2)}
                      className="mc-gradcam-compare-img"
                      alt="Current GradCAM"
                    />
                  </div>
                </div>
              </div>
            );

          case 'result':
            return <div key={i} className="mc-result">{seg.content}</div>;

          case 'error':
            return <div key={i} className="mc-error">{seg.content}</div>;

          case 'complete':
            return <div key={i} className="mc-complete">{seg.content}</div>;

          case 'references':
            return <References key={i} content={seg.content} />;

          case 'observation':
            return <div key={i} className="mc-observation">{seg.content}</div>;

          case 'text':
            return seg.content.trim() ? (
              <div key={i} className="mc-text">{seg.content}</div>
            ) : null;

          default:
            return null;
        }
      })}
    </div>
  );
}

export interface Patient {
  id: string;
  name: string;
  created_at: string;
}

export interface ToolCall {
  id: string;
  tool: 'analyze_image' | 'compare_images' | string;
  status: 'calling' | 'complete' | 'error';
  result?: {
    // analyze_image
    diagnosis?: string;
    full_name?: string;
    confidence?: number;
    all_predictions?: { class: string; full_name: string; probability: number }[];
    image_url?: string;
    // compare_images
    status_label?: 'STABLE' | 'MINOR_CHANGE' | 'SIGNIFICANT_CHANGE' | 'IMPROVED';
    feature_changes?: Record<string, { prev: number; curr: number }>;
    summary?: string;
    prev_image_url?: string;
    curr_image_url?: string;
  };
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: string;
  image_url?: string;
  tool_calls?: ToolCall[];
}

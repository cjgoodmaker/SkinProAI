interface StreamChatCallbacks {
  onText: (chunk: string) => void;
  onToolStart: (tool: string, callId: string) => void;
  onToolResult: (tool: string, callId: string, result: Record<string, unknown>) => void;
  onDone: () => void;
  onError: (message: string) => void;
}

export async function streamChatMessage(
  patientId: string,
  content: string,
  image: File | null,
  callbacks: StreamChatCallbacks
): Promise<void> {
  try {
    const formData = new FormData();
    formData.append('content', content);
    if (image) formData.append('image', image);

    const response = await fetch(`/api/patients/${patientId}/chat`, {
      method: 'POST',
      body: formData,
    });

    const reader = response.body?.getReader();
    const decoder = new TextDecoder();

    if (!reader) {
      callbacks.onDone();
      return;
    }

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const text = decoder.decode(value);
      const lines = text.split('\n').filter(l => l.startsWith('data: '));

      for (const line of lines) {
        const data = line.slice('data: '.length);
        try {
          const event = JSON.parse(data);
          if (event.type === 'text') {
            callbacks.onText(event.content ?? '');
          } else if (event.type === 'tool_start') {
            callbacks.onToolStart(event.tool, event.call_id);
          } else if (event.type === 'tool_result') {
            callbacks.onToolResult(event.tool, event.call_id, event.result ?? {});
          } else if (event.type === 'done') {
            callbacks.onDone();
            return;
          } else if (event.type === 'error') {
            callbacks.onError(event.message ?? 'Unknown error');
          }
        } catch {
          // skip malformed lines
        }
      }
    }
    callbacks.onDone();
  } catch (error) {
    console.error('Stream error:', error);
    callbacks.onError(String(error));
  }
}

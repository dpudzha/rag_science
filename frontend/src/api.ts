import { Source, ConfigValues } from './types';

interface StreamMetadata {
  sources?: Source[];
  session_id?: string;
  relevance_score?: number;
  retry_count?: number;
  tool_used?: string;
}

export async function queryStream(
  question: string,
  sessionId: string,
  onToken: (token: string) => void,
  onMetadata: (meta: StreamMetadata) => void,
  onDone: () => void,
  onError: (error: string) => void
): Promise<void> {
  try {
    const response = await fetch('/api/query/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, session_id: sessionId }),
    });

    if (!response.ok) {
      onError(`HTTP ${response.status}: ${response.statusText}`);
      return;
    }

    const reader = response.body?.getReader();
    if (!reader) {
      onError('No response body');
      return;
    }

    const decoder = new TextDecoder();
    let buffer = '';
    let currentEvent = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() ?? '';

      for (const line of lines) {
        if (line.startsWith('event: ')) {
          currentEvent = line.slice(7).trim();
          continue;
        }
        if (!line.startsWith('data: ')) continue;
        const raw = line.slice(6);

        try {
          const data = JSON.parse(raw);
          switch (currentEvent) {
            case 'metadata':
              onMetadata(data);
              break;
            case 'token':
              onToken(data.token);
              break;
            case 'done':
              onDone();
              return;
            case 'error':
              onError(data.detail ?? 'Unknown error');
              return;
          }
        } catch {
          // skip malformed lines
        }
        currentEvent = '';
      }
    }

    onDone();
  } catch (err) {
    onError(err instanceof Error ? err.message : 'Network error');
  }
}

export async function triggerIngest(): Promise<{ status: string; detail: string }> {
  const response = await fetch('/api/ingest', { method: 'POST' });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(body.detail ?? `HTTP ${response.status}`);
  }
  return response.json();
}

export async function getConfig(): Promise<ConfigValues> {
  const response = await fetch('/api/config');
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const body = await response.json();
  return body.config;
}

export async function updateConfig(config: Partial<ConfigValues>): Promise<ConfigValues> {
  const response = await fetch('/api/config', {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(config),
  });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const body = await response.json();
  return body.config;
}

export async function saveConfig(): Promise<void> {
  const response = await fetch('/api/config/save', { method: 'POST' });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
}

export async function loadConfig(): Promise<ConfigValues> {
  const response = await fetch('/api/config/load', { method: 'POST' });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const body = await response.json();
  return body.config;
}

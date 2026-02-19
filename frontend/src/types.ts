export interface Source {
  file: string;
  page: number | string;
}

export interface Message {
  role: 'user' | 'assistant';
  content: string;
  sources?: Source[];
  relevanceScore?: number;
  toolUsed?: string;
  retryCount?: number;
  streaming?: boolean;
}

export interface ConfigValues {
  [key: string]: string | number | boolean;
}

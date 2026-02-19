import { useState, useRef, useEffect, useCallback } from 'react'
import { Message } from '../types'
import { queryStream } from '../api'
import './ChatPanel.css'

function generateId(): string {
  return crypto.randomUUID()
}

function ChatPanel() {
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [sessionId] = useState(generateId)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const scrollToBottom = useCallback(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages, scrollToBottom])

  const handleSend = async () => {
    const question = input.trim()
    if (!question || isStreaming) return

    setInput('')
    setMessages(prev => [...prev, { role: 'user', content: question }])
    setIsStreaming(true)

    const assistantIndex = messages.length + 1

    setMessages(prev => [
      ...prev,
      { role: 'assistant', content: '', streaming: true },
    ])

    await queryStream(
      question,
      sessionId,
      (token) => {
        setMessages(prev => {
          const updated = [...prev]
          const msg = updated[assistantIndex]
          if (msg) {
            updated[assistantIndex] = { ...msg, content: msg.content + token }
          }
          return updated
        })
      },
      (meta) => {
        setMessages(prev => {
          const updated = [...prev]
          const msg = updated[assistantIndex]
          if (msg) {
            updated[assistantIndex] = {
              ...msg,
              sources: meta.sources,
              relevanceScore: meta.relevance_score,
              toolUsed: meta.tool_used,
              retryCount: meta.retry_count,
            }
          }
          return updated
        })
      },
      () => {
        setMessages(prev => {
          const updated = [...prev]
          const msg = updated[assistantIndex]
          if (msg) {
            updated[assistantIndex] = { ...msg, streaming: false }
          }
          return updated
        })
        setIsStreaming(false)
        inputRef.current?.focus()
      },
      (error) => {
        setMessages(prev => {
          const updated = [...prev]
          const msg = updated[assistantIndex]
          if (msg) {
            updated[assistantIndex] = {
              ...msg,
              content: msg.content || `Error: ${error}`,
              streaming: false,
            }
          }
          return updated
        })
        setIsStreaming(false)
      }
    )
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="chat-panel">
      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            Ask a question about your scientific papers
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`chat-bubble ${msg.role}`}>
            <div className="bubble-content">
              {msg.content}
              {msg.streaming && <span className="cursor-blink" />}
            </div>
            {msg.role === 'assistant' && msg.relevanceScore != null && (
              <span className="relevance-badge">
                relevance: {(msg.relevanceScore * 100).toFixed(0)}%
              </span>
            )}
            {msg.role === 'assistant' && msg.retryCount != null && msg.retryCount > 0 && (
              <span className="retry-badge">
                retried {msg.retryCount}x
              </span>
            )}
            {msg.role === 'assistant' && msg.sources && msg.sources.length > 0 && (
              <SourcesList sources={msg.sources} />
            )}
          </div>
        ))}
        <div ref={messagesEndRef} />
      </div>
      <div className="chat-input-bar">
        <input
          ref={inputRef}
          className="chat-input"
          type="text"
          placeholder="Type your question..."
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isStreaming}
        />
        <button
          className="chat-send"
          onClick={handleSend}
          disabled={isStreaming || !input.trim()}
        >
          Send
        </button>
      </div>
    </div>
  )
}

function SourcesList({ sources }: { sources: { file: string; page: number | string }[] }) {
  const [open, setOpen] = useState(false)

  return (
    <div className="sources">
      <button className="sources-toggle" onClick={() => setOpen(o => !o)}>
        {open ? 'Hide' : 'Show'} sources ({sources.length})
      </button>
      {open && (
        <ul className="sources-list">
          {sources.map((s, i) => (
            <li key={i}>{s.file} (p. {s.page})</li>
          ))}
        </ul>
      )}
    </div>
  )
}

export default ChatPanel

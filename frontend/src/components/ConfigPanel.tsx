import { useState, useEffect } from 'react'
import { ConfigValues } from '../types'
import { getConfig, updateConfig, saveConfig, loadConfig } from '../api'
import './ConfigPanel.css'

const MODEL_KEYS = ['LLM_MODEL', 'EMBEDDING_MODEL', 'RERANK_MODEL']
const RETRIEVAL_KEYS = ['TOP_K', 'TOP_K_CANDIDATES', 'RELEVANCE_THRESHOLD', 'BM25_WEIGHT', 'DENSE_WEIGHT']
const CHUNKING_KEYS = ['CHUNK_SIZE', 'CHUNK_OVERLAP']
const SLIDER_KEYS = ['BM25_WEIGHT', 'DENSE_WEIGHT']

const SECTION_ORDER: { title: string; keys: string[] }[] = [
  { title: 'Models', keys: MODEL_KEYS },
  { title: 'Retrieval', keys: RETRIEVAL_KEYS },
  { title: 'Chunking', keys: CHUNKING_KEYS },
]

function isBoolean(val: unknown): val is boolean {
  return typeof val === 'boolean'
}

function ConfigPanel() {
  const [config, setConfig] = useState<ConfigValues>({})
  const [loading, setLoading] = useState(true)
  const [status, setStatus] = useState<{ type: 'success' | 'error'; message: string } | null>(null)

  useEffect(() => {
    getConfig()
      .then(setConfig)
      .catch(() => setStatus({ type: 'error', message: 'Failed to load config' }))
      .finally(() => setLoading(false))
  }, [])

  const showStatus = (type: 'success' | 'error', message: string) => {
    setStatus({ type, message })
    setTimeout(() => setStatus(null), 3000)
  }

  const handleChange = (key: string, value: string | number | boolean) => {
    setConfig(prev => ({ ...prev, [key]: value }))
  }

  const handleApply = async () => {
    try {
      const updated = await updateConfig(config)
      setConfig(updated)
      showStatus('success', 'Configuration applied.')
    } catch {
      showStatus('error', 'Failed to apply configuration.')
    }
  }

  const handleSave = async () => {
    try {
      await saveConfig()
      showStatus('success', 'Configuration saved to file.')
    } catch {
      showStatus('error', 'Failed to save configuration.')
    }
  }

  const handleLoad = async () => {
    try {
      const loaded = await loadConfig()
      setConfig(loaded)
      showStatus('success', 'Configuration loaded from file.')
    } catch {
      showStatus('error', 'Failed to load configuration from file.')
    }
  }

  if (loading) {
    return <div className="config-panel"><p className="config-loading">Loading configuration...</p></div>
  }

  const knownKeys = new Set([...MODEL_KEYS, ...RETRIEVAL_KEYS, ...CHUNKING_KEYS])
  const flagKeys = Object.keys(config).filter(k => !knownKeys.has(k) && isBoolean(config[k]))

  return (
    <div className="config-panel">
      <h2 className="config-title">Configuration</h2>

      {SECTION_ORDER.map(section => (
        <div key={section.title} className="config-section">
          <h3 className="config-section-title">{section.title}</h3>
          <div className="config-fields">
            {section.keys.map(key => {
              const val = config[key]
              if (val === undefined) return null

              if (SLIDER_KEYS.includes(key)) {
                return (
                  <label key={key} className="config-field">
                    <span className="config-label">{key}</span>
                    <div className="slider-row">
                      <input
                        type="range"
                        min="0"
                        max="1"
                        step="0.1"
                        value={Number(val)}
                        onChange={e => handleChange(key, parseFloat(e.target.value))}
                      />
                      <span className="slider-value">{Number(val).toFixed(1)}</span>
                    </div>
                  </label>
                )
              }

              if (typeof val === 'number') {
                return (
                  <label key={key} className="config-field">
                    <span className="config-label">{key}</span>
                    <input
                      type="number"
                      className="config-input"
                      value={val}
                      onChange={e => handleChange(key, parseFloat(e.target.value) || 0)}
                    />
                  </label>
                )
              }

              return (
                <label key={key} className="config-field">
                  <span className="config-label">{key}</span>
                  <input
                    type="text"
                    className="config-input"
                    value={String(val)}
                    onChange={e => handleChange(key, e.target.value)}
                  />
                </label>
              )
            })}
          </div>
        </div>
      ))}

      {flagKeys.length > 0 && (
        <div className="config-section">
          <h3 className="config-section-title">Feature Flags</h3>
          <div className="config-flags">
            {flagKeys.map(key => (
              <label key={key} className="config-toggle">
                <input
                  type="checkbox"
                  checked={config[key] as boolean}
                  onChange={e => handleChange(key, e.target.checked)}
                />
                <span className="toggle-switch" />
                <span className="toggle-label">{key}</span>
              </label>
            ))}
          </div>
        </div>
      )}

      <div className="config-actions">
        <button className="config-btn primary" onClick={handleApply}>Apply</button>
        <button className="config-btn" onClick={handleSave}>Save to File</button>
        <button className="config-btn" onClick={handleLoad}>Load from File</button>
      </div>

      {status && (
        <div className={`config-status ${status.type}`}>{status.message}</div>
      )}
    </div>
  )
}

export default ConfigPanel

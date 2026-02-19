import { useState } from 'react'
import { triggerIngest } from '../api'
import './IngestPanel.css'

type Status = 'idle' | 'ingesting' | 'done' | 'error'

function IngestPanel() {
  const [status, setStatus] = useState<Status>('idle')
  const [detail, setDetail] = useState('')

  const handleIngest = async () => {
    setStatus('ingesting')
    setDetail('')
    try {
      const result = await triggerIngest()
      setStatus('done')
      setDetail(result.detail ?? 'Ingestion complete.')
    } catch (err) {
      setStatus('error')
      setDetail(err instanceof Error ? err.message : 'Unknown error')
    }
  }

  return (
    <div className="ingest-panel">
      <h2 className="ingest-title">Document Ingestion</h2>
      <p className="ingest-instructions">
        Place your documents (PDF, DOCX, XLSX) in the <code>papers/</code> directory,
        then click the button below.
      </p>
      <button
        className="ingest-button"
        onClick={handleIngest}
        disabled={status === 'ingesting'}
      >
        {status === 'ingesting' ? (
          <>
            <span className="spinner" /> Ingesting...
          </>
        ) : (
          'Ingest Papers'
        )}
      </button>
      {status === 'done' && (
        <div className="ingest-status success">{detail}</div>
      )}
      {status === 'error' && (
        <div className="ingest-status error">{detail}</div>
      )}
    </div>
  )
}

export default IngestPanel

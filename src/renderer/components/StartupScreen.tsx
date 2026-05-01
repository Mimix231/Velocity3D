import React, { useEffect, useRef, useState } from 'react'
import './StartupScreen.css'

interface Props {
  onReady: () => void
  onError: (msg: string) => void
}

export default function StartupScreen({ onReady, onError }: Props) {
  const [logs, setLogs] = useState<string[]>(['Velocity3D starting up...'])
  const [venvPath, setVenvPath] = useState('')
  const [showVenvInput, setShowVenvInput] = useState(false)
  const [status, setStatus] = useState<'starting' | 'error' | 'ready'>('starting')
  const [errorMsg, setErrorMsg] = useState('')
  const logsEndRef = useRef<HTMLDivElement>(null)

  const addLog = (line: string) => setLogs((prev) => [...prev, line])

  useEffect(() => {
    // Load saved venv path
    window.velocityAPI.getVenvPath().then((p) => {
      if (p) setVenvPath(p)
    })

    const unsubLog = window.velocityAPI.onBackendLog(addLog)
    const unsubStatus = window.velocityAPI.onBackendStatus((event) => {
      if (event.status === 'ready') {
        setStatus('ready')
        setTimeout(onReady, 600) // brief pause so user sees "ready"
      }
      if (event.status === 'error') {
        setStatus('error')
        setErrorMsg(event.message ?? 'Backend failed to start')
        // Don't call onError — stay on startup screen so user can fix venv
      }
      if (event.status === 'starting') {
        addLog(event.message ?? 'Starting...')
      }
    })

    return () => {
      unsubLog()
      unsubStatus()
    }
  }, [])

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  async function handleBrowseVenv() {
    const selected = await window.velocityAPI.showOpenDialog({
      title: 'Select Python venv folder',
      properties: ['openDirectory']
    })
    if (selected) setVenvPath(selected)
  }

  async function handleSaveVenv() {
    await window.velocityAPI.setVenvPath(venvPath)
    setShowVenvInput(false)
    addLog(`HOME_VENV set to: ${venvPath}`)
    addLog('Restart the app to apply the new venv.')
  }

  return (
    <div className="startup-root">
      <div className="startup-header">
        <span className="startup-logo">Velocity3D</span>
        <span className="startup-subtitle">AI-Powered 3D Generation</span>
      </div>

      <div className="startup-terminal">
        <div className="startup-terminal-bar">
          <span className="startup-terminal-dot red" />
          <span className="startup-terminal-dot yellow" />
          <span className="startup-terminal-dot green" />
          <span className="startup-terminal-title">backend — uvicorn</span>
        </div>
        <div className="startup-logs">
          {logs.map((line, i) => (
            <div key={i} className="startup-log-line">
              <span className="startup-log-prompt">{'>'}</span>
              <span>{line}</span>
            </div>
          ))}
          {status === 'starting' && (
            <div className="startup-log-line">
              <span className="startup-log-prompt">{'>'}</span>
              <span className="startup-cursor">█</span>
            </div>
          )}
          {status === 'ready' && (
            <div className="startup-log-line ready">
              <span className="startup-log-prompt">{'>'}</span>
              <span>✓ Backend ready — launching Velocity3D...</span>
            </div>
          )}
          <div ref={logsEndRef} />
        </div>
      </div>

      <div className="startup-footer">
        {status === 'error' && (
          <div className="startup-error-banner">
            ✕ {errorMsg}
          </div>
        )}

        <div className="startup-venv-row">
          <span className="startup-venv-label">Python venv:</span>
          {showVenvInput ? (
            <>
              <input
                className="startup-venv-input"
                value={venvPath}
                onChange={(e) => setVenvPath(e.target.value)}
                placeholder="Path to venv root (e.g. G:/project/env)"
              />
              <button className="startup-btn" onClick={handleBrowseVenv}>Browse</button>
              <button className="startup-btn primary" onClick={handleSaveVenv}>Save</button>
              <button className="startup-btn" onClick={() => setShowVenvInput(false)}>Cancel</button>
            </>
          ) : (
            <>
              <span className="startup-venv-value">{venvPath || 'Not set (using system Python)'}</span>
              <button className="startup-btn" onClick={() => setShowVenvInput(true)}>
                {venvPath ? 'Change' : 'Set venv'}
              </button>
            </>
          )}
        </div>
      </div>
    </div>
  )
}

import React, { useState } from 'react'
import { useThemeStore, THEMES, ThemeId } from '../stores/themeStore'
import './SetupWizard.css'

interface Props {
  onComplete: (venvPath: string) => void
}

export default function SetupWizard({ onComplete }: Props) {
  const [step, setStep] = useState<'welcome' | 'venv' | 'theme' | 'done'>('welcome')
  const [venvPath, setVenvPath] = useState('')
  const { themeId, setTheme } = useThemeStore()

  async function handleBrowse() {
    const p = await window.velocityAPI.showOpenDialog({
      title: 'Select your Python venv folder',
      properties: ['openDirectory'],
    })
    if (p) setVenvPath(p)
  }

  async function handleFinish() {
    await window.velocityAPI.setConfig({ venvPath, setupComplete: true })
    onComplete(venvPath)
  }

  return (
    <div className="setup-root">
      <div className="setup-card">
        {/* Progress dots */}
        <div className="setup-steps">
          {(['welcome', 'venv', 'theme'] as const).map((s, i) => (
            <div key={s} className={`setup-step-dot ${step === s ? 'active' : i < ['welcome','venv','theme'].indexOf(step) ? 'done' : ''}`} />
          ))}
        </div>

        {step === 'welcome' && (
          <div className="setup-page">
            <div className="setup-icon">⚡</div>
            <h1 className="setup-title">Welcome to Velocity3D</h1>
            <p className="setup-desc">
              AI-powered 3D model generation from text and images.<br />
              Let's get you set up in 2 quick steps.
            </p>
            <button className="setup-btn-primary" onClick={() => setStep('venv')}>
              Get Started →
            </button>
          </div>
        )}

        {step === 'venv' && (
          <div className="setup-page">
            <div className="setup-icon">🐍</div>
            <h1 className="setup-title">Python Environment</h1>
            <p className="setup-desc">
              Point Velocity3D to the backend virtual environment you want to run.<br />
              Python 3.11 is the safest default. Some upstream 3D providers still target Python 3.10-3.12, and the Model Browser will show those compatibility notes.
            </p>
            <div className="setup-venv-row">
              <input
                className="setup-input"
                value={venvPath}
                onChange={(e) => setVenvPath(e.target.value)}
                placeholder="e.g. G:/AIExperiments/Velocity3D/env"
              />
              <button className="setup-btn-secondary" onClick={handleBrowse}>Browse</button>
            </div>
            {venvPath && (
              <div className="setup-venv-hint">
                Python: <code>{venvPath}{navigator.platform.includes('Win') ? '\\Scripts\\python.exe' : '/bin/python3'}</code>
              </div>
            )}
            <div className="setup-nav">
              <button className="setup-btn-ghost" onClick={() => setStep('welcome')}>← Back</button>
              <button className="setup-btn-primary" onClick={() => setStep('theme')}>
                {venvPath ? 'Next →' : 'Skip for now →'}
              </button>
            </div>
          </div>
        )}

        {step === 'theme' && (
          <div className="setup-page">
            <div className="setup-icon">🎨</div>
            <h1 className="setup-title">Choose a Theme</h1>
            <p className="setup-desc">Pick your preferred look. You can change this anytime.</p>
            <div className="setup-themes">
              {(Object.values(THEMES)).map((t) => (
                <button
                  key={t.id}
                  className={`setup-theme-chip ${themeId === t.id ? 'selected' : ''}`}
                  style={{ '--chip-accent': t.accent, '--chip-bg': t.bg, '--chip-text': t.text } as any}
                  onClick={() => setTheme(t.id as ThemeId)}
                >
                  <span className="setup-theme-swatch" style={{ background: t.bg, border: `2px solid ${t.accent}` }}>
                    <span style={{ background: t.accent, width: 10, height: 10, borderRadius: 2, display: 'block', margin: '4px auto' }} />
                  </span>
                  {t.name}
                  {themeId === t.id && <span className="setup-theme-check">✓</span>}
                </button>
              ))}
            </div>
            <div className="setup-nav">
              <button className="setup-btn-ghost" onClick={() => setStep('venv')}>← Back</button>
              <button className="setup-btn-primary" onClick={handleFinish}>
                Launch Velocity3D →
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

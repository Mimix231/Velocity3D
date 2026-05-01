import React, { useEffect, useMemo, useState } from 'react'

import type { GenerationMode, ModelCatalogItem } from '../api/modelApi'
import { useModelCatalogStore } from '../stores/modelCatalogStore'
import './ModelDownloader.css'

interface Props {
  onClose: () => void
  onSelectModel: (mode: GenerationMode, modelId: string) => void
  onInstallModel: (modelId: string, mode: GenerationMode) => void
  preferredMode?: GenerationMode
}

type BrowserFilter = 'all' | 'text' | 'image' | 'generators' | 'assistants' | 'installed'

function statusLabel(model: ModelCatalogItem): string {
  switch (model.status) {
    case 'ready':
      return 'Ready'
    case 'downloaded':
      return 'Downloaded'
    case 'library_only':
      return 'Assistant'
    default:
      return 'Install / Download'
  }
}

function filterLabel(mode: GenerationMode | 'multiview') {
  if (mode === 'multiview') return 'Multiview'
  return `${mode[0].toUpperCase()}${mode.slice(1)} to 3D`
}

function chooseInstallMode(model: ModelCatalogItem, preferredMode: GenerationMode): GenerationMode {
  if (model.selection_modes.includes(preferredMode)) {
    return preferredMode
  }
  if (model.selection_modes.includes('image')) {
    return 'image'
  }
  if (model.selection_modes.includes('text')) {
    return 'text'
  }
  return preferredMode
}

export default function ModelDownloader({ onClose, onSelectModel, onInstallModel, preferredMode }: Props) {
  const models = useModelCatalogStore((state) => state.models)
  const loaded = useModelCatalogStore((state) => state.loaded)
  const loading = useModelCatalogStore((state) => state.loading)
  const load = useModelCatalogStore((state) => state.load)
  const refresh = useModelCatalogStore((state) => state.refresh)
  const textModelId = useModelCatalogStore((state) => state.textModelId)
  const imageModelId = useModelCatalogStore((state) => state.imageModelId)

  const [copied, setCopied] = useState<string | null>(null)
  const [filter, setFilter] = useState<BrowserFilter>(preferredMode ?? 'all')
  const [search, setSearch] = useState('')
  const [selectedModelId, setSelectedModelId] = useState<string | null>(null)

  useEffect(() => {
    if (!loaded && !loading) {
      void load()
    }
  }, [loaded, loading, load])

  const summary = useMemo(() => ({
    ready: models.filter((model) => model.generation_ready).length,
    installed: models.filter((model) => model.downloaded).length,
    generators: models.filter((model) => model.role === 'generator').length,
    assistants: models.filter((model) => model.role === 'assistant').length
  }), [models])

  const visibleModels = useMemo(() => {
    const query = search.trim().toLowerCase()

    return models.filter((model) => {
      if (filter === 'text' && !model.selection_modes.includes('text')) return false
      if (filter === 'image' && !model.selection_modes.includes('image')) return false
      if (filter === 'generators' && model.role !== 'generator') return false
      if (filter === 'assistants' && model.role !== 'assistant') return false
      if (filter === 'installed' && !model.downloaded) return false

      if (!query) return true

      const haystack = [
        model.name,
        model.family,
        model.summary,
        model.description,
        model.license_name ?? '',
        model.vram_hint ?? ''
      ].join(' ').toLowerCase()

      return haystack.includes(query)
    })
  }, [filter, models, search])

  useEffect(() => {
    if (visibleModels.length === 0) {
      setSelectedModelId(null)
      return
    }
    if (!selectedModelId || !visibleModels.some((model) => model.id === selectedModelId)) {
      setSelectedModelId(visibleModels[0].id)
    }
  }, [selectedModelId, visibleModels])

  const selectedModel = useMemo(
    () => models.find((model) => model.id === selectedModelId) ?? visibleModels[0] ?? null,
    [models, selectedModelId, visibleModels]
  )

  function copyInstallSteps(model: ModelCatalogItem) {
    navigator.clipboard.writeText(model.install_steps.join('\n'))
    setCopied(model.id)
    setTimeout(() => setCopied(null), 1800)
  }

  function handleInstall(model: ModelCatalogItem) {
    onInstallModel(model.id, chooseInstallMode(model, preferredMode ?? 'image'))
    onClose()
  }

  return (
    <div className="downloader-overlay" onClick={(event) => event.target === event.currentTarget && onClose()}>
      <div className="downloader-panel">
        <div className="downloader-header">
          <div className="downloader-header-copy">
            <div className="downloader-title">Model Browser</div>
            <div className="downloader-subtitle">
              Download, install, and assign the providers that power Velocity3D.
            </div>
          </div>

          <div className="downloader-header-actions">
            <div className="downloader-summary-strip">
              <span className="downloader-summary-chip">Ready {summary.ready}</span>
              <span className="downloader-summary-chip">Installed {summary.installed}</span>
              <span className="downloader-summary-chip">Generators {summary.generators}</span>
              <span className="downloader-summary-chip">Assistants {summary.assistants}</span>
            </div>
            <button className="downloader-close" aria-label="Close model browser" onClick={onClose}>x</button>
          </div>
        </div>

        <div className="downloader-toolbar">
          <div className="downloader-filter-group" role="tablist" aria-label="Model browser filters">
            <button className={`downloader-filter-btn ${filter === 'all' ? 'active' : ''}`} onClick={() => setFilter('all')}>All</button>
            <button className={`downloader-filter-btn ${filter === 'text' ? 'active' : ''}`} onClick={() => setFilter('text')}>Text</button>
            <button className={`downloader-filter-btn ${filter === 'image' ? 'active' : ''}`} onClick={() => setFilter('image')}>Image</button>
            <button className={`downloader-filter-btn ${filter === 'generators' ? 'active' : ''}`} onClick={() => setFilter('generators')}>Generators</button>
            <button className={`downloader-filter-btn ${filter === 'assistants' ? 'active' : ''}`} onClick={() => setFilter('assistants')}>Assistants</button>
            <button className={`downloader-filter-btn ${filter === 'installed' ? 'active' : ''}`} onClick={() => setFilter('installed')}>Installed</button>
          </div>

          <div className="downloader-toolbar-right">
            <input
              className="downloader-search"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search models, licenses, or families"
            />
            <button className="downloader-refresh-btn" onClick={() => void refresh()}>
              Refresh status
            </button>
          </div>
        </div>

        <div className="downloader-note">
          Providers install into <code>BASE_DIR/models</code>. Checkpoints stay in <code>BASE_DIR/Checkpoints</code>, and progress streams through the inspector terminal.
        </div>

        <div className="downloader-browser-body">
          <aside className="downloader-family-rail" aria-label="Model categories">
            {([
              ['all', 'All models'],
              ['generators', '3D generators'],
              ['image', 'Image pipelines'],
              ['text', 'Text pipelines'],
              ['installed', 'Installed'],
              ['assistants', 'Assistants'],
            ] as Array<[BrowserFilter, string]>).map(([id, label]) => (
              <button
                key={id}
                className={`downloader-family-btn ${filter === id ? 'active' : ''}`}
                onClick={() => setFilter(id)}
              >
                <span>{label}</span>
                <strong>
                  {models.filter((model) => {
                    if (id === 'all') return true
                    if (id === 'generators') return model.role === 'generator'
                    if (id === 'assistants') return model.role === 'assistant'
                    if (id === 'installed') return model.downloaded
                    return model.selection_modes.includes(id)
                  }).length}
                </strong>
              </button>
            ))}
          </aside>

          <div className="downloader-list">
            {loading && (
              <div className="downloader-empty">
                Refreshing model status...
              </div>
            )}

            {visibleModels.length === 0 && !loading && (
              <div className="downloader-empty">
                No models match this filter.
              </div>
            )}

            {visibleModels.map((model) => (
              <article
                key={model.id}
                className={`downloader-item ${model.recommended ? 'recommended' : ''} ${selectedModel?.id === model.id ? 'active' : ''}`}
                onClick={() => setSelectedModelId(model.id)}
              >
                <div className="downloader-item-top">
                  <div className="downloader-item-heading">
                    <div className="downloader-item-name-row">
                      <span className="downloader-item-name">{model.name}</span>
                      {model.recommended && <span className="downloader-recommended">Recommended</span>}
                      <span className={`downloader-status-badge ${model.status}`}>{statusLabel(model)}</span>
                    </div>

                    <div className="downloader-item-meta">
                      {model.library_modes.map((mode) => (
                        <span key={mode} className="downloader-mode-chip">
                          {filterLabel(mode)}
                        </span>
                      ))}
                      {model.vram_hint && <span>{model.vram_hint}</span>}
                      {model.size_hint && <span>{model.size_hint}</span>}
                    </div>
                  </div>
                </div>

                <p className="downloader-item-desc">{model.summary}</p>
                <div className="downloader-default-actions">
                  {textModelId === model.id && <span className="downloader-default-pill">Text default</span>}
                  {imageModelId === model.id && <span className="downloader-default-pill">Image default</span>}
                </div>
              </article>
            ))}
          </div>

          <aside className="downloader-detail">
            {selectedModel ? (
              <>
                <div className="downloader-detail-kicker">{selectedModel.family}</div>
                <div className="downloader-detail-title-row">
                  <div className="downloader-detail-title">{selectedModel.name}</div>
                  <span className={`downloader-status-badge ${selectedModel.status}`}>{statusLabel(selectedModel)}</span>
                </div>
                <p className="downloader-detail-desc">{selectedModel.description}</p>
                <div className="downloader-detail-grid">
                  <span>Role</span><strong>{selectedModel.role}</strong>
                  <span>VRAM</span><strong>{selectedModel.vram_hint ?? 'Model default'}</strong>
                  <span>Size</span><strong>{selectedModel.size_hint ?? 'Provider managed'}</strong>
                  <span>License</span><strong>{selectedModel.license_name ?? 'See docs'}</strong>
                </div>
                <div className="downloader-status-copy">{selectedModel.status_detail}</div>
                {selectedModel.platform_note && <div className="downloader-platform-note">{selectedModel.platform_note}</div>}
                {selectedModel.python_status_detail && (
                  <div className={`downloader-python-note ${selectedModel.python_compatible === false ? 'warning' : ''}`}>
                    {selectedModel.python_status_detail}
                  </div>
                )}
                <details className="downloader-setup-block">
                  <summary>Install plan</summary>
                  <div className="downloader-install-steps">
                    {selectedModel.install_steps.map((step) => (
                      <code key={step} className="downloader-install-step">{step}</code>
                    ))}
                  </div>
                </details>
                <div className="downloader-actions">
                  <button className="downloader-action-btn primary" onClick={() => handleInstall(selectedModel)}>
                    {selectedModel.generation_ready ? 'Refresh install' : 'Install / Download'}
                  </button>
                  <button className="downloader-action-btn" onClick={() => copyInstallSteps(selectedModel)}>
                    {copied === selectedModel.id ? 'Copied' : 'Copy setup'}
                  </button>
                  {selectedModel.docs_url && (
                    <button className="downloader-action-btn ghost" onClick={() => void window.velocityAPI.openExternal(selectedModel.docs_url!)}>
                      Docs
                    </button>
                  )}
                  {selectedModel.huggingface_url && (
                    <button className="downloader-action-btn ghost" onClick={() => void window.velocityAPI.openExternal(selectedModel.huggingface_url!)}>
                      Hugging Face
                    </button>
                  )}
                </div>
                <div className="downloader-default-actions">
                  {selectedModel.selection_modes.includes('text') && (
                    <button
                      className={`downloader-default-btn ${textModelId === selectedModel.id ? 'active' : ''}`}
                      disabled={!selectedModel.generation_ready}
                      onClick={() => onSelectModel('text', selectedModel.id)}
                    >
                      {textModelId === selectedModel.id ? 'Text default' : 'Use for text'}
                    </button>
                  )}
                  {selectedModel.selection_modes.includes('image') && (
                    <button
                      className={`downloader-default-btn ${imageModelId === selectedModel.id ? 'active' : ''}`}
                      disabled={!selectedModel.generation_ready}
                      onClick={() => onSelectModel('image', selectedModel.id)}
                    >
                      {imageModelId === selectedModel.id ? 'Image default' : 'Use for image'}
                    </button>
                  )}
                </div>
              </>
            ) : (
              <div className="downloader-empty">Select a model to inspect its provider details.</div>
            )}
          </aside>
        </div>
      </div>
    </div>
  )
}

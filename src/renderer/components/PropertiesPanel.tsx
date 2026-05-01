import React from 'react'
import type { TextureOptions } from '../api/generationApi'
import type { GenerationMode, ModelCatalogItem, ModelInstallStatusResponse } from '../api/modelApi'
import { useGenerationStore } from '../stores/generationStore'
import './PropertiesPanel.css'

export interface ObjectProperties {
  id: string
  name: string
  vertexCount: number
  faceCount: number
  position: { x: number; y: number; z: number }
  rotation: { x: number; y: number; z: number }
  scale: { x: number; y: number; z: number }
}

export interface InputImagePreview {
  url: string
  backgroundRemovedUrl?: string | null
  name: string
  size: number
}

interface Props {
  selectedObject?: ObjectProperties | null
  activeMode: GenerationMode
  selectedModel?: ModelCatalogItem | null
  inputImage?: InputImagePreview | null
  textureOptions: TextureOptions
  installSession?: (ModelInstallStatusResponse & { mode: GenerationMode }) | null
  onTextureOptionsChange: (next: TextureOptions) => void
  onInstallModel: (modelId: string, mode: GenerationMode) => void
  onOpenModelLibrary: () => void
  onUpdateObjectTransform?: (
    objectId: string,
    transform: Partial<Pick<ObjectProperties, 'position' | 'rotation' | 'scale'>>
  ) => void
}

interface ModelInspectorProps {
  activeMode: GenerationMode
  selectedModel?: ModelCatalogItem | null
  installSession?: (ModelInstallStatusResponse & { mode: GenerationMode }) | null
  onInstallModel: (modelId: string, mode: GenerationMode) => void
  onOpenModelLibrary: () => void
}

interface InputsSectionProps {
  activeMode: GenerationMode
  selectedModel?: ModelCatalogItem | null
  inputImage?: InputImagePreview | null
  textureOptions: TextureOptions
  onTextureOptionsChange: (next: TextureOptions) => void
}

function Vec3Row({ label, value }: { label: string; value: { x: number; y: number; z: number } }) {
  const fmt = (n: number) => n.toFixed(3)
  return (
    <div className="prop-row">
      <span className="prop-label">{label}</span>
      <div className="prop-vec3">
        <span className="prop-axis x">X</span><span className="prop-val">{fmt(value.x)}</span>
        <span className="prop-axis y">Y</span><span className="prop-val">{fmt(value.y)}</span>
        <span className="prop-axis z">Z</span><span className="prop-val">{fmt(value.z)}</span>
      </div>
    </div>
  )
}

function EditableVec3Row({
  label,
  value,
  mode = 'linear',
  onChange,
}: {
  label: string
  value: { x: number; y: number; z: number }
  mode?: 'linear' | 'rotation'
  onChange: (next: { x: number; y: number; z: number }) => void
}) {
  const toDisplay = (n: number) => mode === 'rotation' ? n * 180 / Math.PI : n
  const fromDisplay = (n: number) => mode === 'rotation' ? n * Math.PI / 180 : n

  const updateAxis = (axis: 'x' | 'y' | 'z', raw: string) => {
    const parsed = Number(raw)
    if (!Number.isFinite(parsed)) return
    onChange({
      ...value,
      [axis]: fromDisplay(parsed)
    })
  }

  return (
    <div className="editable-vec3-row">
      <span className="prop-label">{label}</span>
      <div className="editable-vec3">
        {(['x', 'y', 'z'] as const).map((axis) => (
          <label key={axis} className="editable-axis-field">
            <span className={`prop-axis ${axis}`}>{axis.toUpperCase()}</span>
            <input
              type="number"
              step={mode === 'rotation' ? 1 : 0.05}
              value={Number(toDisplay(value[axis]).toFixed(mode === 'rotation' ? 1 : 3))}
              onChange={(event) => updateAxis(axis, event.target.value)}
            />
          </label>
        ))}
      </div>
    </div>
  )
}

function modelStatusText(model: ModelCatalogItem): string {
  if (model.generation_ready) return 'Ready'
  if (model.downloaded) return 'Install required'
  return 'Not downloaded'
}

function installStatusText(session: ModelInstallStatusResponse): string {
  if (session.status === 'complete') return 'Complete'
  if (session.status === 'manual_required') return 'Manual follow-up'
  if (session.status === 'error') return 'Failed'
  return 'Installing'
}

function hasDownloadableAssets(model: ModelCatalogItem): boolean {
  return Boolean(model.huggingface_url || model.repo_url)
}

function formatFileSize(size: number): string {
  if (size >= 1024 * 1024) return `${(size / 1024 / 1024).toFixed(1)} MB`
  return `${Math.max(1, Math.round(size / 1024))} KB`
}

function InputsSection({
  activeMode,
  selectedModel,
  inputImage,
  textureOptions,
  onTextureOptionsChange,
}: InputsSectionProps) {
  const updateTextureEnabled = (event: React.ChangeEvent<HTMLInputElement>) => {
    onTextureOptionsChange({
      ...textureOptions,
      enabled: event.target.checked,
    })
  }

  const updateTextureCheckpoint = (event: React.ChangeEvent<HTMLInputElement>) => {
    onTextureOptionsChange({
      ...textureOptions,
      checkpoint: event.target.value,
    })
  }

  return (
    <section className="inputs-section">
      <div className="prop-section-title">Inputs</div>

      {activeMode === 'image' ? (
        inputImage ? (
          <div className="input-preview-stack">
            <div className="input-preview-block">
              <div className="input-preview-label">Source image</div>
              <img src={inputImage.url} alt="Selected input" className="input-image-preview" />
              <div className="input-preview-meta">
                <span>{inputImage.name}</span>
                <span>{formatFileSize(inputImage.size)}</span>
              </div>
            </div>
            <div className="input-preview-block">
              <div className="input-preview-label">Background removed</div>
              {inputImage.backgroundRemovedUrl ? (
                <img
                  src={inputImage.backgroundRemovedUrl}
                  alt="Background removed input preview"
                  className="input-image-preview cutout"
                />
              ) : (
                <div className="input-preview-empty">Preparing cutout preview...</div>
              )}
            </div>
          </div>
        ) : (
          <div className="input-preview-empty">No image selected.</div>
        )
      ) : (
        <div className="input-preview-empty">Text prompt is entered in the generator bar.</div>
      )}

      <div className="prop-subsection-title">Settings</div>
      <div className="prop-row">
        <span className="prop-label">Mode</span>
        <span className="prop-value">{activeMode === 'image' ? 'Image to 3D' : 'Text to 3D'}</span>
      </div>
      <div className="prop-row">
        <span className="prop-label">Provider</span>
        <span className="prop-value">{selectedModel?.name ?? 'Not selected'}</span>
      </div>
      <label className="settings-toggle">
        <input
          type="checkbox"
          checked={textureOptions.enabled}
          onChange={updateTextureEnabled}
        />
        <span>Stable Diffusion texture pass</span>
      </label>
      {textureOptions.enabled && (
        <label className="settings-field">
          <span>Texture checkpoint</span>
          <input
            value={textureOptions.checkpoint ?? ''}
            onChange={updateTextureCheckpoint}
            placeholder="Stable Diffusion checkpoint or local path"
          />
        </label>
      )}
      <div className="prop-row">
        <span className="prop-label">Output</span>
        <span className="prop-value">GLB viewport asset</span>
      </div>
    </section>
  )
}

function ModelInspector({
  activeMode,
  selectedModel,
  installSession,
  onInstallModel,
  onOpenModelLibrary,
}: ModelInspectorProps) {
  const installLogRef = React.useRef<HTMLDivElement>(null)

  React.useEffect(() => {
    if (!installLogRef.current || !installSession) {
      return
    }
    installLogRef.current.scrollTop = installLogRef.current.scrollHeight
  }, [installSession?.logs, installSession?.status])

  if (!selectedModel) {
    return (
      <div className="model-inspector-empty">
        <div className="prop-section-title">Model</div>
        <div className="model-setup-title">No provider selected</div>
        <div className="model-setup-copy">Choose a text or image provider before generating.</div>
        <button className="model-inspector-btn primary" onClick={onOpenModelLibrary}>
          Open model browser
        </button>
      </div>
    )
  }

  const activeInstall = installSession?.model_id === selectedModel.id ? installSession : null
  const needsSetup = !selectedModel.generation_ready
  const canRefreshAssets = selectedModel.generation_ready && hasDownloadableAssets(selectedModel)

  return (
    <div className={`model-inspector ${needsSetup ? 'needs-setup' : ''}`}>
      <div className="prop-section-title">Model</div>
      <div className="model-card-head">
        <div>
          <div className="model-card-title">{selectedModel.name}</div>
          <div className="model-card-subtitle">{selectedModel.summary}</div>
        </div>
        <span className={`model-status-chip ${selectedModel.generation_ready ? 'ready' : 'pending'}`}>
          {modelStatusText(selectedModel)}
        </span>
      </div>

      {activeInstall && (
        <div className="model-install-progress">
          <div className="model-install-row">
            <span>{installStatusText(activeInstall)}</span>
            <span>
              {activeInstall.step_count > 0
                ? `${activeInstall.current_step} / ${activeInstall.step_count}`
                : 'Preparing'}
            </span>
          </div>
          <div className="model-install-active">{activeInstall.active_step ?? activeInstall.status_detail}</div>
          <div ref={installLogRef} className="model-install-log" role="log" aria-live="polite">
            {activeInstall.logs.length > 0
              ? activeInstall.logs.slice(-80).map((line, index) => (
                <div key={`${activeInstall.job_id}-${index}`} className="model-install-line">
                  {line}
                </div>
              ))
              : <div className="model-install-line muted">Waiting for download output...</div>}
          </div>
        </div>
      )}

      {needsSetup ? (
        <div className="model-setup-page">
          <div className="model-setup-title">
            {selectedModel.downloaded ? 'Provider setup required' : 'Download provider'}
          </div>
          <div className="model-setup-copy">{selectedModel.status_detail}</div>
          {selectedModel.python_status_detail && (
            <div className="model-python-note">{selectedModel.python_status_detail}</div>
          )}

          <button
            className="model-inspector-btn primary"
            onClick={() => onInstallModel(selectedModel.id, activeMode)}
            disabled={activeInstall?.status === 'running'}
          >
            {activeInstall?.status === 'running' ? 'Installing...' : 'Install / Download'}
          </button>
        </div>
      ) : (
        <div className="model-ready-page">
          <div className="prop-row">
            <span className="prop-label">VRAM</span>
            <span className="prop-value">{selectedModel.vram_hint ?? 'Model default'}</span>
          </div>
          <div className="prop-row">
            <span className="prop-label">Size</span>
            <span className="prop-value">{selectedModel.size_hint ?? 'Provider managed'}</span>
          </div>
          {selectedModel.docs_url && (
            <button
              className="model-inspector-btn"
              onClick={() => void window.velocityAPI.openExternal(selectedModel.docs_url!)}
            >
              Open docs
            </button>
          )}
          {canRefreshAssets && (
            <button
              className="model-inspector-btn primary"
              onClick={() => onInstallModel(selectedModel.id, activeMode)}
              disabled={activeInstall?.status === 'running'}
            >
              {activeInstall?.status === 'running' ? 'Downloading assets...' : 'Download / refresh assets'}
            </button>
          )}
        </div>
      )}
    </div>
  )
}

function GenerationTerminal({ selectedModel }: { selectedModel?: ModelCatalogItem | null }) {
  const terminalRef = React.useRef<HTMLDivElement>(null)
  const requestId = useGenerationStore((state) => state.requestId)
  const progress = useGenerationStore((state) => state.progress)
  const logs = useGenerationStore((state) => state.logs)

  React.useEffect(() => {
    if (terminalRef.current) {
      terminalRef.current.scrollTop = terminalRef.current.scrollHeight
    }
  }, [logs, progress])

  return (
    <aside className="properties-panel">
      <div className="properties-header">
        <div className="properties-title">Generation terminal</div>
        <div className="properties-subtitle">{selectedModel?.name ?? 'Selected provider'} is running</div>
      </div>
      <div className="properties-content generation-terminal-content">
        <div className="generation-terminal-meta">
          <span>{requestId ? requestId.slice(0, 8) : 'request'}</span>
          <span>{progress ?? 'Working...'}</span>
        </div>
        <div ref={terminalRef} className="generation-terminal-log" role="log" aria-live="polite">
          {logs.length > 0
            ? logs.map((line, index) => (
              <div key={`${index}-${line.slice(0, 12)}`} className="generation-terminal-line">
                {line}
              </div>
            ))
            : <div className="generation-terminal-line muted">Waiting for backend output...</div>}
        </div>
      </div>
    </aside>
  )
}

export default function PropertiesPanel({
  selectedObject,
  activeMode,
  selectedModel,
  inputImage,
  textureOptions,
  installSession,
  onTextureOptionsChange,
  onInstallModel,
  onOpenModelLibrary,
  onUpdateObjectTransform,
}: Props) {
  const currentModelPath = useGenerationStore((state) => state.currentModelPath)
  const generationStatus = useGenerationStore((state) => state.status)
  const currentModelLabel = currentModelPath ? currentModelPath.split(/[\\/]/).pop() : null

  if (generationStatus === 'generating') {
    return <GenerationTerminal selectedModel={selectedModel} />
  }

  return (
    <aside className="properties-panel">
      <div className="properties-header">
        <div className="properties-title">Inspector</div>
        <div className="properties-subtitle">Provider, inputs, and object details</div>
      </div>
      <div className="properties-content">
        <ModelInspector
          activeMode={activeMode}
          selectedModel={selectedModel}
          installSession={installSession}
          onInstallModel={onInstallModel}
          onOpenModelLibrary={onOpenModelLibrary}
        />
        <InputsSection
          activeMode={activeMode}
          selectedModel={selectedModel}
          inputImage={inputImage}
          textureOptions={textureOptions}
          onTextureOptionsChange={onTextureOptionsChange}
        />
        {!selectedObject ? (
          <>
            <div className="prop-section-title">Viewport</div>
            <div className="prop-row">
              <span className="prop-label">Model</span>
              <span className="prop-value">{currentModelLabel ?? 'No model loaded'}</span>
            </div>
            <div className="properties-empty">Select a mesh in the viewport to inspect transforms and topology.</div>
          </>
        ) : (
          <>
            <div className="prop-section-title">Object</div>
            <div className="prop-row">
              <span className="prop-label">Name</span>
              <span className="prop-value">{selectedObject.name}</span>
            </div>
            <div className="prop-row">
              <span className="prop-label">Vertices</span>
              <span className="prop-value">{selectedObject.vertexCount.toLocaleString()}</span>
            </div>
            <div className="prop-row">
              <span className="prop-label">Faces</span>
              <span className="prop-value">{selectedObject.faceCount.toLocaleString()}</span>
            </div>
            <div className="prop-section-title">Transform</div>
            {onUpdateObjectTransform ? (
              <div className="transform-editor">
                <EditableVec3Row
                  label="Location"
                  value={selectedObject.position}
                  onChange={(position) => onUpdateObjectTransform(selectedObject.id, { position })}
                />
                <EditableVec3Row
                  label="Rotation"
                  value={selectedObject.rotation}
                  mode="rotation"
                  onChange={(rotation) => onUpdateObjectTransform(selectedObject.id, { rotation })}
                />
                <EditableVec3Row
                  label="Scale"
                  value={selectedObject.scale}
                  onChange={(scale) => onUpdateObjectTransform(selectedObject.id, { scale })}
                />
                <button
                  className="model-inspector-btn"
                  onClick={() => onUpdateObjectTransform(selectedObject.id, {
                    position: { x: 0, y: 0, z: 0 },
                    rotation: { x: 0, y: 0, z: 0 },
                    scale: { x: 1, y: 1, z: 1 }
                  })}
                >
                  Reset transform
                </button>
              </div>
            ) : (
              <>
                <Vec3Row label="Location" value={selectedObject.position} />
                <Vec3Row label="Rotation" value={selectedObject.rotation} />
                <Vec3Row label="Scale" value={selectedObject.scale} />
              </>
            )}
          </>
        )}
      </div>
    </aside>
  )
}

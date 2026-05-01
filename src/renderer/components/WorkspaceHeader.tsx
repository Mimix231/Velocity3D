import React from 'react'

import type { GenerationMode, ModelCatalogItem } from '../api/modelApi'
import './WorkspaceHeader.css'

interface Props {
  backendStatus: 'starting' | 'ready' | 'error' | 'stopped'
  textModel: ModelCatalogItem | null
  imageModel: ModelCatalogItem | null
  activeMode: GenerationMode
  onOpenModelLibrary: () => void
}

function StatusPill({ status }: { status: Props['backendStatus'] }) {
  const label = status === 'ready'
    ? 'Backend ready'
    : status === 'error'
      ? 'Backend error'
      : status === 'stopped'
        ? 'Backend stopped'
        : 'Backend starting'

  return (
    <span className={`workspace-status-pill ${status}`}>
      <span className="workspace-status-dot" />
      {label}
    </span>
  )
}

function ModelSummary({ label, model }: { label: string; model: ModelCatalogItem | null }) {
  const badgeLabel = model
    ? (model.generation_ready ? 'Ready' : 'Install')
    : null

  return (
    <div className="workspace-model-summary">
      <span className="workspace-model-label">{label}</span>
      <div className="workspace-model-value-row">
        <span className="workspace-model-name">{model?.name ?? 'Not selected'}</span>
        {model && badgeLabel && (
          <span className={`workspace-model-badge ${model.generation_ready ? 'ready' : 'pending'}`}>
            {badgeLabel}
          </span>
        )}
      </div>
    </div>
  )
}

export default function WorkspaceHeader({
  backendStatus,
  textModel,
  imageModel,
  activeMode,
  onOpenModelLibrary
}: Props) {
  return (
    <header className="workspace-header">
      <div className="workspace-heading">
        <StatusPill status={backendStatus} />
        <div className="workspace-breadcrumbs">
          <span className="workspace-breadcrumb-chip">Editor</span>
          <span className="workspace-breadcrumb-separator">/</span>
          <span className="workspace-breadcrumb-copy">Generation workspace</span>
        </div>
        <span className="workspace-mode-chip">
          {activeMode === 'image' ? 'Image-to-3D' : 'Text-to-3D'}
        </span>
      </div>

      <div className="workspace-actions">
        <ModelSummary label="Text default" model={textModel} />
        <ModelSummary label="Image default" model={imageModel} />
        <button className="workspace-library-button" onClick={onOpenModelLibrary}>
          Open model browser
        </button>
      </div>
    </header>
  )
}

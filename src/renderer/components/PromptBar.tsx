import React, { useEffect, useMemo, useRef, useState } from 'react'

import type { GenerationMode, ModelCatalogItem, ModelInstallStatusResponse } from '../api/modelApi'
import { validateImageFile } from '../validation'
import { useGenerationStore } from '../stores/generationStore'
import type { InputImagePreview } from './PropertiesPanel'
import { removeImageBackground } from '../utils/backgroundRemoval'
import ExportButton from './ExportButton'
import './PromptBar.css'

type InstallSession = ModelInstallStatusResponse & { mode: GenerationMode }

interface Props {
  mode: GenerationMode
  onModeChange: (mode: GenerationMode) => void
  onSubmitText?: (prompt: string) => void
  onSubmitImage?: (file: File, prompt?: string, imageBase64?: string) => void
  onCancel?: () => void
  isGenerating?: boolean
  textModels: ModelCatalogItem[]
  imageModels: ModelCatalogItem[]
  selectedTextModelId: string | null
  selectedImageModelId: string | null
  onSelectModel: (mode: GenerationMode, modelId: string) => void
  onInstallModel: (modelId: string, mode: GenerationMode) => void
  installSession?: InstallSession | null
  onDismissInstall?: () => void
  onImagePreviewChange?: (preview: InputImagePreview | null) => void
}

function installActionLabel(model: ModelCatalogItem | null): string | null {
  if (!model || model.generation_ready) {
    return null
  }
  return 'Install / Download'
}

function installStatusLabel(session: InstallSession): string {
  switch (session.status) {
    case 'complete':
      return 'Complete'
    case 'manual_required':
      return 'Manual follow-up'
    case 'error':
      return 'Install failed'
    default:
      return 'Installing'
  }
}

export default function PromptBar({
  mode,
  onModeChange,
  onSubmitText,
  onSubmitImage,
  onCancel,
  isGenerating = false,
  textModels,
  imageModels,
  selectedTextModelId,
  selectedImageModelId,
  onSelectModel,
  onInstallModel,
  installSession = null,
  onDismissInstall,
  onImagePreviewChange
}: Props) {
  const [prompt, setPrompt] = useState('')
  const [imageFile, setImageFile] = useState<File | null>(null)
  const [imagePreviewUrl, setImagePreviewUrl] = useState<string | null>(null)
  const [backgroundRemovedUrl, setBackgroundRemovedUrl] = useState<string | null>(null)
  const [backgroundRemovedBase64, setBackgroundRemovedBase64] = useState<string | null>(null)
  const [backgroundRemovalBusy, setBackgroundRemovalBusy] = useState(false)
  const [validationError, setValidationError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const textareaRef = useRef<HTMLTextAreaElement>(null)
  const terminalRef = useRef<HTMLDivElement>(null)
  const backgroundRemovalRunRef = useRef(0)

  const { status, errorMessage, dismiss } = useGenerationStore()

  const activeModels = mode === 'image' ? imageModels : textModels
  const selectedModelId = mode === 'image' ? selectedImageModelId : selectedTextModelId
  const selectedModel = useMemo(
    () => activeModels.find((model) => model.id === selectedModelId) ?? null,
    [activeModels, selectedModelId]
  )

  useEffect(() => {
    return () => {
      if (imagePreviewUrl) {
        URL.revokeObjectURL(imagePreviewUrl)
      }
    }
  }, [imagePreviewUrl])

  useEffect(() => {
    return () => {
      onImagePreviewChange?.(null)
    }
  }, [onImagePreviewChange])

  useEffect(() => {
    if (!installSession || !terminalRef.current) {
      return
    }
    terminalRef.current.scrollTop = terminalRef.current.scrollHeight
  }, [installSession?.logs, installSession?.status])

  async function handleFileChange(event: React.ChangeEvent<HTMLInputElement>) {
    const file = event.target.files?.[0]
    if (!file) return

    const result = validateImageFile(file.name)
    if (!result.valid) {
      setValidationError(result.error ?? 'Unsupported image file.')
      setImageFile(null)
      onImagePreviewChange?.(null)
      setBackgroundRemovedUrl(null)
      setBackgroundRemovedBase64(null)
      if (imagePreviewUrl) {
        URL.revokeObjectURL(imagePreviewUrl)
      }
      setImagePreviewUrl(null)
      return
    }

    setValidationError(null)
    setImageFile(file)
    onModeChange('image')
    if (imagePreviewUrl) {
      URL.revokeObjectURL(imagePreviewUrl)
    }
    const nextPreviewUrl = URL.createObjectURL(file)
    const removalRun = backgroundRemovalRunRef.current + 1
    backgroundRemovalRunRef.current = removalRun
    setImagePreviewUrl(nextPreviewUrl)
    onImagePreviewChange?.({
      url: nextPreviewUrl,
      backgroundRemovedUrl: null,
      name: file.name,
      size: file.size
    })
    setBackgroundRemovalBusy(true)
    setBackgroundRemovedUrl(null)
    setBackgroundRemovedBase64(null)

    try {
      const cutout = await removeImageBackground(nextPreviewUrl)
      if (backgroundRemovalRunRef.current !== removalRun) {
        return
      }
      setBackgroundRemovedUrl(cutout.dataUrl)
      setBackgroundRemovedBase64(cutout.base64)
      onImagePreviewChange?.({
        url: nextPreviewUrl,
        backgroundRemovedUrl: cutout.dataUrl,
        name: file.name,
        size: file.size
      })
    } catch (err: any) {
      if (backgroundRemovalRunRef.current !== removalRun) {
        return
      }
      setValidationError(err.message ?? 'Background removal failed.')
      onImagePreviewChange?.({
        url: nextPreviewUrl,
        backgroundRemovedUrl: null,
        name: file.name,
        size: file.size
      })
    } finally {
      if (backgroundRemovalRunRef.current === removalRun) {
        setBackgroundRemovalBusy(false)
      }
    }
  }

  function handleSubmit() {
    if (isGenerating) return

    if (!selectedModel) {
      setValidationError('Select a model before generating.')
      return
    }

    if (!selectedModel.generation_ready) {
      setValidationError(`${selectedModel.name} is not ready yet. Run Install / Download first.`)
      return
    }

    if (mode === 'image') {
      if (!imageFile) {
        setValidationError('Upload an image before running image-to-3D generation.')
        return
      }
      if (backgroundRemovalBusy || !backgroundRemovedBase64) {
        setValidationError('Wait for background removal before generating.')
        return
      }
      onSubmitImage?.(imageFile, prompt.trim() || undefined, backgroundRemovedBase64)
      setValidationError(null)
      return
    }

    if (!prompt.trim()) {
      setValidationError('Enter a prompt to run text-to-3D generation.')
      return
    }

    onSubmitText?.(prompt.trim())
    setValidationError(null)
  }

  function handleKeyDown(event: React.KeyboardEvent) {
    if (event.key === 'Enter' && !event.shiftKey && !isGenerating) {
      event.preventDefault()
      handleSubmit()
    }
  }

  function clearImage() {
    setImageFile(null)
    onImagePreviewChange?.(null)
    backgroundRemovalRunRef.current += 1
    setBackgroundRemovedUrl(null)
    setBackgroundRemovedBase64(null)
    setBackgroundRemovalBusy(false)
    if (imagePreviewUrl) {
      URL.revokeObjectURL(imagePreviewUrl)
    }
    setImagePreviewUrl(null)
    if (fileInputRef.current) {
      fileInputRef.current.value = ''
    }
  }

  function handleTextareaInput(event: React.ChangeEvent<HTMLTextAreaElement>) {
    setPrompt(event.target.value)
    const element = event.target
    element.style.height = 'auto'
    element.style.height = `${Math.min(element.scrollHeight, 120)}px`
  }

  const showError = status === 'error' && errorMessage
  const modeLabel = mode === 'image' ? 'Image model' : 'Text model'
  const submitLabel = mode === 'image'
    ? backgroundRemovalBusy
      ? 'Preparing image...'
      : 'Generate from image'
    : 'Generate from text'
  const pendingInstallLabel = installActionLabel(selectedModel)
  const submitDisabled = (
    isGenerating ||
    !selectedModel ||
    !selectedModel.generation_ready ||
    (mode === 'image' && (!imageFile || backgroundRemovalBusy || !backgroundRemovedBase64)) ||
    (mode === 'text' && !prompt.trim())
  )
  const promptBarClassName = [
    'prompt-bar',
    mode === 'image' ? 'prompt-bar-image' : 'prompt-bar-text',
    imageFile ? 'has-image' : ''
  ].filter(Boolean).join(' ')

  if (installSession) {
    const canLeaveTerminal = installSession.status !== 'running'
    const footerCopy = installSession.status === 'complete'
      ? 'Install finished. Returning to the generator view...'
      : installSession.status === 'running'
        ? 'Install / Download is running in the backend.'
        : installSession.status_detail

    return (
      <div ref={terminalRef} className="prompt-bar prompt-bar-terminal">
        <div className="prompt-terminal-header">
          <div className="prompt-terminal-heading">
            <span className="prompt-terminal-title">Install / Download</span>
            <span className="prompt-terminal-model">{installSession.model_name}</span>
          </div>

          <div className="prompt-terminal-controls">
            <span className={`prompt-terminal-status ${installSession.status}`}>
              {installStatusLabel(installSession)}
            </span>
            <span className="prompt-terminal-step">
              {installSession.step_count > 0
                ? `Step ${Math.max(installSession.current_step, 1)} / ${installSession.step_count}`
                : 'Preparing'}
            </span>
            {canLeaveTerminal && onDismissInstall && (
              <button className="prompt-terminal-close" onClick={onDismissInstall}>
                Back to generator
              </button>
            )}
          </div>
        </div>

        <div className="prompt-terminal-active-step">
          {installSession.active_step ?? installSession.status_detail}
        </div>

        <div className="prompt-terminal-log" role="log" aria-live="polite">
          {installSession.logs.length > 0
            ? installSession.logs.map((line, index) => (
              <div key={`${installSession.job_id}-${index}`} className="prompt-terminal-line">
                {line}
              </div>
            ))
            : <div className="prompt-terminal-line muted">Waiting for install output...</div>}
        </div>

        <div className="prompt-terminal-footer">
          <span>{footerCopy}</span>
          {installSession.error && <span className="prompt-terminal-error">{installSession.error}</span>}
        </div>
      </div>
    )
  }

  return (
    <div className={promptBarClassName}>
      {showError && (
        <div className="prompt-error-banner">
          <span>{errorMessage}</span>
          <button onClick={dismiss}>Dismiss</button>
        </div>
      )}

      <div className="prompt-toolbar">
        <div className="prompt-mode-strip" role="tablist" aria-label="Generation mode">
          <button
            className={`prompt-mode-pill ${mode === 'text' ? 'active' : ''}`}
            onClick={() => onModeChange('text')}
          >
            Text to 3D
          </button>
          <button
            className={`prompt-mode-pill ${mode === 'image' ? 'active' : ''}`}
            onClick={() => onModeChange('image')}
          >
            Image to 3D
          </button>
        </div>

        <label className="prompt-model-picker">
          <span className="prompt-model-label">{modeLabel}</span>
          <select
            className="prompt-model-select"
            value={selectedModelId ?? ''}
            onChange={(event) => onSelectModel(mode, event.target.value)}
            disabled={isGenerating}
          >
            {activeModels.map((model) => (
              <option key={model.id} value={model.id}>
                {model.name}{model.generation_ready ? '' : ' - install needed'}
              </option>
            ))}
          </select>
        </label>

        {selectedModel && (
          selectedModel.generation_ready
            ? (
              <div className="prompt-model-status ready">
                Ready
              </div>
            )
            : (
              <button
                className="prompt-model-install-btn"
                onClick={() => onInstallModel(selectedModel.id, mode)}
                disabled={isGenerating}
              >
                {pendingInstallLabel}
              </button>
            )
        )}
        {selectedModel && !selectedModel.generation_ready && (
          <div className="prompt-model-status pending">
            {selectedModel.downloaded ? 'Downloaded' : 'Not installed'}
          </div>
        )}
      </div>

      {selectedModel && (
        <div className="prompt-model-hint">
          <span className="prompt-model-name">{selectedModel.name}</span>
          <span>{selectedModel.summary}</span>
          {selectedModel.vram_hint && <span>{selectedModel.vram_hint}</span>}
        </div>
      )}

      {mode === 'image' && !imageFile && (
        <div className="prompt-drop-hint">
          Image mode expects one reference image. Add a file first, then use the prompt for optional style or detail guidance.
        </div>
      )}

      {imageFile && imagePreviewUrl && (
        <div className="prompt-image-preview">
          <img src={imagePreviewUrl} alt="Selected input preview" className="prompt-image-thumb" />
          <div className="prompt-image-meta">
            <span className="prompt-image-name">{imageFile.name}</span>
            <span className="prompt-image-size">
              {backgroundRemovalBusy ? 'Removing background...' : `${(imageFile.size / 1024).toFixed(0)} KB cutout ready`}
            </span>
          </div>
          <button className="prompt-image-clear" aria-label="Remove image" onClick={clearImage} title="Remove image">x</button>
        </div>
      )}

      {validationError && <div className="prompt-validation-error">{validationError}</div>}

      <div className="prompt-input-row">
        <button
          className="prompt-upload-btn"
          aria-label="Upload image"
          title="Upload image for image-to-3D generation"
          onClick={() => fileInputRef.current?.click()}
          disabled={isGenerating}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="3" width="18" height="18" rx="2" />
            <circle cx="8.5" cy="8.5" r="1.5" />
            <polyline points="21 15 16 10 5 21" />
          </svg>
        </button>
        <input
          ref={fileInputRef}
          type="file"
          accept=".jpg,.jpeg,.png,.webp,.bmp"
          style={{ display: 'none' }}
          onChange={handleFileChange}
        />
        <textarea
          ref={textareaRef}
          className="prompt-textarea"
          placeholder={
            mode === 'image'
              ? 'Optional: describe materials, silhouette, era, or cleanup changes after the image is attached...'
              : 'Describe the 3D model you want to generate...'
          }
          value={prompt}
          onChange={handleTextareaInput}
          onKeyDown={handleKeyDown}
          disabled={isGenerating}
          rows={1}
        />
        <ExportButton />
        {isGenerating ? (
          <button className="prompt-cancel-btn" onClick={onCancel}>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="currentColor">
              <rect x="4" y="4" width="16" height="16" rx="2" />
            </svg>
            Cancel
          </button>
        ) : (
          <button className="prompt-submit-btn" onClick={handleSubmit} disabled={submitDisabled}>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
              <polygon points="5 3 19 12 5 21 5 3" />
            </svg>
            {submitLabel}
          </button>
        )}
      </div>

      {isGenerating && (
        <div className="prompt-progress">
          <div className="prompt-progress-bar" />
          <span>Generating with {selectedModel?.name ?? 'selected model'}...</span>
        </div>
      )}
    </div>
  )
}

import React, { useCallback, useEffect, useMemo, useState } from 'react'
import { v4 as uuidv4 } from 'uuid'

import './App.css'
import type {
  GenerationMetadata,
  GenerationResponse,
  LatestGeneratedOutput,
  PipelineOptions,
  TextureOptions
} from './api/generationApi'
import {
  fetchModelInstallStatus,
  startModelInstall,
  type GenerationMode,
  type ModelInstallStatusResponse
} from './api/modelApi'
import { cancelGeneration, fetchLatestGeneratedOutput, submitGeneration } from './api/generationApi'
import ModelDownloader from './components/ModelDownloader'
import PromptBar from './components/PromptBar'
import PropertiesPanel from './components/PropertiesPanel'
import Sidebar from './components/Sidebar'
import StartupScreen from './components/StartupScreen'
import SetupWizard from './components/SetupWizard'
import ViewportContainer, { type ViewportTransformCommand } from './components/ViewportContainer'
import WorkspaceHeader from './components/WorkspaceHeader'
import type { InputImagePreview, ObjectProperties } from './components/PropertiesPanel'
import { useGenerationStore } from './stores/generationStore'
import { useHistoryStore } from './stores/historyStore'
import {
  getModelsForMode,
  getSelectedModel,
  useModelCatalogStore
} from './stores/modelCatalogStore'
import { applyThemeToCss, useThemeStore } from './stores/themeStore'

type Phase = 'loading' | 'setup' | 'startup' | 'app'
type InstallSession = ModelInstallStatusResponse & { mode: GenerationMode }
type GenerationRaceResult =
  | { source: 'response'; response: GenerationResponse }
  | { source: 'fallback'; output: LatestGeneratedOutput }
const TEXTURE_SETTINGS_STORAGE_KEY = 'velocity3d-texture-settings'
const PIPELINE_SETTINGS_STORAGE_KEY = 'velocity3d-pipeline-settings'
const DEFAULT_TEXTURE_CHECKPOINT = 'stabilityai/stable-diffusion-xl-base-1.0'
const TEXTURED_OUTPUT_WAIT_LIMIT_MS = 30 * 60 * 1000
const VALID_PIPELINE_PRESETS = new Set<PipelineOptions['preset']>([
  'preview',
  'balanced',
  'building_module',
  'game_asset',
  'production',
])

function loadTextureSettings(): TextureOptions {
  try {
    const raw = localStorage.getItem(TEXTURE_SETTINGS_STORAGE_KEY)
    if (!raw) {
      return { enabled: true, checkpoint: DEFAULT_TEXTURE_CHECKPOINT }
    }
    const parsed = JSON.parse(raw) as Partial<TextureOptions>
    return {
      enabled: parsed.enabled ?? false,
      checkpoint: parsed.checkpoint || DEFAULT_TEXTURE_CHECKPOINT,
    }
  } catch {
    return { enabled: true, checkpoint: DEFAULT_TEXTURE_CHECKPOINT }
  }
}

function loadPipelineSettings(): PipelineOptions {
  try {
    const raw = localStorage.getItem(PIPELINE_SETTINGS_STORAGE_KEY)
    if (!raw) {
      return { preset: 'building_module' }
    }
    const parsed = JSON.parse(raw) as Partial<PipelineOptions>
    const preset = VALID_PIPELINE_PRESETS.has(parsed.preset as PipelineOptions['preset'])
      ? parsed.preset as PipelineOptions['preset']
      : 'building_module'
    return {
      preset,
      target_face_count: Number.isFinite(parsed.target_face_count) ? parsed.target_face_count : undefined,
      texture_size: Number.isFinite(parsed.texture_size) ? parsed.texture_size : undefined,
    }
  } catch {
    return { preset: 'building_module' }
  }
}

export default function App() {
  const [phase, setPhase] = useState<Phase>('loading')
  const [selectedObject, setSelectedObject] = useState<ObjectProperties | null>(null)
  const [showModelLibrary, setShowModelLibrary] = useState(false)
  const [backendStatus, setBackendStatus] = useState<'starting' | 'ready' | 'error' | 'stopped'>('starting')
  const [generationMode, setGenerationMode] = useState<GenerationMode>('text')
  const [installSession, setInstallSession] = useState<InstallSession | null>(null)
  const [textureOptions, setTextureOptions] = useState<TextureOptions>(() => loadTextureSettings())
  const [pipelineOptions, setPipelineOptions] = useState<PipelineOptions>(() => loadPipelineSettings())
  const [inputImage, setInputImage] = useState<InputImagePreview | null>(null)
  const [viewportTransformCommand, setViewportTransformCommand] = useState<ViewportTransformCommand | null>(null)

  const { status, requestId, submit, success, setError, cancel } = useGenerationStore()
  const { theme } = useThemeStore()
  const refreshCatalog = useModelCatalogStore((state) => state.refresh)
  const models = useModelCatalogStore((state) => state.models)
  const modelCatalogError = useModelCatalogStore((state) => state.error)
  const textModelId = useModelCatalogStore((state) => state.textModelId)
  const imageModelId = useModelCatalogStore((state) => state.imageModelId)
  const setSelectedModel = useModelCatalogStore((state) => state.setSelectedModel)

  const selectedTextModel = useMemo(() => getSelectedModel(models, textModelId), [models, textModelId])
  const selectedImageModel = useMemo(() => getSelectedModel(models, imageModelId), [models, imageModelId])
  const activeSelectedModel = generationMode === 'image' ? selectedImageModel : selectedTextModel
  const textModels = useMemo(() => getModelsForMode(models, 'text'), [models])
  const imageModels = useMemo(() => getModelsForMode(models, 'image'), [models])

  useEffect(() => {
    applyThemeToCss(theme)
  }, [theme])

  useEffect(() => {
    try {
      localStorage.setItem(TEXTURE_SETTINGS_STORAGE_KEY, JSON.stringify(textureOptions))
    } catch {
      // Ignore persistence failures.
    }
  }, [textureOptions])

  useEffect(() => {
    try {
      localStorage.setItem(PIPELINE_SETTINGS_STORAGE_KEY, JSON.stringify(pipelineOptions))
    } catch {
      // Ignore persistence failures.
    }
  }, [pipelineOptions])

  useEffect(() => {
    window.velocityAPI.getConfig().then((cfg) => {
      if (cfg.setupComplete) {
        setPhase('startup')
      } else {
        setPhase('setup')
      }
    }).catch(() => {
      setPhase('setup')
    })
  }, [])

  useEffect(() => {
    const unsubscribe = window.velocityAPI.onBackendStatus((event) => {
      setBackendStatus(event.status)
    })
    return unsubscribe
  }, [])

  useEffect(() => {
    const unsubscribe = window.velocityAPI.onBackendLog((line) => {
      const generation = useGenerationStore.getState()
      if (generation.status === 'generating') {
        if (/volume\s+decoding|creating blender|gltf import|export|texture|stable diffusion/i.test(line)) {
          generation.setProgress(line.trim().slice(0, 180))
        } else {
          generation.appendLog(line)
        }
      }
    })
    return unsubscribe
  }, [])

  useEffect(() => {
    if (phase !== 'app' || backendStatus !== 'ready') {
      return
    }

    const timer = window.setTimeout(() => {
      void refreshCatalog()
    }, 180)

    return () => window.clearTimeout(timer)
  }, [phase, backendStatus, refreshCatalog])

  useEffect(() => {
    if (!installSession || installSession.status !== 'running') {
      return
    }

    let cancelled = false
    let timer: number | null = null
    let pollFailures = 0

    const poll = async () => {
      try {
        const next = await fetchModelInstallStatus(installSession.job_id)
        pollFailures = 0
        if (cancelled) {
          return
        }

        setInstallSession((current) => (
          current && current.job_id === next.job_id
            ? { ...current, ...next }
            : current
        ))

        if (next.status === 'running') {
          timer = window.setTimeout(() => {
            void poll()
          }, 900)
          return
        }

        void refreshCatalog()
      } catch (err: any) {
        if (cancelled) {
          return
        }
        pollFailures += 1
        const message = err.message ?? 'Failed to poll install status.'
        setInstallSession((current) => {
          if (!current || current.job_id !== installSession.job_id) {
            return current
          }

          if (current.status === 'running' && pollFailures < 6) {
            const logLine = `! Status poll failed (${pollFailures}/6): ${message}. Retrying...`
            return {
              ...current,
              logs: current.logs.includes(logLine) ? current.logs : [...current.logs, logLine]
            }
          }

          return {
            ...current,
            status: 'error',
            active_step: null,
            error: message,
            status_detail: message,
            logs: current.logs.includes(`! ${message}`) ? current.logs : [...current.logs, `! ${message}`]
          }
        })

        if (pollFailures < 6) {
          timer = window.setTimeout(() => {
            void poll()
          }, Math.min(2000 * pollFailures, 10000))
        }
      }
    }

    void poll()

    return () => {
      cancelled = true
      if (timer !== null) {
        window.clearTimeout(timer)
      }
    }
  }, [installSession?.job_id, installSession?.status, refreshCatalog])

  useEffect(() => {
    if (!installSession || installSession.status !== 'complete') {
      return
    }

    const timer = window.setTimeout(() => {
      setInstallSession((current) => (
        current && current.job_id === installSession.job_id ? null : current
      ))
    }, 1400)

    return () => window.clearTimeout(timer)
  }, [installSession?.job_id, installSession?.status])

  function handleSetupComplete(_venvPath: string) {
    setPhase('startup')
  }

  function handleBackendReady() {
    setBackendStatus('ready')
    setPhase('app')
  }

  const handleInputImagePreviewChange = useCallback((preview: InputImagePreview | null) => {
    setInputImage(preview)
  }, [])

  const handleUpdateObjectTransform = useCallback((
    objectId: string,
    transform: Partial<Pick<ObjectProperties, 'position' | 'rotation' | 'scale'>>
  ) => {
    setSelectedObject((current) => {
      if (!current || current.id !== objectId) return current
      return {
        ...current,
        position: transform.position ?? current.position,
        rotation: transform.rotation ?? current.rotation,
        scale: transform.scale ?? current.scale,
      }
    })
    setViewportTransformCommand({
      id: Date.now(),
      objectId,
      transform
    })
  }, [])

  async function handleSubmitText(prompt: string) {
    if (!selectedTextModel) {
      setError('No text model is selected.')
      return
    }
    if (!selectedTextModel.generation_ready) {
      setError(`${selectedTextModel.name} is not ready yet. Run Install / Download first.`)
      return
    }

    const reqId = uuidv4()
    const startedAtMs = Date.now()
    submit(reqId)
    const responsePromise = submitGeneration({
      type: 'text',
      prompt,
      model_id: selectedTextModel.id,
      texture_options: textureOptions.enabled ? textureOptions : undefined,
      pipeline_options: pipelineOptions,
      request_id: reqId
    })
    const outputPoll = createGeneratedOutputPoll(startedAtMs, Boolean(textureOptions.enabled))

    try {
      const result = await Promise.race<GenerationRaceResult>([
        responsePromise.then((response) => ({ source: 'response', response }) as const),
        outputPoll.promise.then((output) => ({ source: 'fallback', output }) as const),
      ])
      outputPoll.stop()

      const res = result.source === 'response'
        ? result.response
        : buildFallbackGenerationResponse({
          requestId: reqId,
          output: result.output,
          selectedModel: selectedTextModel,
          startedAtMs,
          textureOptions,
          pipelineOptions,
        })

      success(res.model_path)
      useHistoryStore.getState().addEntry({
        id: crypto.randomUUID(),
        project_id: useHistoryStore.getState().activeProjectId,
        type: 'text',
        model_id: res.metadata.model_id,
        model_name: res.metadata.model_name,
        prompt,
        image_filename: null,
        model_path: res.model_path,
        timestamp: new Date().toISOString(),
        metadata: res.metadata
      })

      if (result.source === 'fallback') {
        settleLateGenerationResponse(reqId, responsePromise)
      }
    } catch (err: any) {
      outputPoll.stop()
      setError(err.message ?? 'Generation failed')
    }
  }

  async function handleSubmitImage(file: File, prompt?: string, imageBase64?: string) {
    if (!selectedImageModel) {
      setError('No image model is selected.')
      return
    }
    if (!selectedImageModel.generation_ready) {
      setError(`${selectedImageModel.name} is not ready yet. Run Install / Download first.`)
      return
    }

    const reqId = uuidv4()
    const startedAtMs = Date.now()
    submit(reqId)
    let outputPoll: ReturnType<typeof createGeneratedOutputPoll> | null = null

    try {
      const originalImageBase64 = await fileToBase64(file)
      const providerImageBase64 = imageBase64 ?? originalImageBase64
      const responsePromise = submitGeneration({
        type: 'image',
        image_base64: providerImageBase64,
        reference_image_base64: providerImageBase64,
        prompt,
        model_id: selectedImageModel.id,
        texture_options: textureOptions.enabled ? textureOptions : undefined,
        pipeline_options: pipelineOptions,
        request_id: reqId
      })
      outputPoll = createGeneratedOutputPoll(startedAtMs, Boolean(textureOptions.enabled))

      const result = await Promise.race<GenerationRaceResult>([
        responsePromise.then((response) => ({ source: 'response', response }) as const),
        outputPoll.promise.then((output) => ({ source: 'fallback', output }) as const),
      ])
      outputPoll.stop()

      const res = result.source === 'response'
        ? result.response
        : buildFallbackGenerationResponse({
          requestId: reqId,
          output: result.output,
          selectedModel: selectedImageModel,
          startedAtMs,
          textureOptions,
          pipelineOptions,
        })

      success(res.model_path)
      useHistoryStore.getState().addEntry({
        id: crypto.randomUUID(),
        project_id: useHistoryStore.getState().activeProjectId,
        type: 'image',
        model_id: res.metadata.model_id,
        model_name: res.metadata.model_name,
        prompt: prompt ?? null,
        image_filename: file.name,
        model_path: res.model_path,
        timestamp: new Date().toISOString(),
        metadata: res.metadata
      })

      if (result.source === 'fallback') {
        settleLateGenerationResponse(reqId, responsePromise)
      }
    } catch (err: any) {
      outputPoll?.stop()
      setError(err.message ?? 'Generation failed')
    }
  }

  async function handleCancel() {
    if (requestId) {
      cancel()
      await cancelGeneration(requestId).catch(() => {})
    }
  }

  async function handleStartInstall(modelId: string, mode: GenerationMode) {
    const targetModel = models.find((model) => model.id === modelId)
    if (!targetModel) {
      setError('The selected model is no longer available in the catalog.')
      return
    }

    setGenerationMode(mode)
    setSelectedModel(mode, modelId)
    setShowModelLibrary(false)

    try {
      const started = await startModelInstall(modelId)
      setInstallSession({
        job_id: started.job_id,
        model_id: started.model_id,
        model_name: targetModel.name,
        mode,
        status: 'running',
        current_step: 0,
        step_count: 0,
        active_step: 'Preparing install plan...',
        logs: [`Starting Install / Download for ${targetModel.name}...`],
        status_detail: 'Preparing install plan...',
        error: null
      })
    } catch (err: any) {
      const message = err.message ?? 'Failed to start install.'
      setInstallSession({
        job_id: crypto.randomUUID(),
        model_id: modelId,
        model_name: targetModel.name,
        mode,
        status: 'error',
        current_step: 0,
        step_count: 0,
        active_step: null,
        logs: [`! ${message}`],
        status_detail: message,
        error: message
      })
    }
  }

  if (phase === 'loading') {
    return (
      <div className="app-loading-screen">
        <div className="app-loading-copy">Loading Velocity3D...</div>
      </div>
    )
  }

  if (phase === 'setup') {
    return <SetupWizard onComplete={handleSetupComplete} />
  }

  if (phase === 'startup') {
    return <StartupScreen onReady={handleBackendReady} onError={() => {}} />
  }

  return (
    <>
      <div className="app-shell">
        <div className="app-menubar">
          <nav className="app-menubar-nav" aria-label="Application menu">
            {['File', 'Edit', 'View', 'Generate', 'Window', 'Help'].map((item) => (
              <button key={item} className="app-menubar-link">
                {item}
              </button>
            ))}
          </nav>

          <div className="app-menubar-workspace">
            <span className="app-menubar-tab active">Generate</span>
            <span className="app-menubar-tab">Viewport</span>
            <span className="app-menubar-tab">Inspector</span>
          </div>

          <div className="app-window-controls" aria-label="Window controls">
            <button className="app-window-btn" aria-label="Minimize" onClick={() => void window.velocityAPI.windowMinimize()}>
              <svg width="11" height="11" viewBox="0 0 12 12" aria-hidden="true">
                <path d="M2 6.5h8" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
              </svg>
            </button>
            <button className="app-window-btn" aria-label="Maximize" onClick={() => void window.velocityAPI.windowToggleMaximize()}>
              <svg width="11" height="11" viewBox="0 0 12 12" aria-hidden="true">
                <rect x="2.5" y="2.5" width="7" height="7" rx="1.2" fill="none" stroke="currentColor" strokeWidth="1.2" />
              </svg>
            </button>
            <button className="app-window-btn close" aria-label="Close" onClick={() => void window.velocityAPI.windowClose()}>
              <svg width="11" height="11" viewBox="0 0 12 12" aria-hidden="true">
                <path d="M3 3l6 6M9 3L3 9" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" />
              </svg>
            </button>
          </div>
        </div>

        <div className="app-root">
          <Sidebar backendStatus={backendStatus} />
          <div className="app-center">
            <WorkspaceHeader
              backendStatus={backendStatus}
              textModel={selectedTextModel}
              imageModel={selectedImageModel}
              activeMode={generationMode}
              onOpenModelLibrary={() => setShowModelLibrary(true)}
            />
            {modelCatalogError && (
              <div className="app-shell-banner">
                <span>{modelCatalogError}</span>
                <button onClick={() => void refreshCatalog()}>Retry</button>
              </div>
            )}
            <ViewportContainer
              backendReady={backendStatus === 'ready'}
              onObjectSelected={setSelectedObject}
              transformCommand={viewportTransformCommand}
            />
            <PromptBar
              mode={generationMode}
              onModeChange={setGenerationMode}
              isGenerating={status === 'generating'}
              onSubmitText={handleSubmitText}
              onSubmitImage={handleSubmitImage}
              onCancel={handleCancel}
              textModels={textModels}
              imageModels={imageModels}
              selectedTextModelId={textModelId}
              selectedImageModelId={imageModelId}
              onSelectModel={setSelectedModel}
              installSession={installSession}
              onInstallModel={handleStartInstall}
              onDismissInstall={installSession ? () => setInstallSession(null) : undefined}
              onImagePreviewChange={handleInputImagePreviewChange}
            />
          </div>
          <PropertiesPanel
            selectedObject={selectedObject}
            activeMode={generationMode}
            selectedModel={activeSelectedModel}
            inputImage={inputImage}
            textureOptions={textureOptions}
            pipelineOptions={pipelineOptions}
            onTextureOptionsChange={setTextureOptions}
            onPipelineOptionsChange={setPipelineOptions}
            installSession={installSession}
            onInstallModel={handleStartInstall}
            onOpenModelLibrary={() => setShowModelLibrary(true)}
            onUpdateObjectTransform={handleUpdateObjectTransform}
          />
        </div>
      </div>

      {showModelLibrary && (
        <ModelDownloader
          onClose={() => setShowModelLibrary(false)}
          preferredMode={generationMode}
          onSelectModel={(mode, modelId) => setSelectedModel(mode, modelId)}
          onInstallModel={handleStartInstall}
        />
      )}
    </>
  )
}

function fileToBase64(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader()
    reader.onload = () => {
      const result = reader.result as string
      resolve(result.split(',')[1])
    }
    reader.onerror = reject
    reader.readAsDataURL(file)
  })
}

function createGeneratedOutputPoll(afterMs: number, requireTextured = false) {
  let stopped = false
  let timer: number | null = null
  let gateLogged = false
  let armedLogged = false
  let waitingForTexturedPath: string | null = null
  let waitingForTexturedSince = 0
  let nextTextureReminderAt = 0

  const promise = new Promise<LatestGeneratedOutput>((resolve, reject) => {
    const poll = async () => {
      if (stopped) return

      const generation = useGenerationStore.getState()
      if (!canScanGeneratedOutputs(generation.logs)) {
        if (!gateLogged) {
          generation.appendLog('Waiting for mesh decoding to finish before scanning outputs...')
          gateLogged = true
        }
        if (!stopped) {
          timer = window.setTimeout(poll, 900)
        }
        return
      }

      if (!armedLogged) {
        generation.appendLog('Volume decoding complete; scanning outputs for a stable GLB...')
        armedLogged = true
      }

      try {
        const output = await fetchLatestGeneratedOutput(afterMs)
        if (output && !stopped) {
          if (requireTextured && !isTexturedModelPath(output.model_path)) {
            const now = Date.now()
            const store = useGenerationStore.getState()
            if (waitingForTexturedPath !== output.model_path) {
              waitingForTexturedPath = output.model_path
              waitingForTexturedSince = now
              nextTextureReminderAt = now + 30000
              store.appendLog(`Base GLB detected; texture pass is still running for: ${output.model_path}`)
              store.setProgress('Base mesh is done. Baking textured GLB...')
            } else if (now >= nextTextureReminderAt) {
              const elapsedSeconds = Math.max(1, Math.round((now - waitingForTexturedSince) / 1000))
              if (now - waitingForTexturedSince >= TEXTURED_OUTPUT_WAIT_LIMIT_MS) {
                stopped = true
                reject(new Error(
                  `The base GLB was written, but the textured GLB was not produced after ${elapsedSeconds}s. ` +
                  'Hunyuan3D-Paint likely stalled during remesh, UV unwrap, bake, or texture inpaint.'
                ))
                return
              }
              store.appendLog(`Texture export still running after ${elapsedSeconds}s; waiting for packed textured GLB.`)
              store.setProgress('Still baking UV textures and packed GLB...')
              nextTextureReminderAt = now + 30000
            }
            timer = window.setTimeout(poll, 1800)
            return
          }
          useGenerationStore.getState().appendLog(`Stable GLB detected: ${output.model_path}`)
          resolve(output)
          return
        }
      } catch (err: any) {
        if (!stopped) {
          useGenerationStore.getState().appendLog(`Output poll retry: ${err.message ?? 'latest output unavailable'}`)
        }
      }

      if (!stopped) {
        timer = window.setTimeout(poll, 1200)
      }
    }

    timer = window.setTimeout(poll, 3500)
  })

  return {
    promise,
    stop: () => {
      stopped = true
      if (timer !== null) {
        window.clearTimeout(timer)
      }
    }
  }
}

function isTexturedModelPath(modelPath: string): boolean {
  const lower = modelPath.toLowerCase()
  return lower.includes('_textured.glb') || lower.includes('_material_textured.glb')
}

function canScanGeneratedOutputs(logs: string[]): boolean {
  const joined = logs.join('\n').toLowerCase()

  if (joined.includes('volume decoding')) {
    return /volume\s+decoding:\s*100%/.test(joined) || /volume\s+decoding:[\s\S]*100%/.test(joined)
  }

  return (
    /\.glb\b/.test(joined) ||
    /\bexport(?:ing|ed)?\b/.test(joined) ||
    /\bwrote\b/.test(joined) ||
    /\bsaved\b/.test(joined)
  )
}

function buildFallbackGenerationResponse({
  requestId,
  output,
  selectedModel,
  startedAtMs,
  textureOptions,
  pipelineOptions,
}: {
  requestId: string
  output: LatestGeneratedOutput
  selectedModel: NonNullable<ReturnType<typeof getSelectedModel>>
  startedAtMs: number
  textureOptions: TextureOptions
  pipelineOptions: PipelineOptions
}): GenerationResponse {
  const metadata: GenerationMetadata = {
    vertex_count: output.vertex_count,
    face_count: output.face_count,
    generation_time_ms: Math.max(0, Date.now() - startedAtMs),
    pipeline: selectedModel.family,
    model_id: selectedModel.id,
    model_name: selectedModel.name,
    texture_applied: output.model_path.toLowerCase().includes('_textured'),
    texture_checkpoint: textureOptions.enabled ? textureOptions.checkpoint ?? null : null,
    material_texture_dir: null,
    material_textures: [],
    pipeline_preset: pipelineOptions.preset,
    target_face_count: pipelineOptions.target_face_count ?? null,
    texture_size: pipelineOptions.texture_size ?? null,
  }

  return {
    request_id: requestId,
    model_path: output.model_path,
    metadata
  }
}

function settleLateGenerationResponse(requestId: string, responsePromise: Promise<GenerationResponse>) {
  void responsePromise.then((response) => {
    const generation = useGenerationStore.getState()
    if (generation.requestId !== requestId) {
      return
    }
    if (generation.currentModelPath !== response.model_path) {
      generation.success(response.model_path)
    }
  }).catch(() => {
    // The recovered GLB is already loaded; late provider unwind errors should not
    // replace a usable viewport state.
  })
}

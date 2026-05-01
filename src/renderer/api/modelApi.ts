export type GenerationMode = 'text' | 'image'
export type LibraryMode = GenerationMode | 'multiview'
export type ModelRole = 'generator' | 'assistant'
export type ModelStatus = 'ready' | 'downloaded' | 'setup_required' | 'library_only'
export type InstallJobStatus = 'running' | 'complete' | 'error' | 'manual_required'

export interface ModelCatalogItem {
  id: string
  name: string
  family: string
  role: ModelRole
  summary: string
  description: string
  selection_modes: GenerationMode[]
  library_modes: LibraryMode[]
  recommended: boolean
  repo_url?: string | null
  docs_url?: string | null
  huggingface_url?: string | null
  license_name?: string | null
  vram_hint?: string | null
  size_hint?: string | null
  platform_note?: string | null
  preferred_python?: string | null
  supported_python: string[]
  current_python?: string | null
  python_compatible?: boolean | null
  python_status_detail?: string | null
  install_steps: string[]
  downloaded: boolean
  generation_ready: boolean
  status: ModelStatus
  status_detail: string
}

export interface ModelCatalogResponse {
  models: ModelCatalogItem[]
}

interface DownloadModelResponse {
  model_id: string
  destination: string
}

export interface ModelInstallStartResponse {
  job_id: string
  model_id: string
}

export interface ModelInstallStatusResponse {
  job_id: string
  model_id: string
  model_name: string
  status: InstallJobStatus
  current_step: number
  step_count: number
  active_step?: string | null
  logs: string[]
  status_detail: string
  error?: string | null
}

interface ApiError {
  error: string
  details: string
}

async function parseError(payload: ApiError | string, status: number): Promise<never> {
  if (typeof payload === 'string') {
    throw new Error(payload || `HTTP ${status}`)
  }
  throw new Error(payload.details ?? payload.error ?? `HTTP ${status}`)
}

export async function fetchModelCatalog(): Promise<ModelCatalogItem[]> {
  const res = await window.velocityAPI.backendRequest<ModelCatalogResponse | ApiError>({
    path: '/models',
    method: 'GET'
  })
  if (!res.ok) {
    await parseError(res.data as ApiError, res.status)
  }
  return (res.data as ModelCatalogResponse).models
}

export async function downloadModelRepo(modelId: string): Promise<DownloadModelResponse> {
  const res = await window.velocityAPI.backendRequest<DownloadModelResponse | ApiError>({
    path: '/models/download',
    method: 'POST',
    body: { model_id: modelId }
  })
  if (!res.ok) {
    await parseError(res.data as ApiError, res.status)
  }
  return res.data as DownloadModelResponse
}

export async function startModelInstall(modelId: string): Promise<ModelInstallStartResponse> {
  const res = await window.velocityAPI.backendRequest<ModelInstallStartResponse | ApiError>({
    path: '/models/install',
    method: 'POST',
    body: { model_id: modelId }
  })
  if (!res.ok) {
    await parseError(res.data as ApiError, res.status)
  }
  return res.data as ModelInstallStartResponse
}

export async function fetchModelInstallStatus(jobId: string): Promise<ModelInstallStatusResponse> {
  const res = await window.velocityAPI.backendRequest<ModelInstallStatusResponse | ApiError>({
    path: `/models/install/${jobId}`,
    method: 'GET'
  })
  if (!res.ok) {
    await parseError(res.data as ApiError, res.status)
  }
  return res.data as ModelInstallStatusResponse
}

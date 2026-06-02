export interface TextureOptions {
  enabled: boolean
  checkpoint?: string
}

export type PipelinePresetId = 'preview' | 'balanced' | 'building_module' | 'game_asset' | 'production'

export interface PipelineOptions {
  preset: PipelinePresetId
  target_face_count?: number
  texture_size?: number
}

export interface GenerationRequest {
  type: 'text' | 'image'
  prompt?: string
  image_base64?: string
  reference_image_base64?: string
  model_id?: string
  texture_options?: TextureOptions
  pipeline_options?: PipelineOptions
  request_id: string
}

export interface GenerationMetadata {
  vertex_count: number
  face_count: number
  generation_time_ms: number
  pipeline: string
  model_id: string
  model_name: string
  texture_applied: boolean
  texture_checkpoint?: string | null
  material_texture_dir?: string | null
  material_textures?: string[]
  pipeline_preset: PipelinePresetId
  target_face_count?: number | null
  texture_size?: number | null
}

export interface GenerationResponse {
  request_id: string
  model_path: string
  metadata: GenerationMetadata
}

export interface LatestGeneratedOutput {
  model_path: string
  size: number
  modified_ms: number
  vertex_count: number
  face_count: number
}

export interface ApiError {
  error: string
  details: string
}

async function parseError(payload: ApiError | string, status: number): Promise<never> {
  if (typeof payload === 'string') {
    throw new Error(payload || `HTTP ${status}`)
  }
  throw new Error(payload.details ?? payload.error ?? `HTTP ${status}`)
}

export async function submitGeneration(req: GenerationRequest): Promise<GenerationResponse> {
  const res = await window.velocityAPI.backendRequest<GenerationResponse | ApiError>({
    path: '/generate',
    method: 'POST',
    body: req
  })

  if (!res.ok) {
    await parseError(res.data as ApiError, res.status)
  }

  return res.data as GenerationResponse
}

export async function cancelGeneration(requestId: string): Promise<void> {
  const res = await window.velocityAPI.backendRequest<unknown | ApiError>({
    path: '/cancel',
    method: 'POST',
    body: { request_id: requestId }
  })
  if (!res.ok) {
    await parseError(res.data as ApiError, res.status)
  }
}

export async function fetchLatestGeneratedOutput(afterMs: number, allowBackendFallback = false): Promise<LatestGeneratedOutput | null> {
  const local = await window.velocityAPI.getLatestModelFile({
    afterMs: Math.max(0, Math.floor(afterMs)),
    stableMs: 1800
  }).catch(() => null)

  if (local) {
    return local
  }

  if (!allowBackendFallback) {
    return null
  }

  const res = await window.velocityAPI.backendRequest<LatestGeneratedOutput | ApiError>({
    path: `/outputs/latest?after_ms=${Math.max(0, Math.floor(afterMs))}`,
    method: 'GET'
  })

  if (res.status === 404) {
    return null
  }

  if (!res.ok) {
    await parseError(res.data as ApiError, res.status)
  }

  return res.data as LatestGeneratedOutput
}

export async function exportModel(
  modelPath: string,
  outputPath: string,
  format: 'glb' | 'obj' | 'fbx'
): Promise<string> {
  const res = await window.velocityAPI.backendRequest<{ output_path: string } | ApiError>({
    path: '/export',
    method: 'POST',
    body: { model_path: modelPath, output_path: outputPath, format }
  })

  if (!res.ok) {
    await parseError(res.data as ApiError, res.status)
  }

  const data = res.data as { output_path: string }
  return data.output_path
}

import { contextBridge, ipcRenderer } from 'electron'

export interface SaveDialogOptions {
  title?: string
  defaultPath?: string
  filters?: Array<{ name: string; extensions: string[] }>
}

export interface BackendStatusEvent {
  status: 'starting' | 'ready' | 'error' | 'stopped'
  message?: string
}

export interface BackendRequestPayload {
  path: string
  method?: 'GET' | 'POST'
  body?: unknown
}

export interface BackendResponse<T = unknown> {
  ok: boolean
  status: number
  data: T
}

export interface LatestModelFileRequest {
  afterMs?: number
  stableMs?: number
}

export interface LatestModelFileResponse {
  model_path: string
  size: number
  modified_ms: number
  vertex_count: number
  face_count: number
}

export interface UpdateInfo {
  version: string
  releaseNotes?: string
}

export interface VelocityAPI {
  getBackendPort(): Promise<number>
  backendRequest<T = unknown>(payload: BackendRequestPayload): Promise<BackendResponse<T>>
  readModelFile(absolutePath: string): Promise<ArrayBuffer>
  getLatestModelFile(request: LatestModelFileRequest): Promise<LatestModelFileResponse | null>
  showSaveDialog(options: SaveDialogOptions): Promise<string | null>
  openExternal(url: string): Promise<void>
  windowMinimize(): Promise<void>
  windowToggleMaximize(): Promise<void>
  windowClose(): Promise<void>
  onBackendStatus(cb: (event: BackendStatusEvent) => void): () => void
  onBackendLog(cb: (line: string) => void): () => void
  onUpdateAvailable(cb: (info: UpdateInfo) => void): () => void
  readHistoryFile(): Promise<ArrayBuffer>
  writeHistoryFile(content: string): Promise<void>
  getVenvPath(): Promise<string>
  setVenvPath(venvPath: string): Promise<void>
  showOpenDialog(options: { title?: string; properties: string[] }): Promise<string | null>
  getConfig(): Promise<Record<string, any>>
  setConfig(partial: Record<string, any>): Promise<void>
}

const velocityAPI: VelocityAPI = {
  getBackendPort(): Promise<number> {
    return ipcRenderer.invoke('get-backend-port')
  },

  backendRequest<T = unknown>(payload: BackendRequestPayload): Promise<BackendResponse<T>> {
    return ipcRenderer.invoke('backend-request', payload)
  },

  readModelFile(absolutePath: string): Promise<ArrayBuffer> {
    return ipcRenderer.invoke('read-model-file', absolutePath)
  },

  getLatestModelFile(request: LatestModelFileRequest): Promise<LatestModelFileResponse | null> {
    return ipcRenderer.invoke('get-latest-model-file', request)
  },

  showSaveDialog(options: SaveDialogOptions): Promise<string | null> {
    return ipcRenderer.invoke('show-save-dialog', options)
  },

  openExternal(url: string): Promise<void> {
    return ipcRenderer.invoke('open-external', url)
  },

  windowMinimize(): Promise<void> {
    return ipcRenderer.invoke('window-minimize')
  },

  windowToggleMaximize(): Promise<void> {
    return ipcRenderer.invoke('window-toggle-maximize')
  },

  windowClose(): Promise<void> {
    return ipcRenderer.invoke('window-close')
  },

  showOpenDialog(options): Promise<string | null> {
    return ipcRenderer.invoke('show-open-dialog', options)
  },

  onBackendStatus(cb: (event: BackendStatusEvent) => void): () => void {
    const handler = (_: Electron.IpcRendererEvent, event: BackendStatusEvent) => cb(event)
    ipcRenderer.on('backend-status', handler)
    return () => ipcRenderer.removeListener('backend-status', handler)
  },

  onBackendLog(cb: (line: string) => void): () => void {
    const handler = (_: Electron.IpcRendererEvent, line: string) => cb(line)
    ipcRenderer.on('backend-log', handler)
    return () => ipcRenderer.removeListener('backend-log', handler)
  },

  onUpdateAvailable(cb: (info: UpdateInfo) => void): () => void {
    const handler = (_: Electron.IpcRendererEvent, info: UpdateInfo) => cb(info)
    ipcRenderer.on('update-available', handler)
    return () => ipcRenderer.removeListener('update-available', handler)
  },

  readHistoryFile(): Promise<ArrayBuffer> {
    return ipcRenderer.invoke('read-history-file')
  },

  writeHistoryFile(content: string): Promise<void> {
    return ipcRenderer.invoke('write-history-file', content)
  },

  getVenvPath(): Promise<string> {
    return ipcRenderer.invoke('get-venv-path')
  },

  setVenvPath(venvPath: string): Promise<void> {
    return ipcRenderer.invoke('set-venv-path', venvPath)
  },

  getConfig(): Promise<Record<string, any>> {
    return ipcRenderer.invoke('get-config')
  },

  setConfig(partial: Record<string, any>): Promise<void> {
    return ipcRenderer.invoke('set-config', partial)
  }
}

contextBridge.exposeInMainWorld('velocityAPI', velocityAPI)

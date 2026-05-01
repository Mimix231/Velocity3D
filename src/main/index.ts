import { app, BrowserWindow, ipcMain, dialog, shell } from 'electron'
import { join } from 'path'
import * as fs from 'fs'
import * as http from 'http'
import { getAvailablePort } from './portUtils'
import { BackendManager } from './BackendManager'
import { initAutoUpdater } from './updater'

let backendManager: BackendManager | null = null
let mainWindow: BrowserWindow | null = null

const CONFIG_FILE = () => join(app.getPath('userData'), 'velocity3d.config.json')

interface AppConfig {
  venvPath?: string
  setupComplete?: boolean
  theme?: string
}

interface BackendRequestPayload {
  path: string
  method?: 'GET' | 'POST'
  body?: unknown
}

interface LocalHttpResponse {
  status: number
  headers: http.IncomingHttpHeaders
  body: string
}

interface LatestModelFileRequest {
  afterMs?: number
  stableMs?: number
}

interface LatestModelFileResponse {
  model_path: string
  size: number
  modified_ms: number
  vertex_count: number
  face_count: number
}

async function readConfig(): Promise<AppConfig> {
  try {
    const raw = await fs.promises.readFile(CONFIG_FILE(), 'utf-8')
    return JSON.parse(raw)
  } catch {
    return {}
  }
}

async function writeConfig(cfg: AppConfig): Promise<void> {
  await fs.promises.writeFile(CONFIG_FILE(), JSON.stringify(cfg, null, 2), 'utf-8')
}

function getPythonFallback(): string {
  return process.platform === 'win32' ? 'python' : 'python3'
}

function getBackendDir(): string {
  if (app.isPackaged) {
    return join(process.resourcesPath, 'backend')
  }
  return join(app.getAppPath())
}

function getBaseDir(): string {
  if (app.isPackaged) {
    return app.getPath('userData')
  }
  return app.getAppPath()
}

function backendRequestTimeoutMs(path: string): number {
  if (path.startsWith('/generate')) return 4 * 60 * 60 * 1000
  if (path.startsWith('/export')) return 60 * 60 * 1000
  if (path.startsWith('/models/install')) return 30 * 60 * 1000
  if (path.startsWith('/images/remove-background')) return 10 * 60 * 1000
  return 2 * 60 * 1000
}

async function looksLikeGlb(path: string): Promise<boolean> {
  const handle = await fs.promises.open(path, 'r')
  try {
    const header = Buffer.alloc(4)
    const result = await handle.read(header, 0, 4, 0)
    return result.bytesRead === 4 && header.toString('utf-8') === 'glTF'
  } finally {
    await handle.close()
  }
}

async function readGlbMeshStats(path: string): Promise<{ vertex_count: number; face_count: number }> {
  const buffer = await fs.promises.readFile(path)
  if (buffer.length < 20 || buffer.subarray(0, 4).toString('utf-8') !== 'glTF') {
    return { vertex_count: 0, face_count: 0 }
  }

  const totalLength = buffer.readUInt32LE(8)
  let offset = 12
  while (offset + 8 <= Math.min(totalLength, buffer.length)) {
    const chunkLength = buffer.readUInt32LE(offset)
    const chunkType = buffer.subarray(offset + 4, offset + 8).toString('utf-8')
    offset += 8
    if (offset + chunkLength > buffer.length) break

    if (chunkType === 'JSON') {
      const rawJson = buffer.subarray(offset, offset + chunkLength).toString('utf-8').trim()
      const document = JSON.parse(rawJson)
      const accessors = Array.isArray(document.accessors) ? document.accessors : []
      let vertex_count = 0
      let face_count = 0

      for (const mesh of Array.isArray(document.meshes) ? document.meshes : []) {
        for (const primitive of Array.isArray(mesh.primitives) ? mesh.primitives : []) {
          const positionAccessor = primitive.attributes?.POSITION
          if (Number.isInteger(positionAccessor) && positionAccessor < accessors.length) {
            vertex_count += Number(accessors[positionAccessor].count ?? 0)
          }

          const indicesAccessor = primitive.indices
          if (Number.isInteger(indicesAccessor) && indicesAccessor < accessors.length) {
            face_count += Math.floor(Number(accessors[indicesAccessor].count ?? 0) / 3)
          } else if (Number.isInteger(positionAccessor) && positionAccessor < accessors.length) {
            face_count += Math.floor(Number(accessors[positionAccessor].count ?? 0) / 3)
          }
        }
      }

      return { vertex_count, face_count }
    }

    offset += chunkLength
  }

  return { vertex_count: 0, face_count: 0 }
}

async function findLatestStableModelFile({
  afterMs = 0,
  stableMs = 1800
}: LatestModelFileRequest): Promise<LatestModelFileResponse | null> {
  const outputDir = join(getBaseDir(), 'outputs')
  const now = Date.now()
  const cutoff = Math.max(0, afterMs - 500)

  let entries: string[]
  try {
    entries = await fs.promises.readdir(outputDir)
  } catch {
    return null
  }

  let latest: { path: string; stat: fs.Stats } | null = null
  for (const entry of entries) {
    if (!entry.toLowerCase().endsWith('.glb')) continue

    const path = join(outputDir, entry)
    let stat: fs.Stats
    try {
      stat = await fs.promises.stat(path)
      if (!stat.isFile()) continue
      if (stat.mtimeMs < cutoff) continue
      if (now - stat.mtimeMs < stableMs) continue
      if (stat.size < 20) continue
      if (!(await looksLikeGlb(path))) continue
    } catch {
      continue
    }

    if (!latest || stat.mtimeMs > latest.stat.mtimeMs) {
      latest = { path, stat }
    }
  }

  if (!latest) return null

  const stats = await readGlbMeshStats(latest.path).catch(() => ({ vertex_count: 0, face_count: 0 }))
  return {
    model_path: latest.path,
    size: latest.stat.size,
    modified_ms: Math.floor(latest.stat.mtimeMs),
    ...stats
  }
}

function requestLocalBackend(url: URL, payload: BackendRequestPayload): Promise<LocalHttpResponse> {
  const method = payload.method ?? 'GET'
  const body = payload.body !== undefined ? JSON.stringify(payload.body) : undefined
  const timeoutMs = backendRequestTimeoutMs(payload.path)

  return new Promise((resolve, reject) => {
    const req = http.request(
      {
        hostname: url.hostname,
        port: url.port,
        path: `${url.pathname}${url.search}`,
        method,
        headers: body !== undefined
          ? {
              'Content-Type': 'application/json',
              'Content-Length': Buffer.byteLength(body)
            }
          : undefined,
      },
      (res) => {
        const chunks: Buffer[] = []
        res.on('data', (chunk: Buffer) => chunks.push(chunk))
        res.on('end', () => {
          resolve({
            status: res.statusCode ?? 0,
            headers: res.headers,
            body: Buffer.concat(chunks).toString('utf-8')
          })
        })
      }
    )

    req.on('error', reject)
    req.setTimeout(timeoutMs, () => {
      req.destroy(new Error(`Backend request timed out after ${Math.round(timeoutMs / 1000)}s: ${payload.path}`))
    })

    if (body !== undefined) {
      req.write(body)
    }
    req.end()
  })
}

async function createWindow(): Promise<void> {
  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    minWidth: 900,
    minHeight: 600,
    frame: false,
    autoHideMenuBar: true,
    webPreferences: {
      preload: join(__dirname, '../preload/index.js'),
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: false
    },
    backgroundColor: '#0d0d0d',
    title: 'Velocity3D',
    show: false
  })

  mainWindow.once('ready-to-show', () => {
    mainWindow?.show()
  })

  if (process.env.NODE_ENV === 'development') {
    mainWindow.loadURL('http://localhost:5173')
  } else {
    mainWindow.loadFile(join(__dirname, '../renderer/index.html'))
  }
}

// ── IPC: venv config ──────────────────────────────────────────────────────────
ipcMain.handle('get-venv-path', async () => {
  if (process.env.HOME_VENV) return process.env.HOME_VENV
  const cfg = await readConfig()
  return cfg.venvPath ?? ''
})

ipcMain.handle('set-venv-path', async (_event, venvPath: string) => {
  process.env.HOME_VENV = venvPath
  const cfg = await readConfig()
  cfg.venvPath = venvPath
  await writeConfig(cfg)
})

ipcMain.handle('get-config', async () => {
  return readConfig()
})

ipcMain.handle('set-config', async (_event, partial: Partial<AppConfig>) => {
  const cfg = await readConfig()
  Object.assign(cfg, partial)
  await writeConfig(cfg)
  // Apply venv immediately if provided
  if (partial.venvPath !== undefined) {
    process.env.HOME_VENV = partial.venvPath
  }
})

// ── IPC: file / dialog ────────────────────────────────────────────────────────
// Port is stored here so it's always available even before backend is ready
let _cachedPort: number | null = null
ipcMain.handle('get-backend-port', () => _cachedPort)

ipcMain.handle('backend-request', async (_event, payload: BackendRequestPayload) => {
  if (!_cachedPort) {
    throw new Error('Backend port is not ready yet.')
  }

  const method = payload.method ?? 'GET'
  const url = new URL(`http://127.0.0.1:${_cachedPort}${payload.path}`)
  const res = await requestLocalBackend(url, { ...payload, method })

  const contentType = String(res.headers['content-type'] ?? '')
  let data: unknown = res.body
  if (contentType.includes('application/json') && res.body) {
    data = JSON.parse(res.body)
  }

  return {
    ok: res.status >= 200 && res.status < 300,
    status: res.status,
    data
  }
})

ipcMain.handle('read-model-file', async (_event, absolutePath: string) => {
  const buffer = await fs.promises.readFile(absolutePath)
  return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
})

ipcMain.handle('get-latest-model-file', async (_event, request: LatestModelFileRequest) => {
  return findLatestStableModelFile(request ?? {})
})

ipcMain.handle('show-save-dialog', async (_event, options) => {
  if (!mainWindow) return null
  const result = await dialog.showSaveDialog(mainWindow, options)
  return result.canceled ? null : result.filePath ?? null
})

ipcMain.handle('show-open-dialog', async (_event, options) => {
  if (!mainWindow) return null
  const result = await dialog.showOpenDialog(mainWindow, options)
  return result.canceled || !result.filePaths.length ? null : result.filePaths[0]
})

ipcMain.handle('open-external', async (_event, url: string) => {
  await shell.openExternal(url)
})

ipcMain.handle('window-minimize', () => {
  mainWindow?.minimize()
})

ipcMain.handle('window-toggle-maximize', () => {
  if (!mainWindow) return
  if (mainWindow.isMaximized()) {
    mainWindow.unmaximize()
  } else {
    mainWindow.maximize()
  }
})

ipcMain.handle('window-close', () => {
  mainWindow?.close()
})

ipcMain.handle('read-history-file', async () => {
  const historyPath = join(app.getPath('userData'), 'history.json')
  try {
    const buffer = await fs.promises.readFile(historyPath)
    return buffer.buffer.slice(buffer.byteOffset, buffer.byteOffset + buffer.byteLength)
  } catch {
    return new ArrayBuffer(0)
  }
})

ipcMain.handle('write-history-file', async (_event, content: string) => {
  const historyPath = join(app.getPath('userData'), 'history.json')
  await fs.promises.writeFile(historyPath, content, 'utf-8')
})

// ── App lifecycle ─────────────────────────────────────────────────────────────
app.whenReady().then(async () => {
  await createWindow()

  if (mainWindow) initAutoUpdater(mainWindow)

  try {
    // Load saved venv path into env if not already set
    if (!process.env.HOME_VENV) {
      const cfg = await readConfig()
      if (cfg.venvPath) process.env.HOME_VENV = cfg.venvPath
    }

    const port = await getAvailablePort()
    _cachedPort = port
    const pythonPath = BackendManager.resolvePythonPath(getPythonFallback())
    const backendDir = getBackendDir()
    const baseDir = getBaseDir()

    backendManager = new BackendManager(port, pythonPath, backendDir, baseDir)

    // Forward status and log events to renderer
    backendManager.on('status', (status: string, message?: string) => {
      mainWindow?.webContents.send('backend-status', { status, message })
    })

    backendManager.on('log', (line: string) => {
      mainWindow?.webContents.send('backend-log', line)
    })

    backendManager.start().catch((err) => {
      console.error('Backend failed to start:', err)
      mainWindow?.webContents.send('backend-status', {
        status: 'error',
        message: err.message
      })
    })
  } catch (err) {
    console.error('App initialization error:', err)
    mainWindow?.webContents.send('backend-status', {
      status: 'error',
      message: (err as Error).message
    })
  }
})

app.on('window-all-closed', () => {
  backendManager?.stop()
  if (process.platform !== 'darwin') app.quit()
})

app.on('before-quit', () => {
  backendManager?.stop()
})

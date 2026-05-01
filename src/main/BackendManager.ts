import { ChildProcess, spawn } from 'child_process'
import { EventEmitter } from 'events'
import * as path from 'path'

export type BackendStatus = 'starting' | 'ready' | 'error' | 'stopped'

export class BackendManager extends EventEmitter {
  private process: ChildProcess | null = null
  private port: number
  private pythonPath: string
  private backendDir: string
  private baseDir: string
  private status: BackendStatus = 'stopped'

  constructor(port: number, pythonPath: string, backendDir: string, baseDir?: string) {
    super()
    this.port = port
    this.pythonPath = pythonPath
    this.backendDir = backendDir
    this.baseDir = baseDir ?? backendDir
  }

  getPort(): number {
    return this.port
  }

  /**
   * Resolve the Python executable to use.
   * Priority: HOME_VENV env var > constructor pythonPath
   *
   * HOME_VENV should point to the venv root, e.g. G:/AIExperiments/Velocity3D/env
   * We derive the python executable from it automatically.
   */
  static resolvePythonPath(fallback: string): string {
    const venvRoot = process.env.HOME_VENV
    if (venvRoot) {
      const isWin = process.platform === 'win32'
      const exe = isWin
        ? path.join(venvRoot, 'Scripts', 'python.exe')
        : path.join(venvRoot, 'bin', 'python3')
      return exe
    }
    return fallback
  }

  async start(): Promise<void> {
    this.status = 'starting'
    this.emit('status', 'starting', 'Initializing backend...')

    const resolvedPython = BackendManager.resolvePythonPath(this.pythonPath)
    const huggingFaceDir = path.join(this.baseDir, 'HuggingFace')
    const huggingFaceHubDir = path.join(huggingFaceDir, 'hub')
    const transformersDir = path.join(huggingFaceDir, 'transformers')
    const hy3dgenDir = path.join(huggingFaceDir, 'hy3dgen')

    const env = {
      ...process.env,
      VELOCITY_PORT: String(this.port),
      VELOCITY_BASE_DIR: this.baseDir,
      HF_HOME: huggingFaceDir,
      HF_HUB_CACHE: huggingFaceHubDir,
      HUGGINGFACE_HUB_CACHE: huggingFaceHubDir,
      TRANSFORMERS_CACHE: transformersDir,
      DIFFUSERS_CACHE: path.join(huggingFaceDir, 'diffusers'),
      HF_DATASETS_CACHE: path.join(huggingFaceDir, 'datasets'),
      HF_ASSETS_CACHE: path.join(huggingFaceDir, 'assets'),
      HF_HUB_DISABLE_PROGRESS_BARS: '1',
      TORCH_HOME: path.join(huggingFaceDir, 'torch'),
      HY3DGEN_MODELS: hy3dgenDir,
      PYTHONUNBUFFERED: '1',
      // Pass venv path into the Python process so it can use it if needed
      HOME_VENV: process.env.HOME_VENV ?? ''
    }

    this.emit('log', `Using Python: ${resolvedPython}`)
    this.emit('log', `Backend dir: ${this.backendDir}`)
    this.emit('log', `Base dir: ${this.baseDir}`)
    this.emit('log', `Hugging Face cache: ${huggingFaceDir}`)
    this.emit('log', `Port: ${this.port}`)
    this.emit('log', 'Starting uvicorn...')

    this.process = spawn(
      resolvedPython,
      ['-m', 'uvicorn', 'backend.main:app', '--host', '127.0.0.1', '--port', String(this.port)],
      {
        cwd: this.backendDir,
        env,
        stdio: ['ignore', 'pipe', 'pipe']
      }
    )

    this.process.stdout?.on('data', (data: Buffer) => {
      const line = data.toString().trim()
      console.log('[Backend]', line)
      this.emit('log', line)
    })

    this.process.stderr?.on('data', (data: Buffer) => {
      const line = data.toString().trim()
      console.error('[Backend ERR]', line)
      this.emit('log', line)
    })

    this.process.on('exit', (code) => {
      if (this.status !== 'stopped') {
        this.status = 'error'
        this.emit('status', 'error', `Backend exited with code ${code}`)
      }
    })

    try {
      await this.waitForHealth(30000)
      this.status = 'ready'
      this.emit('log', 'Backend ready.')
      this.emit('status', 'ready')
    } catch (err) {
      this.status = 'error'
      this.emit('status', 'error', (err as Error).message)
      throw err
    }
  }

  private async waitForHealth(timeoutMs: number): Promise<void> {
    const start = Date.now()
    let delay = 300

    while (Date.now() - start < timeoutMs) {
      try {
        const res = await fetch(`http://127.0.0.1:${this.port}/health`)
        if (res.ok) return
      } catch {
        // not ready yet
      }
      await new Promise((r) => setTimeout(r, delay))
      delay = Math.min(delay * 1.5, 2000)
    }

    throw new Error(`Backend did not become ready within ${timeoutMs / 1000}s`)
  }

  stop(): void {
    this.status = 'stopped'
    if (this.process) {
      this.process.kill('SIGTERM')
      this.process = null
    }
  }
}

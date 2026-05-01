import { create } from 'zustand'

export type GenerationStatus = 'idle' | 'generating' | 'complete' | 'error' | 'cancelled'

export interface GenerationState {
  status: GenerationStatus
  requestId: string | null
  progress: string | null
  logs: string[]
  errorMessage: string | null
  currentModelPath: string | null
}

interface GenerationActions {
  submit: (requestId: string) => void
  setProgress: (progress: string) => void
  appendLog: (line: string) => void
  success: (modelPath: string) => void
  setError: (message: string) => void
  cancel: () => void
  dismiss: () => void
  reset: () => void
}

const initialState: GenerationState = {
  status: 'idle',
  requestId: null,
  progress: null,
  logs: [],
  errorMessage: null,
  currentModelPath: null
}

export const useGenerationStore = create<GenerationState & GenerationActions>((set) => ({
  ...initialState,

  submit: (requestId: string) =>
    set({
      status: 'generating',
      requestId,
      progress: 'Starting generation...',
      logs: [`Starting generation ${requestId}...`],
      errorMessage: null
    }),

  setProgress: (progress: string) =>
    set((state) => state.status === 'generating'
      ? {
        progress,
        logs: state.logs[state.logs.length - 1] === progress
          ? state.logs
          : [...state.logs, progress].slice(-400)
      }
      : {}),

  appendLog: (line: string) =>
    set((state) => state.status === 'generating'
      ? {
        logs: state.logs[state.logs.length - 1] === line
          ? state.logs
          : [...state.logs, line].slice(-400)
      }
      : {}),

  success: (modelPath: string) =>
    set((state) => ({
      status: 'complete',
      progress: null,
      logs: [...state.logs, `Generation complete: ${modelPath}`].slice(-400),
      currentModelPath: modelPath
    })),

  setError: (message: string) =>
    set((state) => ({
      status: 'error',
      progress: null,
      logs: [...state.logs, `! ${message}`].slice(-400),
      errorMessage: message
    })),

  cancel: () =>
    set((state) => ({
      status: 'cancelled',
      requestId: null,
      progress: null,
      logs: [...state.logs, 'Generation cancelled.'].slice(-400)
    })),

  dismiss: () =>
    set((state) => {
      if (state.status === 'complete' || state.status === 'error' || state.status === 'cancelled') {
        return { status: 'idle', errorMessage: null, progress: null }
      }
      return {}
    }),

  reset: () => set(initialState)
}))

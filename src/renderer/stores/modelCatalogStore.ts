import { create } from 'zustand'
import { fetchModelCatalog, type GenerationMode, type ModelCatalogItem } from '../api/modelApi'

const STORAGE_KEY = 'velocity3d-model-preferences'

interface StoredPreferences {
  textModelId?: string
  imageModelId?: string
}

interface ModelCatalogState {
  models: ModelCatalogItem[]
  loaded: boolean
  loading: boolean
  error: string | null
  textModelId: string | null
  imageModelId: string | null
}

interface ModelCatalogActions {
  load: () => Promise<void>
  refresh: () => Promise<void>
  setSelectedModel: (mode: GenerationMode, modelId: string) => void
}

function loadPreferences(): StoredPreferences {
  try {
    const raw = localStorage.getItem(STORAGE_KEY)
    return raw ? JSON.parse(raw) as StoredPreferences : {}
  } catch {
    return {}
  }
}

function persistPreferences(next: StoredPreferences) {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(next))
  } catch {
    // Ignore persistence failures in the renderer.
  }
}

function sleep(ms: number) {
  return new Promise((resolve) => setTimeout(resolve, ms))
}

function chooseDefaultModel(models: ModelCatalogItem[], mode: GenerationMode, preferredId?: string): string | null {
  const compatible = models.filter((model) => model.selection_modes.includes(mode))
  if (preferredId && compatible.some((model) => model.id === preferredId)) {
    return preferredId
  }

  const readyRecommended = compatible.find((model) => model.generation_ready && model.recommended)
  if (readyRecommended) return readyRecommended.id

  const readyAny = compatible.find((model) => model.generation_ready)
  if (readyAny) return readyAny.id

  const recommended = compatible.find((model) => model.recommended)
  if (recommended) return recommended.id

  return compatible[0]?.id ?? null
}

export const useModelCatalogStore = create<ModelCatalogState & ModelCatalogActions>((set, get) => ({
  models: [],
  loaded: false,
  loading: false,
  error: null,
  textModelId: loadPreferences().textModelId ?? null,
  imageModelId: loadPreferences().imageModelId ?? null,

  load: async () => {
    if (get().loaded || get().loading) {
      return
    }
    await get().refresh()
  },

  refresh: async () => {
    set({ loading: true, error: null })
    try {
      let models: ModelCatalogItem[] | null = null
      let lastError: any = null

      for (let attempt = 0; attempt < 4; attempt += 1) {
        try {
          models = await fetchModelCatalog()
          break
        } catch (err: any) {
          lastError = err
          await sleep(250 * (attempt + 1))
        }
      }

      if (!models) {
        throw lastError ?? new Error('Failed to load model catalog')
      }

      const prefs = loadPreferences()
      const textModelId = chooseDefaultModel(models, 'text', prefs.textModelId ?? get().textModelId ?? undefined)
      const imageModelId = chooseDefaultModel(models, 'image', prefs.imageModelId ?? get().imageModelId ?? undefined)
      persistPreferences({ textModelId: textModelId ?? undefined, imageModelId: imageModelId ?? undefined })
      set({
        models,
        loaded: true,
        loading: false,
        textModelId,
        imageModelId,
        error: null
      })
    } catch (err: any) {
      set({
        loading: false,
        loaded: true,
        error: err.message ?? 'Failed to load model catalog'
      })
    }
  },

  setSelectedModel: (mode, modelId) => {
    set((state) => {
      const next = mode === 'text'
        ? { textModelId: modelId, imageModelId: state.imageModelId }
        : { textModelId: state.textModelId, imageModelId: modelId }
      persistPreferences({
        textModelId: next.textModelId ?? undefined,
        imageModelId: next.imageModelId ?? undefined
      })
      return next
    })
  }
}))

export function getSelectedModel(models: ModelCatalogItem[], modelId: string | null): ModelCatalogItem | null {
  if (!modelId) return null
  return models.find((model) => model.id === modelId) ?? null
}

export function getModelsForMode(models: ModelCatalogItem[], mode: GenerationMode): ModelCatalogItem[] {
  return models.filter((model) => model.selection_modes.includes(mode))
}

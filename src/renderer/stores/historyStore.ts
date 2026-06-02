import { create } from 'zustand'

export interface GenerationMetadata {
  vertex_count: number
  face_count: number
  generation_time_ms: number
  pipeline: string
  model_id?: string
  model_name?: string
  texture_applied?: boolean
  texture_checkpoint?: string | null
  material_texture_dir?: string | null
  material_textures?: string[]
  pipeline_preset?: string
  target_face_count?: number | null
  texture_size?: number | null
}

export interface HistoryEntry {
  id: string
  project_id: string | null
  type: 'text' | 'image'
  model_id?: string | null
  model_name?: string | null
  prompt: string | null
  image_filename: string | null
  model_path: string
  timestamp: string
  metadata: GenerationMetadata
  file_missing?: boolean
}

export interface Project {
  id: string
  name: string
  created_at: string
  updated_at: string
}

export interface HistoryData {
  version: number
  projects: Project[]
  entries: HistoryEntry[]
}

interface HistoryState {
  projects: Project[]
  entries: HistoryEntry[]
  activeProjectId: string | null
  loaded: boolean
}

interface HistoryActions {
  load: () => Promise<void>
  addEntry: (entry: Omit<HistoryEntry, 'file_missing'>) => void
  deleteEntry: (id: string) => void
  createProject: (name: string) => Project
  renameProject: (id: string, name: string) => void
  deleteProject: (id: string) => void
  setActiveProject: (id: string | null) => void
}

async function readHistoryFile(): Promise<HistoryData | null> {
  try {
    const buffer: ArrayBuffer = await (window as any).velocityAPI.readHistoryFile()
    const text = new TextDecoder().decode(buffer)
    return JSON.parse(text) as HistoryData
  } catch {
    return null
  }
}

async function writeHistoryFile(data: HistoryData): Promise<void> {
  try {
    const text = JSON.stringify(data, null, 2)
    await (window as any).velocityAPI.writeHistoryFile(text)
  } catch (err) {
    console.error('Failed to write history file:', err)
  }
}

export const useHistoryStore = create<HistoryState & HistoryActions>((set, get) => ({
  projects: [],
  entries: [],
  activeProjectId: null,
  loaded: false,

  load: async () => {
    const data = await readHistoryFile()
    if (!data) {
      set({ loaded: true })
      return
    }

    const entries = (data.entries ?? []).map((e) => ({
      ...e,
      file_missing: false
    }))

    set({
      projects: data.projects ?? [],
      entries,
      loaded: true
    })
  },

  addEntry: (entry) => {
    set((state) => {
      const newEntry: HistoryEntry = {
        ...entry,
        project_id: entry.project_id ?? state.activeProjectId,
        file_missing: false
      }
      const entries = [newEntry, ...state.entries]
      const data: HistoryData = {
        version: 1,
        projects: state.projects,
        entries
      }
      writeHistoryFile(data)
      return { entries }
    })
  },

  deleteEntry: (id) => {
    set((state) => {
      const entries = state.entries.filter((e) => e.id !== id)
      writeHistoryFile({ version: 1, projects: state.projects, entries })
      return { entries }
    })
  },

  createProject: (name) => {
    const project: Project = {
      id: crypto.randomUUID(),
      name,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    }
    set((state) => {
      const projects = [...state.projects, project]
      writeHistoryFile({ version: 1, projects, entries: state.entries })
      return { projects }
    })
    return project
  },

  renameProject: (id, name) => {
    set((state) => {
      const projects = state.projects.map((p) =>
        p.id === id ? { ...p, name, updated_at: new Date().toISOString() } : p
      )
      writeHistoryFile({ version: 1, projects, entries: state.entries })
      return { projects }
    })
  },

  deleteProject: (id) => {
    set((state) => {
      const projects = state.projects.filter((p) => p.id !== id)
      const entries = state.entries.map((e) =>
        e.project_id === id ? { ...e, project_id: null } : e
      )
      writeHistoryFile({ version: 1, projects, entries })
      return {
        projects,
        entries,
        activeProjectId: state.activeProjectId === id ? null : state.activeProjectId
      }
    })
  },

  setActiveProject: (id) => set({ activeProjectId: id })
}))

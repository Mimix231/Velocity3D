import React, { useEffect, useState } from 'react'

import { useGenerationStore } from '../stores/generationStore'
import { useHistoryStore } from '../stores/historyStore'
import { useModelCatalogStore } from '../stores/modelCatalogStore'
import { THEME_CATEGORIES, THEMES, ThemeId, useThemeStore } from '../stores/themeStore'
import './Sidebar.css'

function CollapsibleSection({
  title,
  children,
  defaultOpen = true,
  action
}: {
  title: string
  children: React.ReactNode
  defaultOpen?: boolean
  action?: React.ReactNode
}) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <section className="sidebar-section">
      <div className="sidebar-section-header-row">
        <button className="sidebar-section-header" onClick={() => setOpen(!open)}>
          <span className={`sidebar-arrow ${open ? 'open' : ''}`}>{'>'}</span>
          {title}
        </button>
        {action}
      </div>
      {open && <div className="sidebar-section-body">{children}</div>}
    </section>
  )
}

interface Props {
  backendStatus: 'starting' | 'ready' | 'error' | 'stopped'
}

function statusLabel(status: Props['backendStatus']) {
  if (status === 'ready') return 'Backend ready'
  if (status === 'error') return 'Backend error'
  if (status === 'stopped') return 'Backend stopped'
  return 'Backend starting'
}

export default function Sidebar({ backendStatus }: Props) {
  const { projects, entries, activeProjectId, loaded, load, createProject, setActiveProject } = useHistoryStore()
  const { currentModelPath } = useGenerationStore()
  const { themeId, setTheme } = useThemeStore()
  const models = useModelCatalogStore((state) => state.models)
  const [showThemes, setShowThemes] = useState(false)

  useEffect(() => { void load() }, [load])

  function handleNewProject() {
    const name = prompt('Project name:')
    if (name?.trim()) {
      createProject(name.trim())
    }
  }

  function handleEntryClick(modelPath: string) {
    useGenerationStore.setState({ currentModelPath: modelPath, status: 'complete' })
  }

  const filteredEntries = activeProjectId
    ? entries.filter((entry) => entry.project_id === activeProjectId)
    : entries

  const readyModels = models.filter((model) => model.generation_ready)
  const textModels = models.filter((model) => model.selection_modes.includes('text'))
  const imageModels = models.filter((model) => model.selection_modes.includes('image'))

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="sidebar-brand-mark">V3D</div>
        <div>
          <div className="sidebar-brand-title">Velocity3D</div>
          <div className="sidebar-brand-subtitle">AI 3D editor</div>
        </div>
      </div>

      <div className="sidebar-overview-card">
        <div className={`sidebar-status-pill ${backendStatus}`}>
          <span className="sidebar-status-dot" />
          {statusLabel(backendStatus)}
        </div>
        <div className="sidebar-overview-row">
          <span>Ready</span>
          <strong>{readyModels.length}</strong>
        </div>
        <div className="sidebar-overview-row">
          <span>Text</span>
          <strong>{textModels.length}</strong>
        </div>
        <div className="sidebar-overview-row">
          <span>Image</span>
          <strong>{imageModels.length}</strong>
        </div>
      </div>

      <nav className="sidebar-nav">
        <CollapsibleSection
          title="Projects"
          action={
            <button className="sidebar-action-btn" onClick={handleNewProject} title="New project">
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                <line x1="12" y1="5" x2="12" y2="19" />
                <line x1="5" y1="12" x2="19" y2="12" />
              </svg>
            </button>
          }
        >
          <button
            className={`sidebar-project-item ${activeProjectId === null ? 'active' : ''}`}
            onClick={() => setActiveProject(null)}
          >
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
            </svg>
            <span>All projects</span>
          </button>
          {projects.map((project) => (
            <button
              key={project.id}
              className={`sidebar-project-item ${activeProjectId === project.id ? 'active' : ''}`}
              onClick={() => setActiveProject(project.id)}
            >
              <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z" />
              </svg>
              <span>{project.name}</span>
            </button>
          ))}
          {projects.length === 0 && <div className="sidebar-empty">No projects yet</div>}
        </CollapsibleSection>

        <CollapsibleSection title="History">
          {!loaded && <div className="sidebar-empty">Loading history...</div>}
          {loaded && filteredEntries.length === 0 && (
            <div className="sidebar-empty">No generations yet</div>
          )}
          {filteredEntries.map((entry) => (
            <button
              key={entry.id}
              className={`sidebar-history-item ${entry.file_missing ? 'missing' : ''} ${currentModelPath === entry.model_path ? 'active' : ''}`}
              onClick={() => !entry.file_missing && handleEntryClick(entry.model_path)}
              title={entry.file_missing ? 'File not found' : (entry.prompt ?? entry.image_filename ?? 'Model')}
            >
              <span className="sidebar-history-icon">
                {entry.type === 'image'
                  ? (
                    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <rect x="3" y="3" width="18" height="18" rx="2" />
                      <circle cx="8.5" cy="8.5" r="1.5" />
                      <polyline points="21 15 16 10 5 21" />
                    </svg>
                  )
                  : (
                    <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                    </svg>
                  )}
              </span>
              <span className="sidebar-history-copy">
                <span className="sidebar-history-label">{entry.prompt ?? entry.image_filename ?? 'Model'}</span>
                <span className="sidebar-history-meta">{entry.model_name ?? entry.metadata.model_name ?? 'Unknown model'}</span>
              </span>
              {entry.file_missing && <span className="sidebar-missing-icon">!</span>}
            </button>
          ))}
        </CollapsibleSection>
      </nav>

      <div className="sidebar-bottom">
        <button className="sidebar-bottom-btn sidebar-bottom-btn-full" onClick={() => setShowThemes(!showThemes)}>
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="3" />
            <path d="M12 1v2M12 21v2M4.22 4.22l1.42 1.42M18.36 18.36l1.42 1.42M1 12h2M21 12h2M4.22 19.78l1.42-1.42M18.36 5.64l1.42-1.42" />
          </svg>
          Theme
        </button>

        {showThemes && (
          <div className="sidebar-theme-popover">
            {THEME_CATEGORIES.map((category) => {
              const categoryThemes = Object.values(THEMES).filter((theme) => theme.category === category.id)
              return (
                <div className="sidebar-theme-category" key={category.id}>
                  <div className="sidebar-theme-category-title">{category.name}</div>
                  {categoryThemes.map((theme) => (
                    <button
                      key={theme.id}
                      className={`sidebar-theme-item ${themeId === theme.id ? 'active' : ''}`}
                      onClick={() => {
                        setTheme(theme.id as ThemeId)
                        setShowThemes(false)
                      }}
                    >
                      <span className="sidebar-theme-dot" style={{ background: theme.accent }} />
                      {theme.name}
                      {themeId === theme.id && <span className="sidebar-theme-check">Active</span>}
                    </button>
                  ))}
                </div>
              )
            })}
          </div>
        )}
      </div>
    </aside>
  )
}

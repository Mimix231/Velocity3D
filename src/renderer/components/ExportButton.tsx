import React, { useState } from 'react'

import { exportModel } from '../api/generationApi'
import { useGenerationStore } from '../stores/generationStore'
import './ExportButton.css'

type ExportFormat = 'glb' | 'obj' | 'fbx'

interface ToastState {
  message: string
  type: 'success' | 'error'
}

export default function ExportButton() {
  const currentModelPath = useGenerationStore((state) => state.currentModelPath)
  const [toast, setToast] = useState<ToastState | null>(null)
  const [exporting, setExporting] = useState(false)

  function showToast(message: string, type: 'success' | 'error') {
    setToast({ message, type })
    setTimeout(() => setToast(null), 4000)
  }

  async function handleExport(format: ExportFormat) {
    if (!currentModelPath) return

    const filters: Record<ExportFormat, { name: string; extensions: string[] }> = {
      glb: { name: 'GL Transmission Format', extensions: ['glb'] },
      obj: { name: 'Wavefront OBJ', extensions: ['obj'] },
      fbx: { name: 'Autodesk FBX', extensions: ['fbx'] }
    }

    const outputPath = await window.velocityAPI.showSaveDialog({
      title: `Export as ${format.toUpperCase()}`,
      filters: [filters[format], { name: 'All Files', extensions: ['*'] }]
    })

    if (!outputPath) return

    setExporting(true)
    try {
      const saved = await exportModel(currentModelPath, outputPath, format)
      showToast(`Exported to ${saved}`, 'success')
    } catch (err: any) {
      showToast(err.message ?? 'Export failed', 'error')
    } finally {
      setExporting(false)
    }
  }

  if (!currentModelPath) return null

  return (
    <div className="export-button-root">
      <div className="export-button-group">
        {(['glb', 'obj', 'fbx'] as ExportFormat[]).map((format) => (
          <button
            key={format}
            className="export-format-btn"
            onClick={() => void handleExport(format)}
            disabled={exporting}
          >
            {format}
          </button>
        ))}
      </div>

      {toast && (
        <div className={`export-toast ${toast.type}`}>
          {toast.message}
        </div>
      )}
    </div>
  )
}

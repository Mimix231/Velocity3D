import React, { useEffect, useMemo, useRef, useState } from 'react'
import * as THREE from 'three'
import { ViewportRenderer } from '../viewport/ViewportRenderer'
import { useGenerationStore } from '../stores/generationStore'
import type { ObjectProperties } from './PropertiesPanel'
import './ViewportContainer.css'

export interface ViewportTransformCommand {
  id: number
  objectId: string
  transform: {
    position?: { x: number; y: number; z: number }
    rotation?: { x: number; y: number; z: number }
    scale?: { x: number; y: number; z: number }
  }
}

interface Props {
  backendReady: boolean
  onObjectSelected?: (props: ObjectProperties | null) => void
  transformCommand?: ViewportTransformCommand | null
}

export default function ViewportContainer({ backendReady, onObjectSelected, transformCommand }: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const rendererRef = useRef<ViewportRenderer | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)
  const [loadError, setLoadError] = useState<string | null>(null)
  const currentModelPath = useGenerationStore((s) => s.currentModelPath)
  const status = useGenerationStore((s) => s.status)
  const progress = useGenerationStore((s) => s.progress)

  const currentModelLabel = useMemo(() => {
    if (!currentModelPath) return 'Scene viewport'
    const parts = currentModelPath.split(/[\\/]/)
    return parts[parts.length - 1] || 'Scene viewport'
  }, [currentModelPath])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return

    const vr = new ViewportRenderer(canvas, {
      onLoadError: (msg) => setLoadError(msg),
      onObjectSelected: (obj) => {
        if (!onObjectSelected) return
        if (!obj) {
          onObjectSelected(null)
          return
        }
        const mesh = obj as THREE.Mesh
        const geo = mesh.geometry
        onObjectSelected({
          id: obj.uuid,
          name: obj.name || 'Mesh',
          vertexCount: geo?.attributes?.position?.count ?? 0,
          faceCount: geo?.index ? geo.index.count / 3 : 0,
          position: { x: obj.position.x, y: obj.position.y, z: obj.position.z },
          rotation: { x: obj.rotation.x, y: obj.rotation.y, z: obj.rotation.z },
          scale: { x: obj.scale.x, y: obj.scale.y, z: obj.scale.z }
        })
      }
    })
    rendererRef.current = vr

    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect
        vr.resize(width, height)
      }
    })
    if (containerRef.current) ro.observe(containerRef.current)

    return () => {
      ro.disconnect()
      vr.dispose()
      rendererRef.current = null
    }
  }, [onObjectSelected])

  useEffect(() => {
    if (!currentModelPath || !rendererRef.current) return
    setLoadError(null)
    window.velocityAPI.readModelFile(currentModelPath).then((buffer) => {
      rendererRef.current?.loadModel(buffer)
    }).catch((err) => {
      setLoadError(`Failed to read model file: ${err.message}`)
    })
  }, [currentModelPath])

  useEffect(() => {
    if (!transformCommand || !rendererRef.current) return
    rendererRef.current.updateObjectTransform(transformCommand.objectId, transformCommand.transform)
  }, [transformCommand])

  return (
    <div ref={containerRef} className="viewport-shell">
      <canvas ref={canvasRef} className="viewport-canvas" />

      <div className="viewport-overlay viewport-overlay-top-left">
        <span className="viewport-chip strong">Perspective</span>
        <span className="viewport-chip">{currentModelLabel}</span>
      </div>

      <div className="viewport-overlay viewport-overlay-top-right">
        <span className="viewport-chip">Orbit</span>
        <span className="viewport-chip">Grid</span>
        <span className="viewport-chip">Meters</span>
      </div>

      <div className="viewport-overlay viewport-overlay-bottom-left">
        <span className="viewport-caption">
          {currentModelPath
            ? 'Use the inspector on the right for mesh details and transforms.'
            : 'Generate or load a model to begin working in the scene.'}
        </span>
      </div>

      {status === 'generating' && (
        <div className="viewport-loading-pill">
          <span className="viewport-loading-dot" />
          {progress ?? 'Generating...'}
        </div>
      )}

      {loadError && (
        <div className="viewport-error-overlay">
          <div className="viewport-error-card">
            <div className="viewport-error-title">Failed to load model</div>
            <div className="viewport-error-copy">{loadError}</div>
            <button onClick={() => setLoadError(null)} className="viewport-error-button">Dismiss</button>
          </div>
        </div>
      )}

      {!backendReady && !currentModelPath && (
        <div className="viewport-empty-state">
          Initializing backend...
        </div>
      )}
    </div>
  )
}

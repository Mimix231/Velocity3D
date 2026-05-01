import * as THREE from 'three'
import { RoomEnvironment } from 'three/examples/jsm/environments/RoomEnvironment.js'
import { CameraController } from './CameraController'
import { SceneHelpers } from './SceneHelpers'
import { ModelLoader, ModelLoadError } from './ModelLoader'
import { SelectionManager, ObjectSelectedCallback } from './SelectionManager'

export interface ViewportRendererOptions {
  onObjectSelected?: ObjectSelectedCallback
  onLoadError?: (message: string) => void
}

export interface ObjectTransformUpdate {
  position?: { x: number; y: number; z: number }
  rotation?: { x: number; y: number; z: number }
  scale?: { x: number; y: number; z: number }
}

export class ViewportRenderer {
  private renderer: THREE.WebGLRenderer
  private scene: THREE.Scene
  private camera: THREE.PerspectiveCamera
  private cameraController: CameraController
  private sceneHelpers: SceneHelpers
  private modelLoader: ModelLoader
  private selectionManager: SelectionManager
  private animationId: number | null = null
  private currentModel: THREE.Group | null = null
  private environment: THREE.Texture | null = null
  private options: ViewportRendererOptions

  constructor(canvas: HTMLCanvasElement, options: ViewportRendererOptions = {}) {
    this.options = options

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true })
    this.renderer.setPixelRatio(window.devicePixelRatio)
    this.renderer.setClearColor(0x1e1e1e)
    this.renderer.outputColorSpace = THREE.SRGBColorSpace
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping
    this.renderer.toneMappingExposure = 1.2
    this.renderer.shadowMap.enabled = true
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap

    this.scene = new THREE.Scene()
    const pmrem = new THREE.PMREMGenerator(this.renderer)
    this.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture
    this.scene.environment = this.environment
    pmrem.dispose()

    this.camera = new THREE.PerspectiveCamera(45, canvas.clientWidth / canvas.clientHeight, 0.01, 1000)
    this.camera.position.set(5, 5, 5)
    this.camera.lookAt(0, 0, 0)

    this.cameraController = new CameraController(this.camera, canvas)
    this.sceneHelpers = new SceneHelpers(this.scene)
    this.modelLoader = new ModelLoader()

    this.selectionManager = new SelectionManager(
      this.camera,
      this.scene,
      canvas,
      options.onObjectSelected ?? (() => {})
    )

    const ambient = new THREE.AmbientLight(0xffffff, 0.52)
    const hemi = new THREE.HemisphereLight(0xdbe7ff, 0x2b2520, 0.72)
    const keyLight = new THREE.DirectionalLight(0xfff3df, 2.15)
    keyLight.position.set(5, 8, 6)
    keyLight.castShadow = true
    keyLight.shadow.mapSize.set(2048, 2048)
    const fillLight = new THREE.DirectionalLight(0x8fb6ff, 0.78)
    fillLight.position.set(-6, 4, -5)
    const rimLight = new THREE.DirectionalLight(0xffffff, 1.1)
    rimLight.position.set(-2, 6, 7)
    this.scene.add(ambient, hemi, keyLight, fillLight, rimLight)

    this.startLoop()
  }

  private startLoop(): void {
    const animate = () => {
      this.animationId = requestAnimationFrame(animate)
      this.cameraController.update()
      this.renderer.render(this.scene, this.camera)
    }
    animate()
  }

  resize(width: number, height: number): void {
    this.renderer.setSize(width, height, false)
    this.camera.aspect = width / height
    this.camera.updateProjectionMatrix()
  }

  async loadModel(buffer: ArrayBuffer): Promise<void> {
    try {
      const model = await this.modelLoader.load(buffer)
      this.polishLoadedModel(model)

      if (this.currentModel) {
        this.scene.remove(this.currentModel)
      }

      this.currentModel = model
      this.scene.add(model)

      // Focus camera on the loaded model
      const box = new THREE.Box3().setFromObject(model)
      this.cameraController.focusOnBoundingBox(box)
      this.sceneHelpers.setVisible(true)
    } catch (err) {
      const message = err instanceof ModelLoadError ? err.message : 'Failed to load model'
      this.options.onLoadError?.(message)
    }
  }

  private polishLoadedModel(model: THREE.Group): void {
    model.traverse((object) => {
      const mesh = object as THREE.Mesh
      if (!mesh.isMesh) return

      mesh.castShadow = true
      mesh.receiveShadow = true

      const materials = Array.isArray(mesh.material) ? mesh.material : [mesh.material]
      for (const material of materials) {
        if (!material) continue

        const standard = material as THREE.MeshStandardMaterial
        if (standard.map) {
          standard.map.colorSpace = THREE.SRGBColorSpace
          standard.map.anisotropy = Math.min(8, this.renderer.capabilities.getMaxAnisotropy())
        }
        if (standard.normalMap) {
          standard.normalScale = new THREE.Vector2(0.48, 0.48)
        }
        if ('roughness' in standard && standard.roughness === undefined) {
          standard.roughness = 0.68
        }
        if ('metalness' in standard && standard.metalness === undefined) {
          standard.metalness = 0.0
        }
        standard.envMapIntensity = standard.map ? 0.82 : 0.38
        standard.side = THREE.DoubleSide
        standard.needsUpdate = true
      }
    })
  }

  clearModel(): void {
    if (this.currentModel) {
      this.scene.remove(this.currentModel)
      this.currentModel = null
    }
  }

  updateObjectTransform(uuid: string, transform: ObjectTransformUpdate): void {
    const object = this.scene.getObjectByProperty('uuid', uuid)
    if (!object) return

    if (transform.position) {
      object.position.set(transform.position.x, transform.position.y, transform.position.z)
    }
    if (transform.rotation) {
      object.rotation.set(transform.rotation.x, transform.rotation.y, transform.rotation.z)
    }
    if (transform.scale) {
      object.scale.set(
        Math.max(0.001, transform.scale.x),
        Math.max(0.001, transform.scale.y),
        Math.max(0.001, transform.scale.z)
      )
    }
    object.updateMatrixWorld(true)
    this.options.onObjectSelected?.(object)
  }

  dispose(): void {
    if (this.animationId !== null) cancelAnimationFrame(this.animationId)
    this.cameraController.dispose()
    this.sceneHelpers.dispose()
    this.selectionManager.dispose()
    this.environment?.dispose()
    this.renderer.dispose()
  }
}

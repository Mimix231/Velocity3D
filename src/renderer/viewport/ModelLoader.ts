import * as THREE from 'three'
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js'
import { validateGlb } from './glbValidator'

export class ModelLoadError extends Error {
  constructor(message: string) {
    super(message)
    this.name = 'ModelLoadError'
  }
}

export class ModelLoader {
  private loader: GLTFLoader

  constructor() {
    this.loader = new GLTFLoader()
  }

  async load(buffer: ArrayBuffer): Promise<THREE.Group> {
    const validation = validateGlb(buffer)
    if (!validation.valid) {
      throw new ModelLoadError(`Invalid GLB file: ${validation.reason}`)
    }

    return new Promise((resolve, reject) => {
      this.loader.parse(
        buffer,
        '',
        (gltf) => resolve(gltf.scene),
        (error) => reject(new ModelLoadError(`Failed to parse GLB: ${error.message}`))
      )
    })
  }
}

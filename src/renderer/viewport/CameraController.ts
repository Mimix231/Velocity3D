import * as THREE from 'three'
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js'

export class CameraController {
  private controls: OrbitControls
  readonly camera: THREE.PerspectiveCamera

  constructor(camera: THREE.PerspectiveCamera, domElement: HTMLElement) {
    this.camera = camera
    this.controls = new OrbitControls(camera, domElement)
    this.controls.enableDamping = true
    this.controls.dampingFactor = 0.05
    this.controls.screenSpacePanning = true
    this.controls.minDistance = 0.1
    this.controls.maxDistance = 500
  }

  update(): void {
    this.controls.update()
  }

  focusOnBoundingBox(box: THREE.Box3): void {
    const center = new THREE.Vector3()
    const size = new THREE.Vector3()
    box.getCenter(center)
    box.getSize(size)

    const maxDim = Math.max(size.x, size.y, size.z)
    const fov = this.camera.fov * (Math.PI / 180)
    const distance = Math.abs(maxDim / (2 * Math.tan(fov / 2))) * 1.5

    this.controls.target.copy(center)
    this.camera.position.set(center.x + distance, center.y + distance * 0.5, center.z + distance)
    this.camera.lookAt(center)
    this.controls.update()
  }

  dispose(): void {
    this.controls.dispose()
  }
}

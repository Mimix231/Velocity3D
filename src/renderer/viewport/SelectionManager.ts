import * as THREE from 'three'

export type ObjectSelectedCallback = (object: THREE.Object3D | null) => void

export class SelectionManager {
  private raycaster: THREE.Raycaster
  private camera: THREE.Camera
  private scene: THREE.Scene
  private domElement: HTMLElement
  private onObjectSelected: ObjectSelectedCallback
  private lastClickTime = 0

  constructor(
    camera: THREE.Camera,
    scene: THREE.Scene,
    domElement: HTMLElement,
    onObjectSelected: ObjectSelectedCallback
  ) {
    this.raycaster = new THREE.Raycaster()
    this.camera = camera
    this.scene = scene
    this.domElement = domElement
    this.onObjectSelected = onObjectSelected
    this.domElement.addEventListener('dblclick', this.handleDoubleClick)
  }

  private handleDoubleClick = (event: MouseEvent): void => {
    const rect = this.domElement.getBoundingClientRect()
    const x = ((event.clientX - rect.left) / rect.width) * 2 - 1
    const y = -((event.clientY - rect.top) / rect.height) * 2 + 1

    this.raycaster.setFromCamera(new THREE.Vector2(x, y), this.camera)
    const intersects = this.raycaster.intersectObjects(this.scene.children, true)

    if (intersects.length > 0) {
      this.onObjectSelected(intersects[0].object)
    } else {
      this.onObjectSelected(null)
    }
  }

  dispose(): void {
    this.domElement.removeEventListener('dblclick', this.handleDoubleClick)
  }
}

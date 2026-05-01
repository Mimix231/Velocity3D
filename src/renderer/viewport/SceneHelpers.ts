import * as THREE from 'three'

export class SceneHelpers {
  private gridHelper: THREE.GridHelper
  private axesHelper: THREE.AxesHelper

  constructor(scene: THREE.Scene) {
    this.gridHelper = new THREE.GridHelper(20, 20, 0x333333, 0x222222)
    this.axesHelper = new THREE.AxesHelper(1)
    scene.add(this.gridHelper)
    scene.add(this.axesHelper)
  }

  setVisible(visible: boolean): void {
    this.gridHelper.visible = visible
    this.axesHelper.visible = visible
  }

  dispose(): void {
    this.gridHelper.geometry.dispose()
    ;(this.gridHelper.material as THREE.Material).dispose()
    this.axesHelper.geometry.dispose()
    ;(this.axesHelper.material as THREE.Material).dispose()
  }
}

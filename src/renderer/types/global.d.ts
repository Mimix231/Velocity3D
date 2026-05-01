import type { VelocityAPI } from '../../preload/index'

declare global {
  interface Window {
    velocityAPI: VelocityAPI
  }
}

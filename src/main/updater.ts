import { BrowserWindow } from 'electron'
import { autoUpdater } from 'electron-updater'
import * as log from 'electron-log'

autoUpdater.logger = log
autoUpdater.autoDownload = false

export function initAutoUpdater(win: BrowserWindow): void {
  autoUpdater.on('update-available', (info) => {
    win.webContents.send('update-available', {
      version: info.version,
      releaseNotes: info.releaseNotes ?? undefined
    })
  })

  autoUpdater.on('error', (err) => {
    log.error('Auto-updater error:', err)
    win.webContents.send('update-error', { message: err.message })
    // App continues normally — do not throw
  })

  try {
    autoUpdater.checkForUpdates().catch((err) => {
      log.warn('Update check failed (non-fatal):', err.message)
    })
  } catch (err) {
    log.warn('Auto-updater init failed (non-fatal):', err)
  }
}

import type { ApiError } from '../api/generationApi'

export interface BackgroundRemovalResult {
  dataUrl: string
  base64: string
}

interface BackgroundRemovalApiResponse {
  image_base64: string
  mime_type?: string
}

type Rgb = [number, number, number]

function loadImage(sourceUrl: string): Promise<HTMLImageElement> {
  return new Promise((resolve, reject) => {
    const image = new Image()
    image.onload = () => resolve(image)
    image.onerror = () => reject(new Error('Could not read image for background removal.'))
    image.src = sourceUrl
  })
}

function rgbDistanceSq(a: Rgb, b: Rgb): number {
  const dr = a[0] - b[0]
  const dg = a[1] - b[1]
  const db = a[2] - b[2]
  return dr * dr + dg * dg + db * db
}

function luminance(rgb: Rgb): number {
  return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
}

function saturation(rgb: Rgb): number {
  const max = Math.max(rgb[0], rgb[1], rgb[2])
  const min = Math.min(rgb[0], rgb[1], rgb[2])
  if (max <= 0) return 0
  return (max - min) / max
}

function pixelRgb(data: Uint8ClampedArray, pixelIndex: number): Rgb {
  const offset = pixelIndex * 4
  return [data[offset], data[offset + 1], data[offset + 2]]
}

function median(values: number[]): number {
  const sorted = [...values].sort((a, b) => a - b)
  return sorted[Math.floor(sorted.length / 2)] ?? 0
}

function scaledImageDataUrl(sourceUrl: string, maxSize: number): Promise<string> {
  return loadImage(sourceUrl).then((image) => {
    const scale = Math.min(1, maxSize / Math.max(image.naturalWidth, image.naturalHeight))
    const width = Math.max(1, Math.round(image.naturalWidth * scale))
    const height = Math.max(1, Math.round(image.naturalHeight * scale))
    const canvas = document.createElement('canvas')
    const ctx = canvas.getContext('2d', { willReadFrequently: true })

    if (!ctx) {
      throw new Error('Canvas is unavailable for background removal.')
    }

    canvas.width = width
    canvas.height = height
    ctx.drawImage(image, 0, 0, width, height)
    return canvas.toDataURL('image/png')
  })
}

function dataUrlBase64(dataUrl: string): string {
  return dataUrl.split(',')[1] ?? ''
}

function parseRemovalError(payload: ApiError | string, status: number): Error {
  if (typeof payload === 'string') {
    return new Error(payload || `HTTP ${status}`)
  }
  return new Error(payload.details ?? payload.error ?? `HTTP ${status}`)
}

async function removeImageBackgroundBackend(sourceUrl: string, maxSize: number): Promise<BackgroundRemovalResult> {
  const dataUrl = await scaledImageDataUrl(sourceUrl, maxSize)
  const res = await window.velocityAPI.backendRequest<BackgroundRemovalApiResponse | ApiError>({
    path: '/images/remove-background',
    method: 'POST',
    body: { image_base64: dataUrlBase64(dataUrl) }
  })

  if (!res.ok) {
    throw parseRemovalError(res.data as ApiError, res.status)
  }

  const payload = res.data as BackgroundRemovalApiResponse
  if (!payload.image_base64) {
    throw new Error('Background removal returned an empty image.')
  }

  const mimeType = payload.mime_type || 'image/png'
  return {
    dataUrl: `data:${mimeType};base64,${payload.image_base64}`,
    base64: payload.image_base64
  }
}

async function removeImageBackgroundCanvasFallback(sourceUrl: string, maxSize: number): Promise<BackgroundRemovalResult> {
  const image = await loadImage(sourceUrl)
  const scale = Math.min(1, maxSize / Math.max(image.naturalWidth, image.naturalHeight))
  const width = Math.max(1, Math.round(image.naturalWidth * scale))
  const height = Math.max(1, Math.round(image.naturalHeight * scale))
  const canvas = document.createElement('canvas')
  const ctx = canvas.getContext('2d', { willReadFrequently: true })

  if (!ctx) {
    throw new Error('Canvas is unavailable for background removal.')
  }

  canvas.width = width
  canvas.height = height
  ctx.drawImage(image, 0, 0, width, height)

  const imageData = ctx.getImageData(0, 0, width, height)
  const data = imageData.data
  const borderPixels: Rgb[] = []
  const sampleStride = Math.max(1, Math.floor(Math.max(width, height) / 220))

  for (let x = 0; x < width; x += sampleStride) {
    borderPixels.push(pixelRgb(data, x))
    borderPixels.push(pixelRgb(data, (height - 1) * width + x))
  }
  for (let y = 0; y < height; y += sampleStride) {
    borderPixels.push(pixelRgb(data, y * width))
    borderPixels.push(pixelRgb(data, y * width + width - 1))
  }

  const bg: Rgb = [
    median(borderPixels.map((pixel) => pixel[0])),
    median(borderPixels.map((pixel) => pixel[1])),
    median(borderPixels.map((pixel) => pixel[2]))
  ]
  const bgLuma = luminance(bg)
  const bgSaturation = saturation(bg)

  const visited = new Uint8Array(width * height)
  const queue: number[] = []
  const threshold = bgLuma < 55 && bgSaturation < 0.2 ? 96 : 72
  const relaxedThreshold = threshold + 28
  const thresholdSq = threshold * threshold
  const relaxedThresholdSq = relaxedThreshold * relaxedThreshold

  const backgroundLike = (pixelIndex: number, relaxed = false) => {
    const rgb = pixelRgb(data, pixelIndex)
    const distanceSq = rgbDistanceSq(rgb, bg)
    if (distanceSq <= (relaxed ? relaxedThresholdSq : thresholdSq)) return true

    const luma = luminance(rgb)
    const sat = saturation(rgb)
    const isDarkNeutral = bgLuma < 70 && luma < Math.max(92, bgLuma + 50) && sat < 0.24
    if (isDarkNeutral && distanceSq <= relaxedThresholdSq * 1.45) return true

    const isStudioShadow = bgLuma < 75 && luma < 115 && sat < 0.32 && Math.abs(luma - bgLuma) < 72
    return relaxed && isStudioShadow
  }

  const enqueueIfBackground = (x: number, y: number) => {
    if (x < 0 || y < 0 || x >= width || y >= height) return
    const pixelIndex = y * width + x
    if (visited[pixelIndex]) return
    if (!backgroundLike(pixelIndex)) return
    visited[pixelIndex] = 1
    queue.push(pixelIndex)
  }

  for (let x = 0; x < width; x += 1) {
    enqueueIfBackground(x, 0)
    enqueueIfBackground(x, height - 1)
  }
  for (let y = 0; y < height; y += 1) {
    enqueueIfBackground(0, y)
    enqueueIfBackground(width - 1, y)
  }

  let queueIndex = 0
  while (queueIndex < queue.length) {
    const pixelIndex = queue[queueIndex]
    queueIndex += 1
    const x = pixelIndex % width
    const y = Math.floor(pixelIndex / width)
    enqueueIfBackground(x + 1, y)
    enqueueIfBackground(x - 1, y)
    enqueueIfBackground(x, y + 1)
    enqueueIfBackground(x, y - 1)
  }

  const hasVisitedNeighbor = (index: number) => {
    const x = index % width
    const y = Math.floor(index / width)
    return (
      (x > 0 && visited[index - 1]) ||
      (x < width - 1 && visited[index + 1]) ||
      (y > 0 && visited[index - width]) ||
      (y < height - 1 && visited[index + width])
    )
  }

  for (let pass = 0; pass < 5; pass += 1) {
    const toRemove: number[] = []
    for (let index = 0; index < visited.length; index += 1) {
      if (visited[index] || !hasVisitedNeighbor(index)) continue
      if (backgroundLike(index, true)) {
        toRemove.push(index)
      }
    }
    if (toRemove.length === 0) break
    for (const index of toRemove) {
      visited[index] = 1
    }
  }

  for (let index = 0; index < visited.length; index += 1) {
    const alphaOffset = index * 4 + 3
    if (visited[index]) {
      data[alphaOffset] = 0
      continue
    }

    if (hasVisitedNeighbor(index) && backgroundLike(index, true)) {
      data[alphaOffset] = Math.min(data[alphaOffset], 92)
    }
  }

  ctx.putImageData(imageData, 0, 0)
  const dataUrl = canvas.toDataURL('image/png')
  return {
    dataUrl,
    base64: dataUrlBase64(dataUrl)
  }
}

export async function removeImageBackground(sourceUrl: string, maxSize = 1024): Promise<BackgroundRemovalResult> {
  try {
    return await removeImageBackgroundBackend(sourceUrl, maxSize)
  } catch (err) {
    console.warn('Backend rembg cutout failed; using canvas fallback.', err)
    return removeImageBackgroundCanvasFallback(sourceUrl, maxSize)
  }
}

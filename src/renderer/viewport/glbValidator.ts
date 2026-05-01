// GLB files start with ASCII bytes "glTF". DataView reads uint32 values
// little-endian below, so the numeric value is byte-reversed.
const GLB_MAGIC_LE = 0x46546c67
const GLB_MIN_LENGTH = 12 // header only

export interface GlbValidationResult {
  valid: boolean
  reason?: string
}

/**
 * Validates that a buffer contains a well-formed GLB file.
 * Checks: magic bytes, non-zero length, valid JSON chunk header.
 */
export function validateGlb(buffer: ArrayBuffer): GlbValidationResult {
  if (buffer.byteLength < GLB_MIN_LENGTH) {
    return { valid: false, reason: `Buffer too small: ${buffer.byteLength} bytes` }
  }

  const view = new DataView(buffer)

  // Check magic bytes (little-endian uint32 at offset 0)
  const magic = view.getUint32(0, true)
  if (magic !== GLB_MAGIC_LE) {
    return { valid: false, reason: `Invalid magic bytes: 0x${magic.toString(16)}` }
  }

  // Check total length field matches buffer size
  const totalLength = view.getUint32(8, true)
  if (totalLength === 0) {
    return { valid: false, reason: 'Total length field is zero' }
  }
  if (totalLength > buffer.byteLength) {
    return { valid: false, reason: `Declared length ${totalLength} exceeds buffer size ${buffer.byteLength}` }
  }

  // Check first chunk (JSON chunk) exists
  if (buffer.byteLength < 20) {
    return { valid: false, reason: 'Buffer too small to contain JSON chunk header' }
  }

  const jsonChunkLength = view.getUint32(12, true)
  const jsonChunkType = view.getUint32(16, true)

  // JSON chunk type must be 0x4E4F534A ("JSON")
  if (jsonChunkType !== 0x4e4f534a) {
    return { valid: false, reason: `First chunk is not JSON type: 0x${jsonChunkType.toString(16)}` }
  }

  if (jsonChunkLength === 0) {
    return { valid: false, reason: 'JSON chunk length is zero' }
  }

  return { valid: true }
}

export interface ValidationResult {
  valid: boolean
  error?: string
}

const ALLOWED_IMAGE_EXTENSIONS = new Set(['jpg', 'jpeg', 'png', 'webp', 'bmp'])

/**
 * Validates a text prompt. Rejects empty strings and whitespace-only strings.
 */
export function validatePrompt(s: string): ValidationResult {
  if (!s || !s.trim()) {
    return { valid: false, error: 'Prompt cannot be empty or whitespace.' }
  }
  return { valid: true }
}

/**
 * Validates an image filename by checking its extension (case-insensitive).
 * Accepts: jpg, jpeg, png, webp, bmp
 */
export function validateImageFile(filename: string): ValidationResult {
  const parts = filename.split('.')
  if (parts.length < 2) {
    return { valid: false, error: 'File has no extension.' }
  }
  const ext = parts[parts.length - 1].toLowerCase()
  if (!ALLOWED_IMAGE_EXTENSIONS.has(ext)) {
    return {
      valid: false,
      error: `Unsupported file type ".${ext}". Allowed: ${[...ALLOWED_IMAGE_EXTENSIONS].join(', ')}`
    }
  }
  return { valid: true }
}

import { create } from 'zustand'

export type ThemeCategoryId = 'core' | 'fruit' | 'videogame' | 'anime' | 'nature'

export interface ThemeCategory {
  id: ThemeCategoryId
  name: string
}

export interface Theme {
  id: string
  name: string
  category: ThemeCategoryId
  bg: string
  bgAlt: string
  bgPanel: string
  border: string
  text: string
  textMuted: string
  accent: string
  accentHover: string
  error: string
  success: string
}

export const THEME_CATEGORIES: ThemeCategory[] = [
  { id: 'core', name: 'Core' },
  { id: 'fruit', name: 'Fruit' },
  { id: 'videogame', name: 'Videogame' },
  { id: 'anime', name: 'Anime' },
  { id: 'nature', name: 'Nature' },
]

export const THEMES = {
  midnight: {
    id: 'midnight', name: 'Midnight Studio', category: 'core',
    bg: '#0b0d12', bgAlt: '#11141d', bgPanel: '#171924',
    border: '#2a2e3d', text: '#edf0f7', textMuted: '#8290bd',
    accent: '#8b7cf6', accentHover: '#a696ff',
    error: '#ff6678', success: '#46e89a',
  },
  obsidian: {
    id: 'obsidian', name: 'Obsidian Glass', category: 'core',
    bg: '#090909', bgAlt: '#131313', bgPanel: '#19191c',
    border: '#303039', text: '#f4f4f7', textMuted: '#8b8b99',
    accent: '#d7d7e0', accentHover: '#ffffff',
    error: '#ff5d5d', success: '#7ee787',
  },
  light: {
    id: 'light', name: 'Soft Light', category: 'core',
    bg: '#f4f6fb', bgAlt: '#e7ebf3', bgPanel: '#ffffff',
    border: '#cfd6e6', text: '#161b28', textMuted: '#69748c',
    accent: '#4667f2', accentHover: '#3354d7',
    error: '#d83a52', success: '#178c58',
  },
  citrus: {
    id: 'citrus', name: 'Citrus Pop', category: 'fruit',
    bg: '#11120b', bgAlt: '#1b1d10', bgPanel: '#232515',
    border: '#404829', text: '#fff9d7', textMuted: '#b8b075',
    accent: '#e7f743', accentHover: '#f4ff70',
    error: '#ff6b4a', success: '#6ef56f',
  },
  watermelon: {
    id: 'watermelon', name: 'Watermelon Slice', category: 'fruit',
    bg: '#111417', bgAlt: '#18201d', bgPanel: '#201a22',
    border: '#3f3641', text: '#fff2f4', textMuted: '#b88796',
    accent: '#ff5d7a', accentHover: '#ff7e96',
    error: '#ff3e55', success: '#5ff28b',
  },
  blueberry: {
    id: 'blueberry', name: 'Blueberry Frost', category: 'fruit',
    bg: '#0b1020', bgAlt: '#121a31', bgPanel: '#171d3b',
    border: '#2f3e70', text: '#edf2ff', textMuted: '#8899d1',
    accent: '#7aa2ff', accentHover: '#9dbaff',
    error: '#ff6b9b', success: '#72e6d0',
  },
  dragonfruit: {
    id: 'dragonfruit', name: 'Dragonfruit Glow', category: 'fruit',
    bg: '#130b17', bgAlt: '#211022', bgPanel: '#2a172e',
    border: '#553057', text: '#fff0fb', textMuted: '#c58abd',
    accent: '#ff5bd8', accentHover: '#ff83e2',
    error: '#ff5a74', success: '#7cffb2',
  },
  arcade: {
    id: 'arcade', name: '8-Bit Arcade', category: 'videogame',
    bg: '#060811', bgAlt: '#111225', bgPanel: '#171832',
    border: '#303565', text: '#f1f5ff', textMuted: '#7f8ac8',
    accent: '#00e5ff', accentHover: '#60f0ff',
    error: '#ff3d81', success: '#6dff4e',
  },
  bossFight: {
    id: 'bossFight', name: 'Boss Fight', category: 'videogame',
    bg: '#100d0f', bgAlt: '#1d1417', bgPanel: '#25191d',
    border: '#4b3137', text: '#fff2e8', textMuted: '#b88b84',
    accent: '#ff9f1c', accentHover: '#ffba52',
    error: '#ff375f', success: '#36e68a',
  },
  manaCrystal: {
    id: 'manaCrystal', name: 'Mana Crystal', category: 'videogame',
    bg: '#071218', bgAlt: '#0e2026', bgPanel: '#122a31',
    border: '#235564', text: '#e9fbff', textMuted: '#76aeba',
    accent: '#2fffd4', accentHover: '#73ffe3',
    error: '#ff5e8a', success: '#9cff5e',
  },
  sakura: {
    id: 'sakura', name: 'Sakura Frame', category: 'anime',
    bg: '#130f18', bgAlt: '#211827', bgPanel: '#291d32',
    border: '#51354f', text: '#fff3fb', textMuted: '#c894b9',
    accent: '#ff9ad5', accentHover: '#ffb7e1',
    error: '#ff6678', success: '#8ef0c2',
  },
  mecha: {
    id: 'mecha', name: 'Mecha Terminal', category: 'anime',
    bg: '#0a0e12', bgAlt: '#101921', bgPanel: '#15212b',
    border: '#284254', text: '#eef8ff', textMuted: '#80a5ba',
    accent: '#ff4f7b', accentHover: '#ff7598',
    error: '#ff445f', success: '#41f2cb',
  },
  magicalDusk: {
    id: 'magicalDusk', name: 'Magical Dusk', category: 'anime',
    bg: '#0f0b1d', bgAlt: '#18112d', bgPanel: '#21183a',
    border: '#41346b', text: '#f5efff', textMuted: '#9d91ca',
    accent: '#b877ff', accentHover: '#ca9aff',
    error: '#ff6693', success: '#81f0ff',
  },
  forest: {
    id: 'forest', name: 'Forest Canopy', category: 'nature',
    bg: '#0b110d', bgAlt: '#121c15', bgPanel: '#18241b',
    border: '#304432', text: '#eef8e8', textMuted: '#8cab83',
    accent: '#7bd45a', accentHover: '#9be47e',
    error: '#ef6b5c', success: '#90f172',
  },
  ocean: {
    id: 'ocean', name: 'Ocean Reef', category: 'nature',
    bg: '#061018', bgAlt: '#0b1c29', bgPanel: '#102433',
    border: '#214a61', text: '#ecfbff', textMuted: '#82aec4',
    accent: '#33c7ff', accentHover: '#6bd8ff',
    error: '#ff6b85', success: '#45e8b5',
  },
  aurora: {
    id: 'aurora', name: 'Aurora Field', category: 'nature',
    bg: '#080f18', bgAlt: '#101c25', bgPanel: '#172531',
    border: '#2c4a56', text: '#ecfff8', textMuted: '#8db8aa',
    accent: '#67ffb1', accentHover: '#91ffc7',
    error: '#ff6686', success: '#67ffb1',
  },
} as const satisfies Record<string, Theme>

export type ThemeId = keyof typeof THEMES

interface ThemeState {
  themeId: ThemeId
  theme: Theme
  setTheme: (id: ThemeId) => void
}

function isThemeId(value: string): value is ThemeId {
  return value in THEMES
}

function loadSavedTheme(): ThemeId {
  try {
    const saved = localStorage.getItem('velocity3d-theme')
    if (saved && isThemeId(saved)) return saved
  } catch {}
  return 'midnight'
}

export const useThemeStore = create<ThemeState>((set) => {
  const savedTheme = loadSavedTheme()
  return {
    themeId: savedTheme,
    theme: THEMES[savedTheme],
    setTheme: (id) => {
      localStorage.setItem('velocity3d-theme', id)
      set({ themeId: id, theme: THEMES[id] })
      applyThemeToCss(THEMES[id])
    },
  }
})

export function applyThemeToCss(theme: Theme) {
  const root = document.documentElement
  root.style.setProperty('--bg', theme.bg)
  root.style.setProperty('--bg-alt', theme.bgAlt)
  root.style.setProperty('--bg-panel', theme.bgPanel)
  root.style.setProperty('--border', theme.border)
  root.style.setProperty('--text', theme.text)
  root.style.setProperty('--text-muted', theme.textMuted)
  root.style.setProperty('--accent', theme.accent)
  root.style.setProperty('--accent-hover', theme.accentHover)
  root.style.setProperty('--error', theme.error)
  root.style.setProperty('--success', theme.success)
}

// Client for the inference API in app/server.py. Vite proxies /api to :8000.
// Every authenticated call carries the Supabase access token; the backend
// verifies it and scopes each query to that user.

import { accessToken } from './supabase'

const BASE = '/api'

async function readError(res) {
  try {
    const body = await res.json()
    if (body?.detail) return body.detail
  } catch {
    /* fall through to the status text */
  }
  return `${res.status} ${res.statusText}`
}

async function authed(path, options = {}) {
  const token = await accessToken()
  const headers = { ...(options.headers || {}) }
  if (token) headers.Authorization = `Bearer ${token}`

  const res = await fetch(`${BASE}${path}`, { ...options, headers })
  if (!res.ok) throw new Error(await readError(res))
  return res.json()
}

// --- unauthenticated -------------------------------------------------------

export async function checkHealth() {
  try {
    const res = await fetch(`${BASE}/health`)
    if (!res.ok) return { online: false }
    const data = await res.json()
    return {
      online: true,
      device: data.device,
      modelsLoaded: data.modelsLoaded,
      storage: data.storage,
    }
  } catch {
    return { online: false }
  }
}

export async function fetchModels() {
  const res = await fetch(`${BASE}/models`)
  if (!res.ok) throw new Error(await readError(res))
  return res.json()
}

// --- records ---------------------------------------------------------------

export function analyzeImage(file, patientName = '') {
  const form = new FormData()
  form.append('file', file)
  form.append('patient_name', patientName)
  return authed('/analyze', { method: 'POST', body: form })
}

/** Returns {persisted, records}. Omits image payloads for speed. */
export function fetchRecords() {
  return authed('/records')
}

/** Full record including per-architecture predictions and image URLs. */
export function fetchRecord(id) {
  return authed(`/records/${encodeURIComponent(id)}`)
}

export function saveNotes(id, notes) {
  return authed(`/records/${encodeURIComponent(id)}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ notes }),
  })
}

export function deleteRecordRemote(id) {
  return authed(`/records/${encodeURIComponent(id)}`, { method: 'DELETE' })
}

// --- profile ---------------------------------------------------------------

export function fetchProfile() {
  return authed('/profile')
}

export function saveProfile(patch) {
  return authed('/profile', {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(patch),
  })
}

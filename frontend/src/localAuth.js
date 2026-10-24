// Drop-in for the parts of supabase.auth that AppContext and api.js use, backed
// by the SQLite API (app/db_sqlite.py). Only active in `vite --mode sqlite`.

const KEY = 'astrascan-local-session'
const listeners = new Set()

function load() {
  try {
    return JSON.parse(localStorage.getItem(KEY)) || null
  } catch {
    return null
  }
}

function store(session) {
  try {
    if (session) localStorage.setItem(KEY, JSON.stringify(session))
    else localStorage.removeItem(KEY)
  } catch {
    /* private mode: the session just won't survive a reload */
  }
  listeners.forEach((fn) => fn('SIGNED_IN', session))
}

async function post(path, body, token) {
  const res = await fetch(`/api/auth/${path}`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
    },
    body: JSON.stringify(body || {}),
  })
  const data = await res.json().catch(() => ({}))
  if (!res.ok) throw new Error(data.detail || `${res.status} ${res.statusText}`)
  return data
}

export const localAuth = {
  async getSession() {
    return { data: { session: load() } }
  },

  onAuthStateChange(fn) {
    listeners.add(fn)
    return { data: { subscription: { unsubscribe: () => listeners.delete(fn) } } }
  },

  async signInWithPassword({ email, password }) {
    try {
      const res = await post('login', { email, password })
      store({ access_token: res.accessToken, user: res.user })
      return { error: null }
    } catch (err) {
      return { error: err }
    }
  },

  async signUp({ email, password, options }) {
    try {
      await post('signup', { email, password, fullName: options?.data?.full_name || '' })
      return { error: null }
    } catch (err) {
      return { error: err }
    }
  },

  async signOut() {
    const session = load()
    if (session) await post('logout', {}, session.access_token).catch(() => {})
    store(null)
    listeners.forEach((fn) => fn('SIGNED_OUT', null))
    return { error: null }
  },
}

import { createClient } from '@supabase/supabase-js'
import { localAuth } from './localAuth'

// The ANON key is meant to be public -- it ships in the browser bundle and is
// constrained by Row Level Security. The service_role key must never appear
// here; it lives only in the backend's .env (see app/db.js -> app/db.py).
const url = import.meta.env.VITE_SUPABASE_URL
const anonKey = import.meta.env.VITE_SUPABASE_ANON_KEY

// `npm run dev:local` (vite --mode sqlite): auth goes to the local SQLite API
// instead of Supabase. localAuth mimics the slice of supabase.auth the app uses.
export const isLocalDb = import.meta.env.VITE_LOCAL_DB === '1'

export const isSupabaseConfigured = isLocalDb || Boolean(url && anonKey)

if (!isSupabaseConfigured) {
  console.warn(
    'Supabase is not configured. Copy frontend/.env.example to frontend/.env ' +
      'and set VITE_SUPABASE_URL and VITE_SUPABASE_ANON_KEY.',
  )
}

export const supabase = isLocalDb
  ? { auth: localAuth }
  : isSupabaseConfigured
    ? createClient(url, anonKey, {
        auth: { persistSession: true, autoRefreshToken: true },
      })
    : null

/** Bearer token for the API, or '' when signed out. */
export async function accessToken() {
  if (!supabase) return ''
  const { data } = await supabase.auth.getSession()
  return data.session?.access_token || ''
}

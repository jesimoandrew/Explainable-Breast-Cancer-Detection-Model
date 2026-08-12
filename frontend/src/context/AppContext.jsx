import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
} from 'react'
import { supabase, isSupabaseConfigured } from '../supabase'
import {
  deleteRecordRemote,
  fetchProfile,
  fetchRecord,
  fetchRecords,
  saveNotes,
  saveProfile,
} from '../api'

const AppContext = createContext(null)

const EMPTY_PROFILE = {
  fullName: '',
  email: '',
  role: '',
  medicalId: '',
}

export function AppProvider({ children }) {
  const [session, setSession] = useState(null)
  const [authReady, setAuthReady] = useState(!isSupabaseConfigured)
  const [profile, setProfileState] = useState(EMPTY_PROFILE)
  const [records, setRecords] = useState([])
  const [loadingRecords, setLoadingRecords] = useState(false)

  const isAuthenticated = Boolean(session)

  // Restore any persisted session, then track sign-in/sign-out.
  useEffect(() => {
    if (!supabase) return undefined
    supabase.auth.getSession().then(({ data }) => {
      setSession(data.session)
      setAuthReady(true)
    })
    const { data: sub } = supabase.auth.onAuthStateChange((_event, next) => {
      setSession(next)
    })
    return () => sub.subscription.unsubscribe()
  }, [])

  // Load the clinician's archive and profile once signed in.
  useEffect(() => {
    if (!isAuthenticated) {
      setRecords([])
      setProfileState(EMPTY_PROFILE)
      return undefined
    }
    let alive = true
    setLoadingRecords(true)

    fetchRecords()
      .then((data) => alive && setRecords(data.records || []))
      .catch(() => alive && setRecords([]))
      .finally(() => alive && setLoadingRecords(false))

    fetchProfile()
      .then((p) => alive && setProfileState(p))
      .catch(() => {
        // Fall back to the email on the session if the profile row is missing.
        if (alive) {
          setProfileState((prev) => ({
            ...prev,
            email: session?.user?.email || '',
          }))
        }
      })

    return () => {
      alive = false
    }
  }, [isAuthenticated, session?.user?.email])

  const login = useCallback(async (email, password) => {
    if (!supabase) throw new Error('Supabase is not configured.')
    const { error } = await supabase.auth.signInWithPassword({ email, password })
    if (error) throw new Error(error.message)
  }, [])

  const signUp = useCallback(async (email, password, fullName) => {
    if (!supabase) throw new Error('Supabase is not configured.')
    const { error } = await supabase.auth.signUp({
      email,
      password,
      options: { data: { full_name: fullName || '' } },
    })
    if (error) throw new Error(error.message)
  }, [])

  const logout = useCallback(async () => {
    if (supabase) await supabase.auth.signOut()
    setSession(null)
  }, [])

  const setProfile = useCallback(async (patch) => {
    setProfileState((prev) => ({ ...prev, ...patch }))
    try {
      const saved = await saveProfile(patch)
      if (saved) setProfileState(saved)
    } catch {
      /* keep the optimistic value; the next load reconciles */
    }
  }, [])

  const getRecord = useCallback(
    (id) => records.find((r) => r.id === id),
    [records],
  )

  /** The list endpoint omits per-model detail; pull the full row when opening. */
  const hydrateRecord = useCallback(async (id) => {
    try {
      const full = await fetchRecord(id)
      setRecords((prev) => prev.map((r) => (r.id === id ? { ...r, ...full } : r)))
    } catch {
      /* keep whatever the list gave us */
    }
  }, [])

  const updateRecord = useCallback((id, patch) => {
    setRecords((prev) => prev.map((r) => (r.id === id ? { ...r, ...patch } : r)))
    if (patch.notes !== undefined) saveNotes(id, patch.notes).catch(() => {})
  }, [])

  const deleteRecord = useCallback((id) => {
    setRecords((prev) => prev.filter((r) => r.id !== id))
    deleteRecordRemote(id).catch(() => {})
  }, [])

  const addRecord = useCallback((record) => {
    setRecords((prev) => [record, ...prev])
  }, [])

  const value = useMemo(
    () => ({
      isAuthenticated,
      authReady,
      session,
      login,
      signUp,
      logout,
      profile,
      setProfile,
      records,
      loadingRecords,
      getRecord,
      hydrateRecord,
      updateRecord,
      deleteRecord,
      addRecord,
    }),
    [
      isAuthenticated, authReady, session, login, signUp, logout,
      profile, setProfile, records, loadingRecords, getRecord,
      hydrateRecord, updateRecord, deleteRecord, addRecord,
    ],
  )

  return <AppContext.Provider value={value}>{children}</AppContext.Provider>
}

export function useApp() {
  const ctx = useContext(AppContext)
  if (!ctx) throw new Error('useApp must be used inside <AppProvider>')
  return ctx
}

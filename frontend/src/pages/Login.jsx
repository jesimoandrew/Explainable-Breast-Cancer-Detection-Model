import { useEffect, useState } from 'react'
import { useLocation, useNavigate } from 'react-router-dom'
import { useApp } from '../context/AppContext'
import { isSupabaseConfigured } from '../supabase'
import { checkHealth } from '../api'
import Footer from '../components/Footer'
import {
  AlertTriangle,
  ArrowRight,
  Hospital,
  Lock,
  LogoMark,
  Mail,
  Spinner,
  User,
} from '../components/Icons'

export default function Login() {
  const { login, signUp } = useApp()
  const navigate = useNavigate()
  const location = useLocation()

  const [tab, setTab] = useState('login')
  const [fullName, setFullName] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [notice, setNotice] = useState(null)
  const [engine, setEngine] = useState({ online: null })

  const from = location.state?.from?.pathname || '/dashboard'

  useEffect(() => {
    let alive = true
    checkHealth().then((h) => alive && setEngine(h))
    return () => {
      alive = false
    }
  }, [])

  async function handleSubmit(e) {
    e.preventDefault()
    if (busy) return
    setBusy(true)
    setError(null)
    setNotice(null)
    try {
      if (tab === 'login') {
        await login(email, password)
        navigate(from, { replace: true })
      } else {
        await signUp(email, password, fullName)
        // With email confirmations on, there is no session yet.
        setNotice('Account created. Check your inbox to confirm, then sign in.')
        setTab('login')
      }
    } catch (err) {
      setError(err.message || 'Authentication failed.')
    } finally {
      setBusy(false)
    }
  }

  function switchTab(next) {
    setTab(next)
    setError(null)
    setNotice(null)
  }

  return (
    <div className="auth-shell">
      <div className="auth-main">
        <div className="auth-card">
          <div className="auth-card__head">
            <span className="brand__mark brand__mark--lg">
              <LogoMark size={30} />
            </span>
            <h1 className="auth-card__title">AstraScan AI</h1>
            <p className="auth-card__eyebrow">Medical Intelligence Portal</p>
          </div>

          <div className="segmented" role="tablist" aria-label="Account access">
            <button
              type="button"
              role="tab"
              aria-selected={tab === 'login'}
              className={tab === 'login' ? 'is-active' : undefined}
              onClick={() => switchTab('login')}
            >
              Login
            </button>
            <button
              type="button"
              role="tab"
              aria-selected={tab === 'signup'}
              className={tab === 'signup' ? 'is-active' : undefined}
              onClick={() => switchTab('signup')}
            >
              Sign Up
            </button>
          </div>

          <form onSubmit={handleSubmit} className="auth-form">
            {tab === 'signup' && (
              <div className="field">
                <label className="field__label" htmlFor="fullName">
                  Full Name
                </label>
                <div className="input-wrap">
                  <span className="input-wrap__icon">
                    <User size={16} />
                  </span>
                  <input
                    id="fullName"
                    value={fullName}
                    required
                    onChange={(e) => setFullName(e.target.value)}
                    placeholder="Dr. Sarah Chen"
                  />
                </div>
              </div>
            )}

            <div className="field">
              <label className="field__label" htmlFor="email">
                Clinician Email
              </label>
              <div className="input-wrap">
                <span className="input-wrap__icon">
                  <Mail size={16} />
                </span>
                <input
                  id="email"
                  type="email"
                  value={email}
                  required
                  autoComplete="email"
                  onChange={(e) => setEmail(e.target.value)}
                  placeholder="dr.smith@hospital.com"
                />
              </div>
            </div>

            <div className="field">
              <div className="field__row">
                <label className="field__label" htmlFor="password">
                  Password
                </label>
                {tab === 'login' && (
                  <a
                    className="field__aside"
                    href="#"
                    onClick={(e) => {
                      e.preventDefault()
                      setNotice(
                        'Password resets are managed in the Supabase dashboard for now.',
                      )
                    }}
                  >
                    Forgot Password?
                  </a>
                )}
              </div>
              <div className="input-wrap">
                <span className="input-wrap__icon">
                  <Lock size={16} />
                </span>
                <input
                  id="password"
                  type="password"
                  value={password}
                  required
                  minLength={6}
                  autoComplete={tab === 'login' ? 'current-password' : 'new-password'}
                  onChange={(e) => setPassword(e.target.value)}
                  placeholder="••••••••"
                />
              </div>
            </div>

            {error && (
              <p className="auth-alert auth-alert--error">
                <AlertTriangle size={14} />
                {error}
              </p>
            )}
            {notice && <p className="auth-alert">{notice}</p>}

            <button
              type="submit"
              className="btn btn--primary btn--block"
              disabled={busy || !isSupabaseConfigured}
            >
              {busy && <Spinner size={17} />}
              {tab === 'login' ? 'Access System' : 'Create Account'}
              {!busy && <ArrowRight size={17} />}
            </button>
          </form>

          <div className="divider">
            <span>Institutional SSO</span>
          </div>

          <button
            type="button"
            className="btn btn--outline btn--block"
            onClick={() =>
              setNotice('Hospital ID sign-in is not enabled on this project yet.')
            }
          >
            <Hospital size={17} />
            Login with Hospital ID
          </button>

          {!isSupabaseConfigured && (
            <p className="auth-alert auth-alert--error auth-alert--spaced">
              <AlertTriangle size={14} />
              Supabase is not configured. Add VITE_SUPABASE_URL and
              VITE_SUPABASE_ANON_KEY to frontend/.env.
            </p>
          )}
        </div>

        <div className="auth-status">
          <span className={engine.online ? 'status-dot' : 'status-dot is-off'} />
          {engine.online === null && 'AI Diagnostic Engine: Connecting…'}
          {engine.online === true &&
            `AI Diagnostic Engine: Online (${engine.device}) & Encrypted`}
          {engine.online === false && 'AI Diagnostic Engine: Offline'}
        </div>
        <p className="auth-legal">
          Compliant with HIPAA, GDPR, and DICOM Standards. For
          <br />
          clinical use by certified professionals only.
        </p>
      </div>

      <Footer brand="AstraScan" />
    </div>
  )
}

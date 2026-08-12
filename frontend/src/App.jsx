import { useEffect } from 'react'
import {
  Navigate,
  Route,
  Routes,
  useLocation,
} from 'react-router-dom'
import { useApp } from './context/AppContext'
import Layout from './components/Layout'
import Login from './pages/Login'
import Dashboard from './pages/Dashboard'
import PatientRecords from './pages/PatientRecords'
import RecordDetails from './pages/RecordDetails'
import AiAnalysis from './pages/AiAnalysis'
import Settings from './pages/Settings'

function ScrollToTop() {
  const { pathname } = useLocation()
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [pathname])
  return null
}

function RequireAuth({ children }) {
  const { isAuthenticated, authReady } = useApp()
  const location = useLocation()
  // Supabase restores a persisted session asynchronously -- redirecting before
  // that resolves would sign the user out on every refresh.
  if (!authReady) return <div className="route-loading">Restoring session…</div>
  if (!isAuthenticated) {
    return <Navigate to="/login" replace state={{ from: location }} />
  }
  return children
}

export default function App() {
  const { isAuthenticated, authReady } = useApp()

  return (
    <>
      <ScrollToTop />
      <Routes>
        <Route
          path="/login"
          element={
            authReady && isAuthenticated ? (
              <Navigate to="/dashboard" replace />
            ) : (
              <Login />
            )
          }
        />

        <Route
          element={
            <RequireAuth>
              <Layout />
            </RequireAuth>
          }
        >
          <Route path="/dashboard" element={<Dashboard />} />
          <Route path="/records" element={<PatientRecords />} />
          <Route path="/records/:id" element={<RecordDetails />} />
          <Route path="/analysis" element={<AiAnalysis />} />
          <Route path="/settings" element={<Navigate to="/settings/profile" replace />} />
          <Route path="/settings/:tab" element={<Settings />} />
        </Route>

        <Route path="/" element={<Navigate to="/dashboard" replace />} />
        <Route path="*" element={<Navigate to="/dashboard" replace />} />
      </Routes>
    </>
  )
}

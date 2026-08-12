import { NavLink, useNavigate } from 'react-router-dom'
import { useApp } from '../context/AppContext'
import { Bell, HelpCircle, LogoMark, User } from './Icons'

const NAV = [
  { to: '/dashboard', label: 'Dashboard' },
  { to: '/records', label: 'Patient Records' },
  { to: '/analysis', label: 'AI Analysis' },
  { to: '/settings', label: 'Settings' },
]

export default function Header() {
  const { profile, logout } = useApp()
  const navigate = useNavigate()

  async function handleLogout() {
    await logout()
    navigate('/login', { replace: true })
  }

  return (
    <header className="site-header">
      <div className="site-header__inner">
        <NavLink to="/dashboard" className="brand" aria-label="AstraScan AI home">
          <span className="brand__mark">
            <LogoMark size={24} />
          </span>
          <span className="brand__name">AstraScan AI</span>
        </NavLink>

        <nav className="main-nav" aria-label="Primary">
          {NAV.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                isActive ? 'main-nav__link is-active' : 'main-nav__link'
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="site-header__actions">
          <button type="button" className="icon-btn" aria-label="Notifications">
            <Bell size={18} />
            <span className="icon-btn__dot" />
          </button>
          <button type="button" className="icon-btn" aria-label="Help">
            <HelpCircle size={18} />
          </button>
          <div className="user-chip">
            <span className="user-chip__avatar">
              <User size={14} />
            </span>
            <span className="user-chip__name">
              {profile.fullName || profile.email || 'Clinician'}
            </span>
          </div>
          <button type="button" className="btn btn--ghost btn--sm" onClick={handleLogout}>
            Logout
          </button>
        </div>
      </div>
    </header>
  )
}

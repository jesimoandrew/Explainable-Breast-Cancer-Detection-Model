import { useEffect, useState } from 'react'
import { NavLink, useParams } from 'react-router-dom'
import { useApp } from '../context/AppContext'
import { ShieldIcon, Sliders, User } from '../components/Icons'

const TABS = [
  { key: 'profile', label: 'Profile', Icon: User },
  { key: 'security', label: 'Security', Icon: ShieldIcon },
  { key: 'preferences', label: 'Preferences', Icon: Sliders },
]

const ROLES = [
  'Senior Radiologist',
  'Radiologist',
  'Radiology Resident',
  'Oncologist',
  'Clinical Researcher',
]

function ProfilePanel() {
  const { profile, setProfile } = useApp()
  const [draft, setDraft] = useState(profile)
  const [saved, setSaved] = useState(false)

  useEffect(() => setDraft(profile), [profile])

  function update(key, value) {
    setDraft((d) => ({ ...d, [key]: value }))
    setSaved(false)
  }

  async function handleSave() {
    // email lives on auth.users and medical_id is hospital-managed -- only the
    // two editable fields go back to the profiles table.
    await setProfile({ fullName: draft.fullName, role: draft.role })
    setSaved(true)
    window.setTimeout(() => setSaved(false), 2200)
  }

  return (
    <section className="card settings-panel">
      <div className="card__head">
        <div>
          <h2 className="card__title">Profile Information</h2>
          <p className="card__subtitle">
            Manage your clinical identity and contact details.
          </p>
        </div>
        <button type="button" className="btn btn--primary" onClick={handleSave}>
          {saved ? 'Saved' : 'Save Changes'}
        </button>
      </div>

      <div className="field-grid">
        <div className="field">
          <label className="field__label" htmlFor="fullName">
            Full Name
          </label>
          <input
            id="fullName"
            className="input"
            value={draft.fullName}
            onChange={(e) => update('fullName', e.target.value)}
          />
        </div>

        <div className="field">
          <label className="field__label" htmlFor="settingsEmail">
            Email Address
          </label>
          <input
            id="settingsEmail"
            type="email"
            className="input"
            value={draft.email}
            disabled
            readOnly
          />
          <p className="field__help">Managed by your sign-in account</p>
        </div>

        <div className="field">
          <label className="field__label" htmlFor="role">
            Clinical Role
          </label>
          <div className="select-wrap">
            <select
              id="role"
              className="input"
              value={draft.role}
              onChange={(e) => update('role', e.target.value)}
            >
              {ROLES.map((r) => (
                <option key={r}>{r}</option>
              ))}
            </select>
          </div>
        </div>

        <div className="field">
          <label className="field__label" htmlFor="medicalId">
            Medical ID / NPI
          </label>
          <input id="medicalId" className="input" value={draft.medicalId} disabled readOnly />
          <p className="field__help">Managed by hospital administration</p>
        </div>
      </div>
    </section>
  )
}

function SecurityPanel() {
  return (
    <section className="card settings-panel">
      <div className="card__head">
        <div>
          <h2 className="card__title">Security</h2>
          <p className="card__subtitle">
            Protect access to patient studies and diagnostic history.
          </p>
        </div>
        <button
          type="button"
          className="btn btn--primary"
          onClick={() => window.alert('Security settings updated.')}
        >
          Save Changes
        </button>
      </div>

      <div className="field-grid">
        <div className="field">
          <label className="field__label" htmlFor="currentPw">
            Current Password
          </label>
          <input id="currentPw" type="password" className="input" defaultValue="demo-password" />
        </div>
        <div className="field">
          <label className="field__label" htmlFor="newPw">
            New Password
          </label>
          <input id="newPw" type="password" className="input" placeholder="••••••••" />
        </div>
      </div>

      <ul className="toggle-list">
        <li>
          <div>
            <p className="toggle-list__title">Two-Factor Authentication</p>
            <p className="toggle-list__text">
              Require a one-time code from your hospital authenticator at sign-in.
            </p>
          </div>
          <label className="switch">
            <input type="checkbox" defaultChecked />
            <span />
          </label>
        </li>
        <li>
          <div>
            <p className="toggle-list__title">Audit Trail Alerts</p>
            <p className="toggle-list__text">
              Email me whenever a record I authored is exported or shared.
            </p>
          </div>
          <label className="switch">
            <input type="checkbox" defaultChecked />
            <span />
          </label>
        </li>
      </ul>
    </section>
  )
}

function PreferencesPanel() {
  return (
    <section className="card settings-panel">
      <div className="card__head">
        <div>
          <h2 className="card__title">Preferences</h2>
          <p className="card__subtitle">
            Tune how analyses and explainability overlays are presented.
          </p>
        </div>
        <button
          type="button"
          className="btn btn--primary"
          onClick={() => window.alert('Preferences saved.')}
        >
          Save Changes
        </button>
      </div>

      <div className="field-grid">
        <div className="field">
          <label className="field__label" htmlFor="overlay">
            Default Overlay
          </label>
          <div className="select-wrap">
            <select id="overlay" className="input" defaultValue="Grad-CAM">
              <option>Grad-CAM</option>
              <option>Grad-CAM++</option>
              <option>Score-CAM</option>
              <option>None</option>
            </select>
          </div>
        </div>
        <div className="field">
          <label className="field__label" htmlFor="threshold">
            Decision Threshold
          </label>
          <div className="select-wrap">
            <select id="threshold" className="input" defaultValue="High-Sensitivity (0.305)">
              <option>Main (0.500)</option>
              <option>Crossover (0.485)</option>
              <option>High-Sensitivity (0.305)</option>
            </select>
          </div>
          <p className="field__help">
            Lower thresholds favour sensitivity for cancer detection.
          </p>
        </div>
      </div>

      <ul className="toggle-list">
        <li>
          <div>
            <p className="toggle-list__title">Show Confidence Bars</p>
            <p className="toggle-list__text">
              Display the model confidence meter on every record card.
            </p>
          </div>
          <label className="switch">
            <input type="checkbox" defaultChecked />
            <span />
          </label>
        </li>
      </ul>
    </section>
  )
}

const PANELS = {
  profile: ProfilePanel,
  security: SecurityPanel,
  preferences: PreferencesPanel,
}

export default function Settings() {
  const { tab } = useParams()
  const Panel = PANELS[tab] || ProfilePanel

  return (
    <div className="page page--settings">
      <div className="settings-layout">
        <aside className="settings-nav">
          <h1 className="settings-nav__title">System Settings</h1>
          <nav aria-label="Settings sections">
            {TABS.map(({ key, label, Icon }) => (
              <NavLink
                key={key}
                to={`/settings/${key}`}
                className={({ isActive }) =>
                  isActive ? 'settings-nav__link is-active' : 'settings-nav__link'
                }
              >
                <Icon size={16} />
                {label}
              </NavLink>
            ))}
          </nav>
        </aside>

        <div className="settings-content">
          <Panel />
        </div>
      </div>
    </div>
  )
}

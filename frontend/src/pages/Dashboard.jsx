import { useNavigate } from 'react-router-dom'
import { useApp } from '../context/AppContext'
import RecordsTable from '../components/RecordsTable'
import { Plus, ScanIcon } from '../components/Icons'

export default function Dashboard() {
  const { profile } = useApp()
  const navigate = useNavigate()
  const name = profile.fullName || ''
  const shortName = name.startsWith('Dr.')
    ? `Dr. ${name.split(' ').pop()}`
    : name || 'Doctor'

  return (
    <div className="page page--dashboard">
      <section className="card hero">
        <div className="hero__head">
          <h1 className="hero__title">Welcome, {shortName}</h1>
          <p className="hero__subtitle">
            Explainable Deep Learning Support System for Breast Cancer Detection.
            Review patient scans with AI-assisted clarity.
          </p>
          <span className="hero__rule" />
        </div>

        <div className="hero__cta">
          <span className="hero__icon">
            <ScanIcon size={26} />
          </span>
          <h2 className="hero__cta-title">Ready for new Analysis?</h2>
          <p className="hero__cta-text">
            Upload mammography images for immediate AI
            <br />
            diagnostic support and explainability mapping.
          </p>
          <button
            type="button"
            className="btn btn--primary"
            onClick={() => navigate('/analysis')}
          >
            <Plus size={16} />
            Analyze Image
          </button>
        </div>
      </section>

      <RecordsTable title="Past Scan Records" limit={3} showViewAll />
    </div>
  )
}

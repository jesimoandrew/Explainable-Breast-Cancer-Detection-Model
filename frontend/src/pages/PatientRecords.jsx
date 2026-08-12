import { useNavigate } from 'react-router-dom'
import RecordsTable from '../components/RecordsTable'
import { Plus } from '../components/Icons'

export default function PatientRecords() {
  const navigate = useNavigate()

  return (
    <div className="page">
      <div className="page-head">
        <div>
          <h1 className="page-title">Patient Records</h1>
          <p className="page-subtitle">
            Every archived mammography study with its AI classification and
            explainability map.
          </p>
        </div>
        <button
          type="button"
          className="btn btn--primary"
          onClick={() => navigate('/analysis')}
        >
          <Plus size={16} />
          Analyze Image
        </button>
      </div>

      <RecordsTable title="All Records" />
    </div>
  )
}

import { useEffect, useState } from 'react'
import { Link, useNavigate, useParams } from 'react-router-dom'
import { useApp } from '../context/AppContext'
import ScanViewer, { CompareModal } from '../components/ScanViewer'
import {
  AlertTriangle,
  CheckCircle,
  ChevronLeft,
  Clock,
  FileText,
  Layers,
  NoteIcon,
  Share,
} from '../components/Icons'

export default function RecordDetails() {
  const { id } = useParams()
  const navigate = useNavigate()
  const { getRecord, updateRecord, deleteRecord, hydrateRecord } = useApp()
  const record = getRecord(id)

  const [notes, setNotes] = useState(record?.notes ?? '')
  const [saved, setSaved] = useState(false)
  const [comparing, setComparing] = useState(false)

  useEffect(() => {
    setNotes(record?.notes ?? '')
  }, [record?.id, record?.notes])

  // The list endpoint omits image payloads to stay fast -- fetch them on open.
  const needsImages = Boolean(record) && !record.analysis?.originalImage
  useEffect(() => {
    if (needsImages) hydrateRecord(id)
  }, [id, needsImages, hydrateRecord])

  if (!record) {
    return (
      <div className="page">
        <section className="card empty-state">
          <h1 className="page-title">Record not found</h1>
          <p className="page-subtitle">
            This case may have been deleted from the archive.
          </p>
          <Link to="/records" className="btn btn--primary">
            Back to Patient Records
          </Link>
        </section>
      </div>
    )
  }

  const malignant = record.result === 'Malignant'
  const analysis = record.analysis || null
  const primary = analysis?.models.find((m) => m.arch === analysis.primaryArch)
  const groundTruth = analysis?.groundTruth || null

  function handleSave() {
    updateRecord(record.id, { notes, notesEdited: 'just now' })
    setSaved(true)
    window.setTimeout(() => setSaved(false), 2200)
  }

  function handleDelete() {
    const ok = window.confirm(
      `Delete case ${record.id}? This permanently removes the study and its notes.`,
    )
    if (ok) {
      deleteRecord(record.id)
      navigate('/records')
    }
  }

  return (
    <div className="page page--details">
      <div className="details-head">
        <div className="details-head__left">
          <button
            type="button"
            className="icon-btn icon-btn--back"
            aria-label="Back to patient records"
            onClick={() => navigate(-1)}
          >
            <ChevronLeft size={18} />
          </button>
          <div>
            <h1 className="page-title">
              {record.patientName || 'Record Details'}
            </h1>
            <p className="page-subtitle">
              Case ID: {record.id} • {record.caseDate}
            </p>
          </div>
        </div>
        <div className="details-head__actions">
          <button type="button" className="btn btn--danger-outline" onClick={handleDelete}>
            Delete Record
          </button>
          <button type="button" className="btn btn--primary" onClick={handleSave}>
            {saved ? 'Saved' : 'Save Changes'}
          </button>
        </div>
      </div>

      <div className="details-grid">
        <div className="details-grid__main">
          <div className="scan-grid">
            <ScanViewer
              src={analysis?.originalImage}
              alt="Original mammography scan"
              label="Original Mammography"
              lesion={record.lesion}
              meta={`Raw scan · ${record.protocol}`}
              onExpand={
                analysis?.originalImage && primary?.heatmap
                  ? () => setComparing(true)
                  : undefined
              }
            />

            <ScanViewer
              src={primary?.heatmap}
              alt="Grad-CAM detection heatmap"
              label="AI Detection Heatmap"
              variant="heatmap"
              lesion={record.lesion}
              badge={
                <span className="chip chip--active">
                  <span className="chip__dot" />
                  Active Layer
                </span>
              }
              meta={
                <>
                  <Layers size={13} />
                  Grad-CAM · {primary ? primary.displayName : record.densityGrade}
                </>
              }
            />
          </div>

          {comparing && (
            <CompareModal
              original={analysis.originalImage}
              heatmap={primary.heatmap}
              title={`${record.name} · Original vs ${primary.displayName} Grad-CAM`}
              onClose={() => setComparing(false)}
            />
          )}

          {analysis && (
            <section className="card arch-card">
              <div className="card__head">
                <div>
                  <h2 className="card__title card__title--icon">
                    <Layers size={17} />
                    Architecture Comparison
                  </h2>
                  <p className="card__subtitle">
                    All four hi-res models run on this scan. DenseNet121 uses the
                    sensitivity-optimized threshold; the rest use 0.500.
                  </p>
                </div>
                <span className="card__meta">{analysis.device}</span>
              </div>

              <div className="arch-grid">
                {analysis.models.map((m) => (
                  <figure
                    key={m.arch}
                    className={m.isPrimary ? 'arch-tile is-primary' : 'arch-tile'}
                  >
                    <div className="arch-tile__frame">
                      <img src={m.heatmap} alt={`${m.displayName} Grad-CAM overlay`} />
                      {m.isPrimary && <span className="arch-tile__badge">Primary</span>}
                    </div>
                    <figcaption className="arch-tile__body">
                      <p className="arch-tile__name">{m.displayName}</p>
                      <span
                        className={
                          m.result === 'Malignant'
                            ? 'pill pill--danger'
                            : 'pill pill--success'
                        }
                      >
                        <span className="pill__dot" />
                        {m.result}
                      </span>
                      <dl className="arch-tile__stats">
                        <div>
                          <dt>p(malignant)</dt>
                          <dd>{m.probability.toFixed(3)}</dd>
                        </div>
                        <div>
                          <dt>threshold</dt>
                          <dd>{m.threshold.toFixed(3)}</dd>
                        </div>
                      </dl>
                      {m.thresholdVariant && (
                        <p className="arch-tile__variant">{m.thresholdVariant}</p>
                      )}
                    </figcaption>
                  </figure>
                ))}
              </div>
            </section>
          )}

          <section className="card notes-card">
            <div className="card__head">
              <h2 className="card__title card__title--icon">
                <NoteIcon size={17} />
                Clinical Notes
              </h2>
              <span className="card__meta">
                <Clock size={13} /> Last edited {record.notesEdited}
              </span>
            </div>
            <textarea
              className="notes-card__field"
              value={notes}
              rows={6}
              placeholder="Document your interpretation, correlation studies, and follow-up plan…"
              onChange={(e) => setNotes(e.target.value)}
            />
          </section>
        </div>

        <aside className="details-grid__side">
          <section
            className={
              malignant
                ? 'card verdict-card verdict-card--danger'
                : 'card verdict-card verdict-card--success'
            }
          >
            <div className="verdict-card__head">
              <span className="verdict-card__label">Classification</span>
              {malignant ? <AlertTriangle size={18} /> : <CheckCircle size={18} />}
            </div>
            <h2 className="verdict-card__value">{record.result}</h2>
            <p className="verdict-card__text">{record.summary}</p>
          </section>

          {groundTruth && (
            <section className="card truth-card">
              <div className="truth-card__head">
                <span className="truth-card__label">Dataset Ground Truth</span>
                <span
                  className={
                    groundTruth.label === 'Malignant'
                      ? 'pill pill--danger'
                      : 'pill pill--success'
                  }
                >
                  <span className="pill__dot" />
                  {groundTruth.label}
                </span>
              </div>
              <div className="truth-card__frame">
                <img src={groundTruth.roiImage} alt="Ground-truth ROI annotation" />
              </div>
              <p className="truth-card__note">
                CBIS-DDSM annotation, shown for comparison only — never fed to a model.
              </p>
            </section>
          )}

          <section className="card export-card">
            <button
              type="button"
              className="link-row"
              onClick={() => window.alert(`Preparing the PDF report for ${record.id}…`)}
            >
              <FileText size={16} />
              Export PDF Report
            </button>
            <button
              type="button"
              className="link-row"
              onClick={() => window.alert(`A secure link for ${record.id} has been copied.`)}
            >
              <Share size={16} />
              Share with Specialist
            </button>
          </section>
        </aside>
      </div>
    </div>
  )
}

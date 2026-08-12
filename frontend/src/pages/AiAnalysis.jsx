import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { useApp } from '../context/AppContext'
import { analyzeImage, checkHealth } from '../api'
import {
  AlertTriangle,
  CheckCircle,
  CloudUpload,
  Spinner,
  UploadArrow,
  User,
} from '../components/Icons'

const ACCEPT = '.png,.jpg,.jpeg,.bmp,.tif,.tiff'

const DISPLAY_DATE = { month: 'short', day: '2-digit', year: 'numeric' }
const LONG_DATE = { month: 'long', day: 'numeric', year: 'numeric' }

/** The API returns the record already assembled; fill in display-only fields. */
function toRecord(payload) {
  const now = new Date()
  return {
    ...payload,
    date: now.toLocaleDateString('en-US', DISPLAY_DATE),
    caseDate: now.toLocaleDateString('en-US', LONG_DATE),
    patientAge: payload.patientAge || '—',
    densityGrade: payload.densityGrade || '—',
    radiologist: payload.radiologist || 'Pending review',
    notesEdited: 'just now',
    lesion: { cx: 0.55, cy: 0.4, r: 0.17, intensity: payload.analysis.probability },
  }
}

export default function AiAnalysis() {
  const { addRecord } = useApp()
  const navigate = useNavigate()
  const inputRef = useRef(null)
  const [file, setFile] = useState(null)
  const [patientName, setPatientName] = useState('')
  const [dragging, setDragging] = useState(false)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState(null)
  const [engine, setEngine] = useState({ online: null })

  useEffect(() => {
    let alive = true
    checkHealth().then((h) => alive && setEngine(h))
    return () => {
      alive = false
    }
  }, [])

  function pick(selected) {
    if (!selected) return
    setFile(selected)
    setError(null)
  }

  function handleDrop(e) {
    e.preventDefault()
    setDragging(false)
    pick(e.dataTransfer.files?.[0])
  }

  async function handleAnalyze() {
    if (!file || busy || !patientName.trim()) return
    setBusy(true)
    setError(null)
    try {
      const record = toRecord(await analyzeImage(file, patientName.trim()))
      addRecord(record)
      navigate(`/records/${record.id}`)
    } catch (err) {
      setError(err.message || 'Analysis failed.')
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="page page--analysis">
      <div className="page-head page-head--stacked">
        <h1 className="page-title">Analyze Diagnostic Image</h1>
        <p className="page-subtitle">
          Upload a mammography scan for instant AI-powered malignancy
          classification and density mapping.
        </p>
      </div>

      <section className="card upload-card">
        <div className="field upload-card__field">
          <label className="field__label" htmlFor="patientName">
            Patient Name
          </label>
          <div className="input-wrap">
            <span className="input-wrap__icon">
              <User size={16} />
            </span>
            <input
              id="patientName"
              value={patientName}
              required
              placeholder="e.g. Maria Santos"
              onChange={(e) => setPatientName(e.target.value)}
            />
          </div>
          <p className="field__help">
            Stored with the scan so the record is identifiable in the archive.
          </p>
        </div>

        <div
          className={
            dragging ? 'dropzone is-dragging' : file ? 'dropzone has-file' : 'dropzone'
          }
          onDragOver={(e) => {
            e.preventDefault()
            setDragging(true)
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => inputRef.current?.click()}
          role="button"
          tabIndex={0}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault()
              inputRef.current?.click()
            }
          }}
        >
          <span className="dropzone__icon">
            <CloudUpload size={30} />
          </span>
          <p className="dropzone__title">
            {file ? file.name : 'Drag & drop mammogram'}
          </p>
          <p className="dropzone__hint">
            Supported formats: DICOM, PNG, JPEG. High-
            <br />
            resolution DICOM preferred for precision.
          </p>
          <div className="dropzone__badges">
            <span className="tag">Max 50MB</span>
            <span className="tag">HQ Quality</span>
          </div>
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPT}
            hidden
            onChange={(e) => pick(e.target.files?.[0])}
          />
        </div>

        <button
          type="button"
          className="btn btn--primary btn--block"
          onClick={handleAnalyze}
          disabled={!file || busy || !patientName.trim()}
        >
          {busy ? <Spinner size={17} /> : <UploadArrow size={17} />}
          {busy ? 'Running four architectures…' : 'Upload & Analyze'}
        </button>

        {error && (
          <p className="upload-card__error">
            <AlertTriangle size={14} />
            {error}
          </p>
        )}

        {!error && (!file || !patientName.trim()) && (
          <p className="upload-card__note">
            {!patientName.trim() && !file && 'Enter a patient name and select a scan.'}
            {!patientName.trim() && file && 'Enter a patient name to enable analysis.'}
            {patientName.trim() && !file && 'Select a scan to enable analysis.'}
          </p>
        )}

        {!error && (
          <p className="upload-card__note">
            {engine.online === null && 'Checking inference engine…'}
            {engine.online === true && (
              <>
                <CheckCircle size={13} /> Engine online on {engine.device} ·{' '}
                {engine.modelsLoaded} architectures ·{' '}
                {engine.storage?.connected
                  ? 'saving to Supabase'
                  : 'session-only (Supabase not connected)'}
              </>
            )}
            {engine.online === false && (
              <>
                <AlertTriangle size={13} /> Inference engine offline — start it with{' '}
                <code>uv run uvicorn app.server:app --port 8000</code>
              </>
            )}
          </p>
        )}
      </section>
    </div>
  )
}

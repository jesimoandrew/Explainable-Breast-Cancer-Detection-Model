import { useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'
import { useApp } from '../context/AppContext'
import Mammogram from './Mammogram'
import { Eye, Filter, Search, Share, Trash } from './Icons'

const FILTERS = ['All', 'Benign', 'Malignant']

export default function RecordsTable({
  title = 'Past Scan Records',
  limit,
  showViewAll = false,
}) {
  const { records, deleteRecord } = useApp()
  const navigate = useNavigate()
  const [query, setQuery] = useState('')
  const [searchOpen, setSearchOpen] = useState(false)
  const [filterOpen, setFilterOpen] = useState(false)
  const [filter, setFilter] = useState('All')
  const dropdownRef = useRef(null)

  useEffect(() => {
    if (!filterOpen) return undefined
    function onDocClick(e) {
      if (!dropdownRef.current?.contains(e.target)) setFilterOpen(false)
    }
    document.addEventListener('mousedown', onDocClick)
    return () => document.removeEventListener('mousedown', onDocClick)
  }, [filterOpen])

  const visible = useMemo(() => {
    const q = query.trim().toLowerCase()
    let rows = records
    if (filter !== 'All') rows = rows.filter((r) => r.result === filter)
    if (q) {
      rows = rows.filter((r) =>
        [r.patientName, r.name, r.id, r.date]
          .filter(Boolean)
          .some((v) => String(v).toLowerCase().includes(q)),
      )
    }
    return limit ? rows.slice(0, limit) : rows
  }, [records, filter, query, limit])

  function handleDelete(record) {
    const ok = window.confirm(`Delete ${record.name}? This cannot be undone.`)
    if (ok) deleteRecord(record.id)
  }

  return (
    <section className="card records-card">
      <div className="card__head">
        <h2 className="card__title">{title}</h2>
        <div className="records-card__tools">
          {searchOpen && (
            <input
              className="records-card__search"
              type="search"
              value={query}
              autoFocus
              placeholder="Search patients…"
              onChange={(e) => setQuery(e.target.value)}
              onBlur={() => !query && setSearchOpen(false)}
            />
          )}
          <div className="dropdown" ref={dropdownRef}>
            <button
              type="button"
              className={filter !== 'All' ? 'icon-btn is-on' : 'icon-btn'}
              aria-label="Filter results"
              aria-expanded={filterOpen}
              onClick={() => setFilterOpen((v) => !v)}
            >
              <Filter size={17} />
            </button>
            {filterOpen && (
              <ul className="dropdown__menu" role="menu">
                {FILTERS.map((f) => (
                  <li key={f}>
                    <button
                      type="button"
                      role="menuitem"
                      className={f === filter ? 'is-selected' : undefined}
                      onClick={() => {
                        setFilter(f)
                        setFilterOpen(false)
                      }}
                    >
                      {f}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <button
            type="button"
            className={searchOpen ? 'icon-btn is-on' : 'icon-btn'}
            aria-label="Search records"
            onClick={() => setSearchOpen((v) => !v)}
          >
            <Search size={17} />
          </button>
        </div>
      </div>

      <div className="table-wrap">
        <table className="records-table">
          <thead>
            <tr>
              <th>Patient Name</th>
              <th>Date</th>
              <th>Result</th>
              <th className="ta-right">Actions</th>
            </tr>
          </thead>
          <tbody>
            {visible.map((r) => (
              <tr key={r.id} onClick={() => navigate(`/records/${r.id}`)}>
                <td>
                  <div className="cell-image">
                    <span className="thumb">
                      {r.analysis?.originalImage ? (
                        <img src={r.analysis.originalImage} alt="" />
                      ) : (
                        <Mammogram lesion={r.lesion} />
                      )}
                    </span>
                    <span className="cell-image__name">
                      {r.patientName || 'Unnamed patient'}
                    </span>
                  </div>
                </td>
                <td className="muted">{r.date}</td>
                <td>
                  <span
                    className={
                      r.result === 'Malignant'
                        ? 'pill pill--danger'
                        : 'pill pill--success'
                    }
                  >
                    <span className="pill__dot" />
                    {r.result}
                  </span>
                </td>
                <td>
                  <div
                    className="row-actions"
                    onClick={(e) => e.stopPropagation()}
                  >
                    <Link
                      to={`/records/${r.id}`}
                      className="icon-btn icon-btn--sm"
                      aria-label={`View ${r.name}`}
                      title="View record"
                    >
                      <Eye size={16} />
                    </Link>
                    <button
                      type="button"
                      className="icon-btn icon-btn--sm"
                      aria-label={`Share ${r.name}`}
                      title="Share with specialist"
                      onClick={() =>
                        window.alert(`A share link for ${r.name} has been copied.`)
                      }
                    >
                      <Share size={16} />
                    </button>
                    <button
                      type="button"
                      className="icon-btn icon-btn--sm icon-btn--danger"
                      aria-label={`Delete ${r.name}`}
                      title="Delete record"
                      onClick={() => handleDelete(r)}
                    >
                      <Trash size={16} />
                    </button>
                  </div>
                </td>
              </tr>
            ))}
            {visible.length === 0 && (
              <tr className="is-empty">
                <td colSpan={4}>No records match the current filters.</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>

      {showViewAll && (
        <div className="card__foot">
          <Link to="/records" className="link-strong">
            View All Records
          </Link>
        </div>
      )}
    </section>
  )
}

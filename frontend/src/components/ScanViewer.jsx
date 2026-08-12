import { useCallback, useEffect, useRef, useState } from 'react'
import Mammogram from './Mammogram'
import { Maximize, Layers } from './Icons'

const LENS = 148          // lens diameter, px
const ZOOM_MIN = 1.5
const ZOOM_MAX = 8
const ZOOM_STEP = 0.35
const ZOOM_DEFAULT = 2.6

/**
 * A mammogram panel with a cursor-following magnifier, and an expand button that
 * opens a full-screen original-vs-heatmap comparison slider.
 */
export default function ScanViewer({
  src,
  alt,
  label,
  meta,
  badge,
  variant = 'original',
  lesion,
  onExpand,
}) {
  const frameRef = useRef(null)
  const [lens, setLens] = useState(null)   // {x, y, bgX, bgY} or null
  const [zoom, setZoom] = useState(ZOOM_DEFAULT)

  // Scroll over the image to zoom in and out inside the lens. Registered
  // natively because React's onWheel is passive and cannot preventDefault,
  // which would let the page scroll away under the cursor.
  useEffect(() => {
    const el = frameRef.current
    if (!el || !src) return undefined
    function onWheel(e) {
      e.preventDefault()
      const dir = e.deltaY > 0 ? -1 : 1
      setZoom((z) =>
        Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, +(z + dir * ZOOM_STEP).toFixed(2))),
      )
    }
    el.addEventListener('wheel', onWheel, { passive: false })
    return () => el.removeEventListener('wheel', onWheel)
  }, [src])

  const handleMove = useCallback(
    (e) => {
      if (!src) return
      const el = frameRef.current
      if (!el) return
      const r = el.getBoundingClientRect()
      const x = e.clientX - r.left
      const y = e.clientY - r.top
      if (x < 0 || y < 0 || x > r.width || y > r.height) {
        setLens(null)
        return
      }
      // Percentage position drives background-position, so the lens shows the
      // same point of the image the cursor is over, magnified.
      setLens({
        x,
        y,
        bgX: (x / r.width) * 100,
        bgY: (y / r.height) * 100,
      })
    },
    [src],
  )

  return (
    <figure className="card scan-panel">
      <figcaption className="scan-panel__head">
        <span className="scan-panel__label">{label}</span>
        {badge}
        {onExpand && (
          <button
            type="button"
            className="icon-btn icon-btn--sm"
            aria-label={`Expand ${label}`}
            title="Expand and compare"
            onClick={onExpand}
          >
            <Maximize size={15} />
          </button>
        )}
      </figcaption>

      <div
        ref={frameRef}
        className={src ? 'scan-panel__frame is-zoomable' : 'scan-panel__frame'}
        onMouseMove={handleMove}
        onMouseLeave={() => setLens(null)}
      >
        {src ? (
          <img className="scan-img" src={src} alt={alt} draggable={false} />
        ) : (
          <Mammogram variant={variant} lesion={lesion} />
        )}

        {lens && (
          <>
            <span
              className="scan-lens"
              aria-hidden="true"
              style={{
                left: lens.x,
                top: lens.y,
                width: LENS,
                height: LENS,
                backgroundImage: `url(${src})`,
                backgroundSize: `${zoom * 100}% ${zoom * 100}%`,
                backgroundPosition: `${lens.bgX}% ${lens.bgY}%`,
              }}
            />
            <span className="scan-zoom-badge" aria-hidden="true">
              {zoom.toFixed(1)}× · scroll to zoom
            </span>
          </>
        )}
      </div>

      {meta && <p className="scan-panel__meta">{meta}</p>}
    </figure>
  )
}

/**
 * Full-screen overlay comparing the original scan against the Grad-CAM overlay
 * with a draggable divider.
 */
export function CompareModal({ original, heatmap, title, onClose }) {
  const [pos, setPos] = useState(50)      // divider position, 0-100
  const areaRef = useRef(null)
  const dragging = useRef(false)

  const setFromClientX = useCallback((clientX) => {
    const el = areaRef.current
    if (!el) return
    const r = el.getBoundingClientRect()
    const pct = ((clientX - r.left) / r.width) * 100
    setPos(Math.min(100, Math.max(0, pct)))
  }, [])

  // Listeners go on the document so a fast drag that leaves the image still tracks.
  useEffect(() => {
    function move(e) {
      if (!dragging.current) return
      e.preventDefault()
      setFromClientX(e.touches ? e.touches[0].clientX : e.clientX)
    }
    function up() {
      dragging.current = false
    }
    document.addEventListener('mousemove', move)
    document.addEventListener('mouseup', up)
    document.addEventListener('touchmove', move, { passive: false })
    document.addEventListener('touchend', up)
    return () => {
      document.removeEventListener('mousemove', move)
      document.removeEventListener('mouseup', up)
      document.removeEventListener('touchmove', move)
      document.removeEventListener('touchend', up)
    }
  }, [setFromClientX])

  useEffect(() => {
    function onKey(e) {
      if (e.key === 'Escape') onClose()
      if (e.key === 'ArrowLeft') setPos((p) => Math.max(0, p - 2))
      if (e.key === 'ArrowRight') setPos((p) => Math.min(100, p + 2))
    }
    document.addEventListener('keydown', onKey)
    document.body.style.overflow = 'hidden'
    return () => {
      document.removeEventListener('keydown', onKey)
      document.body.style.overflow = ''
    }
  }, [onClose])

  return (
    <div className="compare-overlay" role="dialog" aria-modal="true" aria-label={title}>
      <div className="compare-overlay__bar">
        <span className="compare-overlay__title">
          <Layers size={15} />
          {title}
        </span>
        <div className="compare-overlay__legend">
          <span><i className="dot dot--plain" /> Original</span>
          <span><i className="dot dot--heat" /> Grad-CAM</span>
        </div>
        <button type="button" className="btn btn--ghost btn--sm" onClick={onClose}>
          Close
        </button>
      </div>

      <div className="compare-overlay__body" onMouseDown={onClose}>
        <div
          ref={areaRef}
          className="compare"
          onMouseDown={(e) => {
            e.stopPropagation()
            dragging.current = true
            setFromClientX(e.clientX)
          }}
          onTouchStart={(e) => {
            dragging.current = true
            setFromClientX(e.touches[0].clientX)
          }}
        >
          <img className="compare__img" src={heatmap} alt="Grad-CAM overlay" draggable={false} />
          <div className="compare__top" style={{ width: `${pos}%` }}>
            {/* Width is pinned to the frame so the left image does not squash
                as the divider moves -- it is revealed, not resized. */}
            <img
              className="compare__img compare__img--pinned"
              src={original}
              alt="Original mammography"
              draggable={false}
            />
          </div>

          <div className="compare__handle" style={{ left: `${pos}%` }}>
            <span className="compare__grip">
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <path d="m10 7-5 5 5 5M14 7l5 5-5 5" stroke="currentColor"
                      strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            </span>
          </div>

          <input
            className="compare__range"
            type="range"
            min="0"
            max="100"
            value={pos}
            aria-label="Comparison position"
            onChange={(e) => setPos(Number(e.target.value))}
            onMouseDown={(e) => e.stopPropagation()}
          />
        </div>
      </div>

      <p className="compare-overlay__hint">
        Drag the divider · ← → to nudge · Esc to close
      </p>
    </div>
  )
}

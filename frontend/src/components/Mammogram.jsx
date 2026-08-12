// Procedural stand-in for a real mammogram render. The production build swaps
// this for the PNG/DICOM frame returned by the inference service; the heatmap
// variant mirrors the Grad-CAM overlay produced in app/app.py.

import { useId } from 'react'

const LESION = { cx: 0.55, cy: 0.38, r: 0.2, intensity: 1 }

export default function Mammogram({
  variant = 'original',
  lesion = LESION,
  flip = false,
  className = '',
}) {
  // Every instance needs its own gradient/clip ids -- two records that happen to
  // share lesion coordinates would otherwise collide on the same <defs> ids.
  const uid = useId().replace(/:/g, '')
  const W = 200
  const H = 300
  const cx = lesion.cx * W
  const cy = lesion.cy * H
  const r = lesion.r * H

  return (
    <svg
      className={`mammo ${className}`}
      viewBox={`0 0 ${W} ${H}`}
      preserveAspectRatio="xMidYMid slice"
      role="img"
      aria-label={
        variant === 'heatmap'
          ? 'AI detection heatmap overlay'
          : 'Mammography scan'
      }
    >
      <defs>
        <radialGradient id={`tissue-${uid}`} cx="30%" cy="45%" r="85%">
          <stop offset="0%" stopColor="#e9e6e2" />
          <stop offset="45%" stopColor="#b9b3ad" />
          <stop offset="78%" stopColor="#6d6862" />
          <stop offset="100%" stopColor="#2a2724" />
        </radialGradient>
        <radialGradient id={`heat-${uid}`} cx="50%" cy="50%" r="50%">
          <stop offset="0%" stopColor="#ff2d1a" stopOpacity="0.92" />
          <stop offset="35%" stopColor="#ff6a00" stopOpacity="0.78" />
          <stop offset="62%" stopColor="#ffd000" stopOpacity="0.5" />
          <stop offset="82%" stopColor="#39d15c" stopOpacity="0.26" />
          <stop offset="100%" stopColor="#1f6bff" stopOpacity="0" />
        </radialGradient>
        <filter id={`soft-${uid}`}>
          <feGaussianBlur stdDeviation="3.2" />
        </filter>
        <clipPath id={`clip-${uid}`}>
          <path d="M2 0 C 96 14, 158 74, 150 156 C 144 232, 88 288, 2 300 Z" />
        </clipPath>
      </defs>

      <rect width={W} height={H} fill="#0b0c0d" />

      <g transform={flip ? `translate(${W},0) scale(-1,1)` : undefined}>
        <path
          d="M2 0 C 96 14, 158 74, 150 156 C 144 232, 88 288, 2 300 Z"
          fill={`url(#tissue-${uid})`}
        />
        <g clipPath={`url(#clip-${uid})`} filter={`url(#soft-${uid})`}>
          <ellipse cx="52" cy="96" rx="38" ry="30" fill="#fff" opacity="0.14" />
          <ellipse cx="86" cy="170" rx="44" ry="34" fill="#fff" opacity="0.1" />
          <ellipse cx="40" cy="220" rx="34" ry="42" fill="#fff" opacity="0.08" />
          <ellipse cx="100" cy="72" rx="24" ry="18" fill="#fff" opacity="0.12" />
          <ellipse cx={cx} cy={cy} rx={r * 0.5} ry={r * 0.42} fill="#fff" opacity="0.2" />
        </g>

        {variant === 'heatmap' && (
          <g clipPath={`url(#clip-${uid})`}>
            <circle
              cx={cx}
              cy={cy}
              r={r * 1.5}
              fill={`url(#heat-${uid})`}
              opacity={0.55 + 0.45 * lesion.intensity}
            />
            <circle
              cx={cx}
              cy={cy}
              r={r * 0.62}
              fill="none"
              stroke="#fff"
              strokeWidth="0.9"
              strokeDasharray="3 3"
              opacity="0.75"
            />
          </g>
        )}
      </g>
    </svg>
  )
}

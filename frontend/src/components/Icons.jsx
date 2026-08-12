// Single-file icon set so the app stays dependency-free. Every icon inherits
// `currentColor`, so colour is controlled entirely from CSS.

function Svg({ size = 18, children, fill = 'none', ...rest }) {
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 24 24"
      fill={fill}
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
      focusable="false"
      {...rest}
    >
      {children}
    </svg>
  )
}

export const LogoMark = ({ size = 26 }) => (
  <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden="true">
    <rect width="32" height="32" rx="8" fill="currentColor" />
    <path
      d="M16 7.2l6.6 3v4.8c0 4.7-2.8 8.4-6.6 9.2-3.8-.8-6.6-4.5-6.6-9.2v-4.8l6.6-3z"
      stroke="#fff"
      strokeWidth="1.6"
      strokeLinejoin="round"
    />
    <path
      d="M12.7 16.1l2.2 2.2 4.4-4.7"
      stroke="#fff"
      strokeWidth="1.7"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
)

export const Bell = (p) => (
  <Svg {...p}>
    <path d="M18 8a6 6 0 1 0-12 0c0 6-2 7-2 7h16s-2-1-2-7" />
    <path d="M13.7 20a2 2 0 0 1-3.4 0" />
  </Svg>
)

export const HelpCircle = (p) => (
  <Svg {...p}>
    <circle cx="12" cy="12" r="9" />
    <path d="M9.6 9.4a2.5 2.5 0 0 1 4.9.6c0 1.7-2.5 2.5-2.5 2.5" />
    <path d="M12 17h.01" />
  </Svg>
)

export const Search = (p) => (
  <Svg {...p}>
    <circle cx="11" cy="11" r="7" />
    <path d="m20 20-3.6-3.6" />
  </Svg>
)

export const Filter = (p) => (
  <Svg {...p}>
    <path d="M4 6h16M7 12h10M10 18h4" />
  </Svg>
)

export const Eye = (p) => (
  <Svg {...p}>
    <path d="M2 12s3.6-6.5 10-6.5S22 12 22 12s-3.6 6.5-10 6.5S2 12 2 12z" />
    <circle cx="12" cy="12" r="2.7" />
  </Svg>
)

export const Share = (p) => (
  <Svg {...p}>
    <circle cx="18" cy="5" r="2.6" />
    <circle cx="6" cy="12" r="2.6" />
    <circle cx="18" cy="19" r="2.6" />
    <path d="m8.3 10.8 7.4-4.3M8.3 13.2l7.4 4.3" />
  </Svg>
)

export const Trash = (p) => (
  <Svg {...p}>
    <path d="M3.5 6h17M9 6V4.4A1.4 1.4 0 0 1 10.4 3h3.2A1.4 1.4 0 0 1 15 4.4V6" />
    <path d="M18.5 6 18 19.2a1.8 1.8 0 0 1-1.8 1.8H7.8A1.8 1.8 0 0 1 6 19.2L5.5 6" />
    <path d="M10 10.5v6M14 10.5v6" />
  </Svg>
)

export const Plus = (p) => (
  <Svg {...p}>
    <path d="M12 5v14M5 12h14" />
  </Svg>
)

export const CloudUpload = (p) => (
  <Svg {...p}>
    <path d="M7 18a4.5 4.5 0 0 1-.6-8.96 6 6 0 0 1 11.5 1.66A3.9 3.9 0 0 1 17.5 18" />
    <path d="M12 12.5V21M8.8 15.6 12 12.4l3.2 3.2" />
  </Svg>
)

export const UploadArrow = (p) => (
  <Svg {...p}>
    <path d="M12 19V6M6.5 11.5 12 6l5.5 5.5" />
  </Svg>
)

export const ScanIcon = (p) => (
  <Svg {...p}>
    <path d="M4 8.5V6a2 2 0 0 1 2-2h2.5M15.5 4H18a2 2 0 0 1 2 2v2.5M20 15.5V18a2 2 0 0 1-2 2h-2.5M8.5 20H6a2 2 0 0 1-2-2v-2.5" />
    <circle cx="12" cy="12" r="3" />
  </Svg>
)

export const ChevronLeft = (p) => (
  <Svg {...p}>
    <path d="m14.5 5-7 7 7 7" />
  </Svg>
)

export const ChevronDown = (p) => (
  <Svg {...p}>
    <path d="m6 9.5 6 6 6-6" />
  </Svg>
)

export const ArrowRight = (p) => (
  <Svg {...p}>
    <path d="M5 12h14M13 6l6 6-6 6" />
  </Svg>
)

export const Mail = (p) => (
  <Svg {...p}>
    <rect x="3" y="5" width="18" height="14" rx="2.2" />
    <path d="m3.6 6.5 8.4 6 8.4-6" />
  </Svg>
)

export const Lock = (p) => (
  <Svg {...p}>
    <rect x="4.5" y="10.5" width="15" height="10" rx="2.2" />
    <path d="M8 10.5V7.8a4 4 0 0 1 8 0v2.7" />
  </Svg>
)

export const Hospital = (p) => (
  <Svg {...p}>
    <path d="M4 21V6.5A1.5 1.5 0 0 1 5.5 5H11V3h2v2h5.5A1.5 1.5 0 0 1 20 6.5V21" />
    <path d="M2.5 21h19M12 9.5v5M9.5 12h5" />
  </Svg>
)

export const AlertTriangle = (p) => (
  <Svg {...p}>
    <path d="M10.3 3.9 2.6 17.2A2 2 0 0 0 4.3 20.2h15.4a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z" />
    <path d="M12 9v4.5M12 17h.01" />
  </Svg>
)

export const CheckCircle = (p) => (
  <Svg {...p}>
    <circle cx="12" cy="12" r="9" />
    <path d="m8.3 12.2 2.5 2.5 4.9-5.1" />
  </Svg>
)

export const FileText = (p) => (
  <Svg {...p}>
    <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5z" />
    <path d="M14 3v5h5M9 13h6M9 17h4" />
  </Svg>
)

export const NoteIcon = (p) => (
  <Svg {...p}>
    <path d="M4 5.5A1.5 1.5 0 0 1 5.5 4h13A1.5 1.5 0 0 1 20 5.5v13a1.5 1.5 0 0 1-1.5 1.5h-13A1.5 1.5 0 0 1 4 18.5z" />
    <path d="M8 9h8M8 12.5h8M8 16h5" />
  </Svg>
)

export const User = (p) => (
  <Svg {...p}>
    <circle cx="12" cy="8.2" r="3.7" />
    <path d="M4.8 20a7.3 7.3 0 0 1 14.4 0" />
  </Svg>
)

export const ShieldIcon = (p) => (
  <Svg {...p}>
    <path d="M12 3.2 19 6v5.4c0 4.6-2.9 8.4-7 9.4-4.1-1-7-4.8-7-9.4V6l7-2.8z" />
  </Svg>
)

export const Sliders = (p) => (
  <Svg {...p}>
    <path d="M5 20v-6M5 10V4M12 20v-9M12 7V4M19 20v-4M19 12V4" />
    <path d="M2.5 14h5M9.5 7h5M16.5 16h5" />
  </Svg>
)

export const Maximize = (p) => (
  <Svg {...p}>
    <path d="M9 3H4.5A1.5 1.5 0 0 0 3 4.5V9M15 3h4.5A1.5 1.5 0 0 1 21 4.5V9M21 15v4.5a1.5 1.5 0 0 1-1.5 1.5H15M3 15v4.5A1.5 1.5 0 0 0 4.5 21H9" />
  </Svg>
)

export const Layers = (p) => (
  <Svg {...p}>
    <path d="m12 3 8.5 4.6L12 12.2 3.5 7.6 12 3z" />
    <path d="m3.5 12.4 8.5 4.6 8.5-4.6M3.5 16.9l8.5 4.6 8.5-4.6" />
  </Svg>
)

export const Clock = (p) => (
  <Svg {...p}>
    <circle cx="12" cy="12" r="9" />
    <path d="M12 7.2V12l3.2 1.9" />
  </Svg>
)

export const Spinner = ({ size = 18 }) => (
  <svg
    className="spinner"
    width={size}
    height={size}
    viewBox="0 0 24 24"
    fill="none"
    aria-hidden="true"
  >
    <circle cx="12" cy="12" r="9" stroke="currentColor" strokeWidth="2.4" opacity="0.25" />
    <path
      d="M21 12a9 9 0 0 0-9-9"
      stroke="currentColor"
      strokeWidth="2.4"
      strokeLinecap="round"
    />
  </svg>
)

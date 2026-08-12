const LINKS = ['Privacy Policy', 'Terms of Service', 'Clinical Documentation', 'Support']

export default function Footer({ brand = 'AstraScan AI' }) {
  return (
    <footer className="site-footer">
      <div className="site-footer__inner">
        <div className="site-footer__left">
          <span className="site-footer__brand">{brand}</span>
          <span className="site-footer__copy">
            © 2024 AstraScan Medical Intelligence. All rights reserved.
          </span>
        </div>
        <nav className="site-footer__links" aria-label="Footer">
          {LINKS.map((label) => (
            <a key={label} href="#" onClick={(e) => e.preventDefault()}>
              {label}
            </a>
          ))}
        </nav>
      </div>
    </footer>
  )
}

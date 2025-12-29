export default function Resume() {
  return (
    <section className="page">
      <div className="panel section-header">
        <h1>Resume</h1>
        <p>Download or view the latest resume below.</p>
        <div className="pill-row">
          <a className="nav-pill" href="/resumes/resume.pdf" target="_blank" rel="noreferrer">
            Open PDF
          </a>
          <a className="nav-pill nav-pill--accent" href="/resumes/resume.pdf" download>
            Download
          </a>
        </div>
      </div>
      <div className="panel pdf-panel">
        <iframe title="Resume" src="/resumes/resume.pdf" className="pdf-viewer" />
      </div>
    </section>
  );
}

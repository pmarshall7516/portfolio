export default function Resume() {
  const baseUrl = import.meta.env.BASE_URL;

  return (
    <section className="page">
      <div className="panel section-header">
        <h1>Resume</h1>
        <p>Download or view my latest resume below!</p>
        <div className="pill-row">
          <a
            className="nav-pill nav-pill--accent"
            href={`${baseUrl}resumes/resume.pdf`}
            download
          >
            Download
          </a>
        </div>
      </div>
      <div className="panel pdf-panel">
        <iframe
          title="Resume"
          src={`${baseUrl}resumes/resume.pdf`}
          className="pdf-viewer"
        />
      </div>
    </section>
  );
}

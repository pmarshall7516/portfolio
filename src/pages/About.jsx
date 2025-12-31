export default function About() {
  const profileImage = `${import.meta.env.BASE_URL}portfolio-pic.png`;
  const fallbackProfileImage = "/portfolio-pic.png";

  const handleProfileError = (event) => {
    const img = event.currentTarget;
    if (img.dataset.fallback === "true") {
      return;
    }
    img.dataset.fallback = "true";
    img.src = fallbackProfileImage;
  };

  return (
    <section className="page about-page">
      <div className="hero-grid">
        <div className="panel hero-panel">
          <p className="eyebrow">Portfolio</p>
          <h1>Patrick Marshall</h1>
          <div className="degree-list">
            <p>B.S. Computer Science @ Milwaukee School of Engineering - 2025</p>
            <p>M.S. Machine Learning @ Milwaukee School of Engineering - 2026</p>
          </div>
          <div className="pill-row">
            <span className="badge-pill">AI</span>
            <span className="badge-pill">Machine Learning</span>
            <span className="badge-pill">Reinforcement Learning</span>
            <span className="badge-pill">GPU Programming</span>
          </div>
        </div>
        <div className="panel hero-panel image-panel">
          <img
            src={profileImage}
            alt="Profile placeholder"
            className="profile-image"
            onError={handleProfileError}
          />
        </div>
      </div>

      <div className="panel latex-panel">
        <h2>About Me</h2>
        <div className="latex-text">
          I'm <strong>Patrick Marshall</strong>, a M.S. Machine Learning student at Milwaukee School of Engineering, with a focus on building practical systems that bridge modern AI with high-performance computing. 
          As a Software Engineer Contractor at GE Healthcare, I work on GPU benchmarking and workload analysis by automating suites in Docker, comparing CUDA and SYCL performance, and using Nsight tools to understand kernel behavior across real hardware configurations. 
          In my undergraduate research, I co-developed the Behavioral Actor-Critic (BAC) framework: an ensemble actor-critic approach that uses reward decomposition, scalarization-based policy aggregation, and surrogate optimization to let users tune an RL agent's behavior after training.
          Outside of coursework, I'm involved in MSOE's AI Club and bring a disciplined, team-first mindset from competing on the MSOE NCAA Men's Soccer team.  
        </div>
      </div>

      <div className="panel contact-panel">
        <h2>Contact</h2>
        <div className="contact-grid">
          <div className="contact-card">
            <p className="contact-label">Email</p>
            <p className="contact-value">patrick.wayne.marshall@gmail.com</p>
          </div>
          <div className="contact-card">
            <p className="contact-label">Phone</p>
            <p className="contact-value">(815) 715-6134</p>
          </div>
        </div>
      </div>
    </section>
  );
}

import { Link } from "react-router-dom";

export default function ProjectBehavioralControl() {
  const baseUrl = import.meta.env.BASE_URL;
  const apiBase = import.meta.env.VITE_BEHAVIORAL_CONTROL_URL;
  const iframeSrc = apiBase
    ? `${baseUrl}behavioral-control/index.html?apiBase=${encodeURIComponent(apiBase)}`
    : `${baseUrl}behavioral-control/index.html`;

  return (
    <section className="page">
      <Link className="text-link back-link" to="/projects/ai-ml">
        ← Back
      </Link>
      <div className="panel section-header">
        <h1>Behavioral Control</h1>
        <p>
          Behavioral Actor-Critic (BAC) framework for controllable reinforcement learning.
        </p>
      </div>
      <div className="panel frame-panel">
        <iframe
          title="Behavioral Control Demo"
          className="project-frame"
          src={iframeSrc}
        />
      </div>
    </section>
  );
}

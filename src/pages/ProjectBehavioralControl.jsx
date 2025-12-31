import { Link } from "react-router-dom";

export default function ProjectBehavioralControl() {
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
          src="/behavioral-control/index.html"
        />
      </div>
    </section>
  );
}

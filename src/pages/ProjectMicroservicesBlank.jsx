import { Link } from "react-router-dom";

export default function ProjectMicroservicesBlank() {
  return (
    <section className="page">
      <Link className="text-link back-link" to="/projects/microservices">
        ← Back
      </Link>
      <div className="panel section-header">
        <h1>Microservices Project</h1>
        <p>This is a placeholder project page for upcoming work.</p>
      </div>
      <div className="panel">
        <h2>Coming Soon</h2>
        <p className="muted-text">
          Details, architecture diagrams, and demos will be added here.
        </p>
      </div>
    </section>
  );
}

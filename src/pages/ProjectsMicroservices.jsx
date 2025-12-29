import { Link } from "react-router-dom";
import { projects } from "../data/projects.js";

const microserviceProjects = projects.filter(
  (project) => project.category === "Microservices"
);

export default function ProjectsMicroservices() {
  return (
    <section className="page">
      <Link className="text-link back-link" to="/projects">
        ← Back
      </Link>
      <div className="panel section-header">
        <h1>Microservices</h1>
        <p>Distributed systems and service-oriented experiments.</p>
      </div>
      <div className="grid cards-grid">
        {microserviceProjects.map((project) => (
          <article key={project.title} className="panel project-detail-card">
            <div className="project-icon">{project.icon}</div>
            <div className="project-copy">
              <h2>{project.title}</h2>
              <p>{project.description}</p>
              <div className="project-badges">
                {project.badges.map((badge) => (
                  <span key={badge} className="badge-pill">
                    {badge}
                  </span>
                ))}
              </div>
              <Link className="nav-pill nav-pill--accent project-button" to={project.to}>
                View Project
              </Link>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

import { Link } from "react-router-dom";
import { projects } from "../data/projects.js";

const aiMlProjects = projects.filter(
  (project) => project.category === "AI/Machine Learning"
);
const sortedAiMlProjects = [...aiMlProjects].sort((a, b) =>
  a.title.localeCompare(b.title)
);

export default function ProjectsAIML() {
  return (
    <section className="page">
      <Link className="text-link back-link" to="/projects">
        ← Back
      </Link>
      <div className="panel section-header">
        <h1>AI / Machine Learning</h1>
        <p>Projects focused on intelligent systems and reinforcement learning.</p>
      </div>
      <div className="grid project-detail-grid">
        {sortedAiMlProjects.map((project) => (
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

import { Link } from "react-router-dom";
import { projectCategories } from "../data/projects.js";

export default function Projects() {
  return (
    <section className="page">
      <div className="panel section-header">
        <h1>Projects</h1>
        <p>Project Categories</p>
      </div>
      <div className="grid cards-grid">
        {projectCategories.map((category) => (
          <article key={category.title} className="panel project-card">
            <h2>{category.title}</h2>
            <p>{category.description}</p>
            <Link className="text-link" to={category.to}>
              Explore →
            </Link>
          </article>
        ))}
      </div>
    </section>
  );
}

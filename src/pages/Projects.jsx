import { Link } from "react-router-dom";
import { projectCategories } from "../data/projects.js";

export default function Projects() {
  const sortedCategories = [...projectCategories].sort((a, b) => {
    if (a.title === "All Projects") return -1;
    if (b.title === "All Projects") return 1;
    return a.title.localeCompare(b.title);
  });

  return (
    <section className="page">
      <div className="panel section-header">
        <h1>Project Categories</h1>
      </div>
      <div className="grid project-detail-grid">
        {sortedCategories.map((category) => (
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

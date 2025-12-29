import { Link } from "react-router-dom";

export default function ProjectSearchAlgoVisualizer() {
  return (
    <section className="page">
      <Link className="text-link back-link" to="/projects/simulations">
        ← Back
      </Link>
      <div className="panel section-header">
        <h1>Search Algorithm Visualizer</h1>
        <p>Visualize Dijkstra and A* Algorithms using a customizable search grid.</p>
      </div>
      <div className="panel frame-panel">
        <iframe
          title="Search Algorithm Visualizer"
          className="project-frame"
          src="/search-algo-visualizer/index.html"
        />
      </div>
    </section>
  );
}

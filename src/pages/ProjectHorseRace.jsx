import { Link } from "react-router-dom";
import useIframeAutoHeight from "../hooks/useIframeAutoHeight.js";

export default function ProjectHorseRace() {
  const baseUrl = import.meta.env.BASE_URL;
  const { frameRef, onLoad } = useIframeAutoHeight();

  return (
    <section className="page">
      <Link className="text-link back-link" to="/projects/simulations">
        ← Back
      </Link>
      <div className="panel section-header">
        <h1>Horse Race</h1>
        <p>Probability and statistics simulation using the horse racing family board game.</p>
      </div>
      <div className="panel frame-panel">
        <iframe
          title="Horse Race Simulation"
          className="project-frame"
          src={`${baseUrl}horse-race/index.html`}
          ref={frameRef}
          onLoad={onLoad}
        />
      </div>
    </section>
  );
}

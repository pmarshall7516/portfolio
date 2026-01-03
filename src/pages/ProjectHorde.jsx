import { Link } from "react-router-dom";
import useIframeAutoHeight from "../hooks/useIframeAutoHeight.js";

export default function ProjectHorde() {
  const baseUrl = import.meta.env.BASE_URL;
  const apiBase = import.meta.env.VITE_HORDE_API_URL;
  const { frameRef, onLoad } = useIframeAutoHeight();
  const iframeSrc = apiBase
    ? `${baseUrl}horde/index.html?apiBase=${encodeURIComponent(apiBase)}`
    : `${baseUrl}horde/index.html`;

  return (
    <section className="page">
      <Link className="text-link back-link" to="/projects/ai-ml">
        ← Back
      </Link>
      <div className="panel section-header features-panel">
        <h1>Horde TCG</h1>
        <p>
          A custom trading card game built to study reinforcement learning behavior
          in competitive, imperfect information settings.
        </p>
      </div>
      <div className="panel section-header">
        <h2>Features</h2>
        <p>Simulate full match sets with AI agents or take direct control.</p>
        <h3>Controls &amp; Modes</h3>
        <ol className="features-list features-list--numbered">
          <li>Pick agents for both players or switch to User for manual play.</li>
          <li>Set match length and target wins to stress test strategies.</li>
          <li>Toggle between automatic and manual step modes.</li>
        </ol>
        <h3>What to Watch</h3>
        <ul className="features-list features-list--bulleted">
          <li>Observe card pick rates and policy recommendations over time.</li>
          <li>Track combat outcomes with per-round battle logs.</li>
          <li>Compare agent performance using win rates and HP curves.</li>
        </ul>
        <p className="feature-note">
          * Note: If the Simulation does not display any card images or seem to work, click the "Start" button and wait about a minute. Refresh if it still has not changed.
        </p>
      </div>
      <div className="panel frame-panel">
        <iframe
          title="Horde TCG Demo"
          className="project-frame"
          src={iframeSrc}
          ref={frameRef}
          onLoad={onLoad}
        />
      </div>
    </section>
  );
}

import { Link } from "react-router-dom";
import useIframeAutoHeight from "../hooks/useIframeAutoHeight.js";

export default function ProjectBehavioralControl() {
  const baseUrl = import.meta.env.BASE_URL;
  const apiBase = import.meta.env.VITE_BEHAVIORAL_CONTROL_URL;
  const { frameRef, onLoad } = useIframeAutoHeight();
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
          ref={frameRef}
          onLoad={onLoad}
        />
      </div>
      <div className="panel research-card">
        <div className="research-copy">
          <h2>Related Research</h2>
          <h3>
            Surrogate-Based Scalarization for Behavioral Adjustment in Actor-Critic
            Algorithms
          </h3>
          <p>
            In this paper, we extend the concept of reward decomposition and masking for
            AI control to policy-based agents, specifically within the framework of
            reinforcement learning. By utilizing the Masked A2C ensemble method, we
            explore how multiple, independent policy networks can be trained on different
            components of reward decomposition, aggregating them through an ensemble
            approach. This paper discusses both the theoretical background and practical
            implementation, focusing on how policy aggregation with masked policies can
            enhance performance in environments with complex strategic interactions, like
            board games such as Stratego. The use of reward decomposition enables the
            agent to separately handle different components of the reward function,
            adjusting its behavior for a more controlled learning process. Additionally,
            we discuss the challenges of training an ensemble of policies and how
            off-policy corrections are applied to ensure stable learning.
          </p>
          <a
            className="nav-pill research-button"
            href={`${baseUrl}research/bac-paper.pdf`}
            target="_blank"
            rel="noreferrer"
          >
            Open PDF
          </a>
        </div>
      </div>
    </section>
  );
}

const researchItems = [
  {
    title: "Surrogate-Based Scalarization for Behavioral Adjustment in Actor-Critic Algorithms",
    abstract:
      "In this paper, we extend the concept of reward decomposition and masking for AI control to policy-based agents, specifically within the framework of reinforcement learning. By utilizing the Masked A2C ensemble method, we explore how multiple, independent policy networks can be trained on different components of reward decomposition, aggregating them through an ensemble approach. This paper discusses both the theoretical background and practical implementation, focusing on how policy aggregation with masked policies can enhance performance in environments with complex strategic interactions, like board games such as Stratego. The use of reward decomposition enables the agent to separately handle different components of the reward function, adjusting its behavior for a more controlled learning process. Additionally, we discuss the challenges of training an ensemble of policies and how off-policy corrections are applied to ensure stable learning.",
    pdf: "/bac-paper.pdf",
  },
  // {
  //   title: "Dummy Title: Replace This",
  //   abstract:
  //     "Dummy abstract text.",
  //   pdf: "/paper-placeholder.pdf",
  // },
];

export default function Research() {
  return (
    <section className="page">
      <div className="panel section-header">
        <h1>Research</h1>
        <p>Research Publications and ongoing work.</p>
      </div>
      <div className="stack">
        {researchItems.map((item) => (
          <article key={item.title} className="panel research-card">
            <div className="research-copy">
              <h2>{item.title}</h2>
              <p>{item.abstract}</p>
              <a
                className="nav-pill research-button"
                href={`research/${item.pdf}`}
                target="_blank"
                rel="noreferrer"
              >
                Open PDF
              </a>
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}

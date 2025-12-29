export default function ProjectHorseRace() {
  return (
    <section className="page">
      <div className="panel section-header">
        <h1>Horse Race</h1>
        <p>Simulation build embedded from the compiled project.</p>
      </div>
      <div className="panel frame-panel">
        <iframe
          title="Horse Race Simulation"
          className="project-frame"
          src="/horse-race/index.html"
        />
      </div>
    </section>
  );
}

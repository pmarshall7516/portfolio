import { Routes, Route, Navigate } from "react-router-dom";
import NavBar from "./components/NavBar.jsx";
import ScrollToTop from "./components/ScrollToTop.jsx";
import About from "./pages/About.jsx";
import Resume from "./pages/Resume.jsx";
import Research from "./pages/Research.jsx";
import Projects from "./pages/Projects.jsx";
import ProjectsSimulations from "./pages/ProjectsSimulations.jsx";
import ProjectsMicroservices from "./pages/ProjectsMicroservices.jsx";
import ProjectsAIML from "./pages/ProjectsAIML.jsx";
import ProjectsAll from "./pages/ProjectsAll.jsx";
import ProjectHorseRace from "./pages/ProjectHorseRace.jsx";
import ProjectSearchAlgoVisualizer from "./pages/ProjectSearchAlgoVisualizer.jsx";
import ProjectBehavioralControl from "./pages/ProjectBehavioralControl.jsx";
import ProjectMicroservicesBlank from "./pages/ProjectMicroservicesBlank.jsx";

export default function App() {
  return (
    <div className="app-shell">
      <ScrollToTop />
      <NavBar />
      <main className="main-content">
        <Routes>
          <Route path="/" element={<Navigate to="/about" replace />} />
          <Route path="/about" element={<About />} />
          <Route path="/resume" element={<Resume />} />
          <Route path="/research" element={<Research />} />
          <Route path="/projects" element={<Projects />} />
          <Route path="/projects/all" element={<ProjectsAll />} />
          <Route path="/projects/simulations" element={<ProjectsSimulations />} />
          <Route path="/projects/simulations/horse-race" element={<ProjectHorseRace />} />
          <Route
            path="/projects/simulations/search-algo-visualizer"
            element={<ProjectSearchAlgoVisualizer />}
          />
          <Route path="/projects/ai-ml" element={<ProjectsAIML />} />
          <Route
            path="/projects/ai-ml/behavioral-control"
            element={<ProjectBehavioralControl />}
          />
          <Route path="/projects/microservices" element={<ProjectsMicroservices />} />
          <Route
            path="/projects/microservices/blank"
            element={<ProjectMicroservicesBlank />}
          />
          <Route path="*" element={<Navigate to="/about" replace />} />
        </Routes>
      </main>
    </div>
  );
}

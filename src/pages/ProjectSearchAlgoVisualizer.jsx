import { useState } from "react";
import { Link } from "react-router-dom";

export default function ProjectSearchAlgoVisualizer() {
  const tabs = ["JavaScript", "Python", "C++"];
  const baseUrl = import.meta.env.BASE_URL;
  const pseudocode = {
    dijkstra: {
      JavaScript: `function dijkstra(grid, start, goal) {
  const dist = new Map();
  const prev = new Map();
  const pq = new MinHeap();
  dist.set(start, 0);
  pq.push({ node: start, cost: 0 });

  while (!pq.isEmpty()) {
    const { node: current, cost } = pq.popMin();
    if (current === goal) break;
    if (cost !== dist.get(current)) continue;

    for (const neighbor of grid.neighbors(current)) {
      const alt = cost + grid.weight(current, neighbor);
      if (alt < (dist.get(neighbor) ?? Infinity)) {
        dist.set(neighbor, alt);
        prev.set(neighbor, current);
        pq.push({ node: neighbor, cost: alt });
      }
    }
  }

  return reconstructPath(prev, start, goal);
}`,
      Python: `def dijkstra(grid, start, goal):
    dist = {start: 0}
    prev = {}
    pq = [(0, start)]

    while pq:
        cost, current = heapq.heappop(pq)
        if current == goal:
            break
        if cost != dist.get(current, float("inf")):
            continue

        for neighbor in grid.neighbors(current):
            alt = cost + grid.weight(current, neighbor)
            if alt < dist.get(neighbor, float("inf")):
                dist[neighbor] = alt
                prev[neighbor] = current
                heapq.heappush(pq, (alt, neighbor))

    return reconstruct_path(prev, start, goal)`,
      "C++": `vector<Node> dijkstra(Grid& grid, Node start, Node goal) {
  unordered_map<Node, int> dist;
  unordered_map<Node, Node> prev;
  priority_queue<State, vector<State>, greater<State>> pq;
  dist[start] = 0;
  pq.push({0, start});

  while (!pq.empty()) {
    auto [cost, current] = pq.top();
    pq.pop();
    if (current == goal) break;
    if (cost != dist[current]) continue;

    for (auto neighbor : grid.neighbors(current)) {
      int alt = cost + grid.weight(current, neighbor);
      if (!dist.count(neighbor) || alt < dist[neighbor]) {
        dist[neighbor] = alt;
        prev[neighbor] = current;
        pq.push({alt, neighbor});
      }
    }
  }

  return reconstruct_path(prev, start, goal);
}`,
    },
    aStar: {
      JavaScript: `function aStar(grid, start, goal, heuristic) {
  const gScore = new Map();
  const fScore = new Map();
  const prev = new Map();
  const pq = new MinHeap();
  gScore.set(start, 0);
  fScore.set(start, heuristic(start, goal));
  pq.push({ node: start, cost: fScore.get(start) });

  while (!pq.isEmpty()) {
    const { node: current } = pq.popMin();
    if (current === goal) break;

    for (const neighbor of grid.neighbors(current)) {
      const tentative = gScore.get(current) + grid.weight(current, neighbor);
      if (tentative < (gScore.get(neighbor) ?? Infinity)) {
        prev.set(neighbor, current);
        gScore.set(neighbor, tentative);
        fScore.set(neighbor, tentative + heuristic(neighbor, goal));
        pq.push({ node: neighbor, cost: fScore.get(neighbor) });
      }
    }
  }

  return reconstructPath(prev, start, goal);
}`,
      Python: `def a_star(grid, start, goal, heuristic):
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    prev = {}
    pq = [(f_score[start], start)]

    while pq:
        _, current = heapq.heappop(pq)
        if current == goal:
            break

        for neighbor in grid.neighbors(current):
            tentative = g_score[current] + grid.weight(current, neighbor)
            if tentative < g_score.get(neighbor, float("inf")):
                prev[neighbor] = current
                g_score[neighbor] = tentative
                f_score[neighbor] = tentative + heuristic(neighbor, goal)
                heapq.heappush(pq, (f_score[neighbor], neighbor))

    return reconstruct_path(prev, start, goal)`,
      "C++": `vector<Node> a_star(Grid& grid, Node start, Node goal, Heuristic h) {
  unordered_map<Node, int> g_score;
  unordered_map<Node, int> f_score;
  unordered_map<Node, Node> prev;
  priority_queue<State, vector<State>, greater<State>> pq;
  g_score[start] = 0;
  f_score[start] = h(start, goal);
  pq.push({f_score[start], start});

  while (!pq.empty()) {
    auto [_, current] = pq.top();
    pq.pop();
    if (current == goal) break;

    for (auto neighbor : grid.neighbors(current)) {
      int tentative = g_score[current] + grid.weight(current, neighbor);
      if (!g_score.count(neighbor) || tentative < g_score[neighbor]) {
        prev[neighbor] = current;
        g_score[neighbor] = tentative;
        f_score[neighbor] = tentative + h(neighbor, goal);
        pq.push({f_score[neighbor], neighbor});
      }
    }
  }

  return reconstruct_path(prev, start, goal);
}`,
    },
  };

  const [dijkstraTab, setDijkstraTab] = useState(tabs[0]);
  const [aStarTab, setAStarTab] = useState(tabs[0]);

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
          src={`${baseUrl}search-algo-visualizer/index.html`}
        />
      </div>
      <div className="algo-grid">
        <article className="panel algo-panel">
          <header className="algo-header">
            <h2>Dijkstra's Algorithm</h2>
            <p>Baseline shortest-path search with non-negative edge weights.</p>
          </header>
          <div className="tab-row" role="tablist" aria-label="Dijkstra pseudocode tabs">
            {tabs.map((tab) => (
              <button
                key={tab}
                type="button"
                className={`tab-button${dijkstraTab === tab ? " tab-button--active" : ""}`}
                onClick={() => setDijkstraTab(tab)}
                aria-selected={dijkstraTab === tab}
                role="tab"
              >
                {tab}
              </button>
            ))}
          </div>
          <div className="code-panel" role="tabpanel">
            <pre>{pseudocode.dijkstra[dijkstraTab]}</pre>
          </div>
        </article>
        <article className="panel algo-panel">
          <header className="algo-header">
            <h2>A* Search</h2>
            <p>Heuristic-driven search that prioritizes promising paths.</p>
          </header>
          <div className="tab-row" role="tablist" aria-label="A* pseudocode tabs">
            {tabs.map((tab) => (
              <button
                key={tab}
                type="button"
                className={`tab-button${aStarTab === tab ? " tab-button--active" : ""}`}
                onClick={() => setAStarTab(tab)}
                aria-selected={aStarTab === tab}
                role="tab"
              >
                {tab}
              </button>
            ))}
          </div>
          <div className="code-panel" role="tabpanel">
            <pre>{pseudocode.aStar[aStarTab]}</pre>
          </div>
        </article>
      </div>

      <div className="panel section-header">
        <h2>Algorithm Deep Dive</h2>
        <p>Both algorithms guarantee shortest paths on weighted grids, but they search the space differently.</p>
      </div>
      <div className="algo-detail-grid">
        <article className="panel algo-detail-card">
          <h3>Dijkstra: Guaranteed shortest path</h3>
          <ul className="detail-list">
            <li>
              <strong>How it works:</strong> Expands nodes in order of smallest known distance from the start using relaxation and a priority queue.
            </li>
            <li>
              <strong>Strengths:</strong> Optimal with non-negative weights, no heuristic required, reliable baseline on weighted graphs.
            </li>
            <li>
              <strong>Trade-offs:</strong> Explores broadly, so it can visit many nodes that are far from the goal.
            </li>
          </ul>
        </article>
        <article className="panel algo-detail-card">
          <h3>A*: Goal-directed search</h3>
          <ul className="detail-list">
            <li>
              <strong>How it works:</strong> Prioritizes nodes by f(n)=g(n)+h(n), combining path cost so far with a heuristic estimate to the goal.
            </li>
            <li>
              <strong>Strengths:</strong> With an admissible heuristic, it stays optimal while expanding fewer nodes in practice.
            </li>
            <li>
              <strong>Trade-offs:</strong> Heuristic quality matters; a weak heuristic behaves like Dijkstra, and overestimates can lose optimality.
            </li>
          </ul>
        </article>
      </div>

      <div className="panel">
        <h2>Why A* Is Faster in Most Cases</h2>
        <p className="muted-text">
          A* uses problem knowledge to focus the search. On grid maps with Manhattan or Euclidean heuristics, it typically expands fewer nodes than Dijkstra.
        </p>
        <ul className="detail-list">
          <li>
            <strong>Focused frontier:</strong> The heuristic pulls exploration toward the goal instead of radiating evenly.
          </li>
          <li>
            <strong>Earlier convergence:</strong> Promising routes rise to the top of the priority queue sooner.
          </li>
          <li>
            <strong>Same guarantees:</strong> With an admissible heuristic, A* remains optimal but does less work.
          </li>
        </ul>
      </div>
    </section>
  );
}

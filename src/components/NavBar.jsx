import { NavLink } from "react-router-dom";

const navItems = [
  { label: "About", to: "/about" },
  { label: "Resume", to: "/resume" },
  { label: "Research", to: "/research" },
  { label: "Projects", to: "/projects" },
];

const leftNavItems = navItems.slice(0, 2);
const rightNavItems = navItems.slice(2);

export default function NavBar() {
  return (
    <header className="nav-shell">
      <div className="nav-inner nav-split">
        <nav className="nav-bar nav-bar--left">
          {leftNavItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `nav-pill${isActive ? " nav-pill--active" : ""}`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
        <div className="nav-name">
          <strong>Patrick Marshall</strong>
        </div>
        <nav className="nav-bar nav-bar--right">
          {rightNavItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                `nav-pill${isActive ? " nav-pill--active" : ""}`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}

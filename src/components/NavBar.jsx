import { NavLink } from "react-router-dom";

const navItems = [
  { label: "About", to: "/about" },
  { label: "Resume", to: "/resume" },
  { label: "Research", to: "/research" },
  { label: "Projects", to: "/projects" },
];

export default function NavBar() {
  return (
    <header className="nav-shell">
      <nav className="nav-bar">
        {navItems.map((item) => (
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
    </header>
  );
}

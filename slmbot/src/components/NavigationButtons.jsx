import { Link } from "react-router-dom";
import "./NavigationButtons.css";

const NavigationButtons = ({ currentPage }) => {
  const pages = [
    { path: "/", name: "Start" },
    { path: "/analytics", name: "Analytics" },
    { path: "/simulation", name: "Simulation" },
    { path: "/predictions", name: "Predictions" },
  ];

  const currentIndex = pages.findIndex((page) => page.path === currentPage);

  return (
    <div className="navigation-buttons">
      {currentIndex > 0 && (
        <Link
          to={pages[currentIndex - 1].path}
          className="nav-button prev-button"
        >
          ← {pages[currentIndex - 1].name}
        </Link>
      )}

      {currentIndex < pages.length - 1 && (
        <Link
          to={pages[currentIndex + 1].path}
          className="nav-button next-button"
        >
          {pages[currentIndex + 1].name} →
        </Link>
      )}
    </div>
  );
};

export default NavigationButtons;

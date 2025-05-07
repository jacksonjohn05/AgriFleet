import { Routes, Route } from "react-router-dom"; // Only import what you need
import StartSession from "./components/StartSession";
import AnalyticsPage from "./components/AnalyticsPage";
import RoverSimulation from "./components/RoverSimulation";
import Predictions from "./components/Predictions";
// import NavHeader from "./components/NavHeader";

function App() {
  return (
    <>
      {/* <NavHeader /> */}
      <Routes>
        {" "}
        {/* No Router wrapper here */}
        <Route path="/" element={<StartSession />} />
        <Route path="/analytics" element={<AnalyticsPage />} />
        <Route path="/simulation" element={<RoverSimulation />} />
        <Route path="/predictions" element={<Predictions />} />
      </Routes>
    </>
  );
}

export default App;

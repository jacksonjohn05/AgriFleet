import DiseasePrediction from "./DiseasePrediction";
import InsectClassification from "./InsectClassification";
import "./Predictions.css";
import NavigationButtons from "./NavigationButtons";

function Predictions() {
  return (
    <div className="predictions-section">
      <h2 className="title">Field Predictive Tools</h2>
      <div className="predictions-container">
        <DiseasePrediction />
        <InsectClassification />
      </div>
      <NavigationButtons currentPage="/predictions" />
    </div>
  );
}

export default Predictions;

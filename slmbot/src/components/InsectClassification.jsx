import React, { useState } from "react";
import axios from "axios";
import "./PredictionComponents.css"; // Shared CSS file

const InsectClassification = () => {
  const [insectFile, setInsectFile] = useState(null);
  const [insectImagePreview, setInsectImagePreview] = useState(null);
  const [insectPrediction, setInsectPrediction] = useState("No result yet.");
  const [insectUploadStatus, setInsectUploadStatus] = useState("");
  const [insectUploadProgress, setInsectUploadProgress] = useState(0);

  const handleInsectFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setInsectFile(file);
      setInsectUploadStatus("Ready to upload");
      setInsectUploadProgress(0);

      const reader = new FileReader();
      reader.onloadend = () => {
        setInsectImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleInsectPredict = async () => {
    if (!insectFile) {
      alert("Please select an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("file", insectFile);

    try {
      setInsectUploadStatus("Uploading...");
      const response = await axios.post(
        "http://127.0.0.1:8000/api/insect/predict/",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setInsectUploadProgress(percentCompleted);
            setInsectUploadStatus(`Uploading: ${percentCompleted}%`);
          },
        }
      );

      setInsectPrediction(response.data.class);
      setInsectUploadStatus("Uploaded successfully!");
    } catch (error) {
      console.error("Error classifying insect:", error);
      setInsectPrediction("Error classifying insect.");
      setInsectUploadStatus("Upload failed. Please try again.");
    }
  };

  return (
    <div className="prediction-card insect-classification">
      <h3>Entomology Classification</h3>
      <div className="input-container">
        <input
          className="input-file"
          type="file"
          accept="image/*"
          id="insect-image"
          onChange={handleInsectFileChange}
        />
        <label htmlFor="insect-image" className="upload-label">
          <i className="fa-solid fa-upload"></i> Upload Image
        </label>
      </div>
      {insectImagePreview && (
        <div className="image-preview">
          <img src={insectImagePreview} alt="Insect Preview" />
        </div>
      )}
      <button className="predict-button" onClick={handleInsectPredict}>
        Classify
      </button>
      <div className="result-area">
        <h4>Result:</h4>
        <p>{insectPrediction}</p>
      </div>
      {insectUploadStatus && (
        <div className="upload-status">
          <p>{insectUploadStatus}</p>
          {insectUploadProgress > 0 && (
            <div className="progress-bar">
              <div
                className="progress"
                style={{ width: `${insectUploadProgress}%` }}
              ></div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default InsectClassification;

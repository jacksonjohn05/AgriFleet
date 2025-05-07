import React, { useState } from "react";
import axios from "axios";
import "./PredictionComponents.css"; // We'll create this CSS file

const DiseasePrediction = () => {
  const [selectedFile, setSelectedFile] = useState(null);
  const [diseaseImagePreview, setDiseaseImagePreview] = useState(null);
  const [prediction, setPrediction] = useState("No result yet.");
  const [uploadStatus, setUploadStatus] = useState("");
  const [uploadProgress, setUploadProgress] = useState(0);

  const handleFileChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setUploadStatus("Ready to upload");
      setUploadProgress(0);

      const reader = new FileReader();
      reader.onloadend = () => {
        setDiseaseImagePreview(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handlePredict = async () => {
    if (!selectedFile) {
      alert("Please select an image first.");
      return;
    }

    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      setUploadStatus("Uploading...");
      const response = await axios.post(
        "http://127.0.0.1:8000/api/plant/predict/",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
          onUploadProgress: (progressEvent) => {
            const percentCompleted = Math.round(
              (progressEvent.loaded * 100) / progressEvent.total
            );
            setUploadProgress(percentCompleted);
            setUploadStatus(`Uploading: ${percentCompleted}%`);
          },
        }
      );

      setPrediction(response.data.prediction);
      setUploadStatus("Uploaded successfully!");
    } catch (error) {
      console.error("Error predicting disease:", error);
      setPrediction("Error predicting disease.");
      setUploadStatus("Upload failed. Please try again.");
    }
  };

  return (
    <div className="prediction-card disease-prediction">
      <h3>Disease Prediction</h3>
      <div className="input-container">
        <input
          className="input-file"
          type="file"
          accept="image/*"
          id="disease-image"
          onChange={handleFileChange}
        />
        <label htmlFor="disease-image" className="upload-label">
          <i className="fa-solid fa-upload"></i> Upload Image
        </label>
      </div>
      {diseaseImagePreview && (
        <div className="image-preview">
          <img src={diseaseImagePreview} alt="Disease Preview" />
        </div>
      )}
      <button className="predict-button" onClick={handlePredict}>
        Predict
      </button>
      <div className="result-area">
        <h4>Result:</h4>
        <p>{prediction}</p>
      </div>
      {uploadStatus && (
        <div className="upload-status">
          <p>{uploadStatus}</p>
          {uploadProgress > 0 && (
            <div className="progress-bar">
              <div
                className="progress"
                style={{ width: `${uploadProgress}%` }}
              ></div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default DiseasePrediction;

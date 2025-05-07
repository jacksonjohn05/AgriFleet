import React, { useState, useEffect } from "react";
import axios from "axios";
import "./styles/styles.css";
import NavigationButtons from "./NavigationButtons";

const RoverCard = ({ roverName, roverInfo }) => {
  const batteryLevel = roverInfo.battery;
  let batteryClass = "battery-high";
  if (batteryLevel <= 30) batteryClass = "battery-low";
  else if (batteryLevel <= 60) batteryClass = "battery-medium";

  let statusClass = "status-idle";
  if (roverInfo.status === "moving") statusClass = "status-moving";

  return (
    <div className="rover-card">
      <div className={`status-badge ${statusClass}`}>{roverInfo.status}</div>
      <h3 className="rover-name">{roverName}</h3>

      <div className="battery-container">
        <div className="battery-label">
          <span>Battery</span>
          <span>{batteryLevel}%</span>
        </div>
        <div className="battery-bar">
          <div
            className={`battery-level ${batteryClass}`}
            style={{ width: `${batteryLevel}%` }}
          ></div>
        </div>
      </div>

      <div className="rover-details">
        <div>
          <div className="detail-label">Coordinates</div>
          <div className="detail-value">
            ({roverInfo.coordinates.join(", ")})
          </div>
        </div>
        <div>
          <div className="detail-label">Current Task</div>
          <div className="detail-value">{roverInfo.task || "None"}</div>
        </div>
      </div>
    </div>
  );
};

const AnalyticsPanel = ({ fleetData }) => {
  return (
    <div className="analytics-panel">
      <h3>Fleet Analytics</h3>
      <div className="analytics-grid">
        <div className="metric-card">
          <div className="metric-value">{fleetData.length}</div>
          <div className="metric-label">Total Rovers</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">
            {fleetData.filter(([_, info]) => info.status === "moving").length}
          </div>
          <div className="metric-label">Active</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">
            {fleetData.filter(([_, info]) => info.battery <= 30).length}
          </div>
          <div className="metric-label">Low Battery</div>
        </div>
        <div className="metric-card">
          <div className="metric-value">
            {fleetData.filter(([_, info]) => info.task).length}
          </div>
          <div className="metric-label">On Task</div>
        </div>
      </div>
    </div>
  );
};

export default function StartSession() {
  const [sessionId, setSessionId] = useState("");
  const [fleetData, setFleetData] = useState([]);
  const [loading, setLoading] = useState({
    session: false,
    fleet: false,
  });
  const [error, setError] = useState(null);
  const [autoRefresh, setAutoRefresh] = useState(false);
  const [lastUpdated, setLastUpdated] = useState(null);

  useEffect(() => {
    const savedSession = localStorage.getItem("fleet_session_id");
    if (savedSession) {
      setSessionId(savedSession);
    }
  }, []);

  useEffect(() => {
    let interval;
    if (autoRefresh && sessionId) {
      fetchFleetStatus();
      interval = setInterval(fetchFleetStatus, 5000);
    }
    return () => clearInterval(interval);
  }, [autoRefresh, sessionId]);

  const handleStartSession = async () => {
    setLoading((prev) => ({ ...prev, session: true }));
    setError(null);
    try {
      const response = await axios.post("/api/session/start");
      setSessionId(response.data.session_id);
      localStorage.setItem("fleet_session_id", response.data.session_id);
    } catch (error) {
      console.error("Session start error:", error);
      setError("Failed to start session. Please try again.");
    } finally {
      setLoading((prev) => ({ ...prev, session: false }));
    }
  };

  const fetchFleetStatus = async () => {
    if (!sessionId) return;

    setLoading((prev) => ({ ...prev, fleet: true }));
    try {
      const response = await axios.get(
        `/api/fleet/status?session_id=${sessionId}`
      );
      setFleetData(Object.entries(response.data));
      setLastUpdated(new Date());
    } catch (error) {
      console.error("Fleet status error:", error);
      setError("Failed to fetch fleet status. Session may have expired.");
      setAutoRefresh(false);
    } finally {
      setLoading((prev) => ({ ...prev, fleet: false }));
    }
  };

  const handleEndSession = () => {
    setSessionId("");
    setFleetData([]);
    localStorage.removeItem("fleet_session_id");
    setAutoRefresh(false);
  };

  const handleControlAll = () => {
    console.log("Controlling all rovers");
  };

  const handleAnalytics = () => {
    console.log("Analytics of all rovers");
  };

  return (
    <div className="dashboard">
      <header className="header">
        <h1>AgriFleet Control Center</h1>
        {sessionId && (
          <div className="session-info">
            <div className="session-id-display">
              <span>Active Session:</span>
              <span className="session-id">{sessionId}</span>
            </div>
            <button className="button button-end" onClick={handleEndSession}>
              End Session
            </button>
          </div>
        )}
      </header>

      <div className="controls">
        <div className="control-group">
          <button
            className={`button button-primary ${
              loading.session ? "button-disabled" : ""
            }`}
            onClick={handleStartSession}
            disabled={loading.session || !!sessionId}
          >
            {loading.session ? "Starting..." : "Start Session"}
          </button>

          <button
            className={`button button-primary ${
              loading.fleet ? "button-disabled" : ""
            }`}
            onClick={fetchFleetStatus}
            disabled={loading.fleet || !sessionId}
          >
            {loading.fleet ? "Refreshing..." : "Refresh Fleet"}
          </button>

          <div className="refresh-controls">
            <button
              className={`button ${
                autoRefresh ? "button-stop" : "button-secondary"
              }`}
              onClick={() => setAutoRefresh(!autoRefresh)}
              disabled={!sessionId}
            >
              {autoRefresh ? "Stop Auto-Refresh" : "Auto-Refresh"}
            </button>
            {lastUpdated && (
              <span className="last-updated">
                Last updated: {lastUpdated.toLocaleTimeString()}
              </span>
            )}
          </div>
        </div>
      </div>

      {error && (
        <div className="error-message">
          <svg className="error-icon" viewBox="0 0 20 20">
            <path d="M10 18a8 8 0 100-16 8 8 0 000 16zM8.707 7.293a1 1 0 00-1.414 1.414L8.586 10l-1.293 1.293a1 1 0 101.414 1.414L10 11.414l1.293 1.293a1 1 0 001.414-1.414L11.414 10l1.293-1.293a1 1 0 00-1.414-1.414L10 8.586 8.707 7.293z" />
          </svg>
          {error}
        </div>
      )}

      <div className="content-area">
        <div className="rover-grid">
          {fleetData.length > 0 ? (
            fleetData.map(([roverName, roverInfo]) => (
              <RoverCard
                key={roverName}
                roverName={roverName}
                roverInfo={roverInfo}
              />
            ))
          ) : (
            <div className="empty-state">
              {sessionId
                ? "No rovers available"
                : "Start a session to begin monitoring"}
              {sessionId && (
                <button
                  className="button button-small"
                  onClick={fetchFleetStatus}
                >
                  Refresh
                </button>
              )}
            </div>
          )}
        </div>

        <AnalyticsPanel fleetData={fleetData} />
      </div>

      <div className="action-buttons">
        <NavigationButtons currentPage="/" />
        {/* <button 
          className="button button-control"
          onClick={handleControlAll}
          disabled={!sessionId || fleetData.length === 0}
        >
          Control All Rovers
        </button> */}
        {/* <button 
            className="button button-analytics"
            onClick={() => navigate('/analytics')}
            >
            View Analytics
        </button> */}
      </div>
    </div>
  );
}

import React, { useState, useEffect, useRef } from "react";
import axios from "axios";
import { Line } from "react-chartjs-2";
import { Chart, registerables } from "chart.js";
import "./styles/AnalyticsPage.css";
import NavigationButtons from "./NavigationButtons";

Chart.register(...registerables);

const AnalyticsPage = () => {
  const [selectedRovers, setSelectedRovers] = useState([]);
  const [isInitialLoading, setIsInitialLoading] = useState(false);
  const [roverStatus, setRoverStatus] = useState([]);
  const [analyticsData, setAnalyticsData] = useState({});
  const [autoRefresh, setAutoRefresh] = useState(false);
  const refreshInterval = useRef(null);
  const isFirstRun = useRef(true);
  const requestQueue = useRef([]);

  // Fetch initial rover status
  useEffect(() => {
    const fetchRoverStatus = async () => {
      try {
        const response = await axios.get("/api/fleetstatus");
        const fleetData = response.data[0];

        setRoverStatus(
          Object.entries(fleetData).map(([id, data]) => ({
            id,
            status: data.status,
            battery: data.battery,
            coordinates: data.coordinates,
            lastUpdated: new Date().toLocaleTimeString(),
          }))
        );
      } catch (error) {
        console.error("Error fetching rover status:", error);
        // Fallback mock data
        setRoverStatus([
          { id: "Rover-1", status: "idle", battery: 97, coordinates: [-2, -7] },
          { id: "Rover-2", status: "idle", battery: 75, coordinates: [2, -4] },
          { id: "Rover-3", status: "idle", battery: 81, coordinates: [3, 5] },
          { id: "Rover-4", status: "idle", battery: 68, coordinates: [6, 6] },
          { id: "Rover-5", status: "idle", battery: 79, coordinates: [-7, -6] },
        ]);
      }
    };
    fetchRoverStatus();
  }, []);

  // Handle auto-refresh
  useEffect(() => {
    if (autoRefresh) {
      const fetchDataSilently = async () => {
        try {
          await fetchAnalyticsData(false); // Silent refresh
        } catch (error) {
          console.error("Silent refresh failed:", error);
        }
      };

      refreshInterval.current = setInterval(fetchDataSilently, 5000);
    }

    return () => {
      if (refreshInterval.current) {
        clearInterval(refreshInterval.current);
      }
    };
  }, [autoRefresh, selectedRovers]);

  const handleCheckboxChange = (roverId) => {
    setSelectedRovers((prev) =>
      prev.includes(roverId)
        ? prev.filter((id) => id !== roverId)
        : [...prev, roverId]
    );
  };

  const fetchSensorData = async (roverId) => {
    try {
      // Cancel any pending requests for this rover
      const source = axios.CancelToken.source();
      requestQueue.current.push(source);

      const response = await axios.get(`/api/sensor-data?rover_id=${roverId}`, {
        cancelToken: source.token,
      });
      return response.data[0];
    } catch (error) {
      if (!axios.isCancel(error)) {
        console.error(`Error fetching data for ${roverId}:`, error);
        // Return realistic mock data
        return {
          timestamp: Date.now() / 1000,
          rover_id: roverId,
          soil_moisture: 30 + Math.random() * 40,
          soil_pH: 5 + Math.random() * 3,
          temperature: 20 + Math.random() * 15,
          battery_level: 20 + Math.random() * 80,
        };
      }
      return null; // Request was canceled
    }
  };

  const fetchAnalyticsData = async (showLoading = true) => {
    if (selectedRovers.length === 0) return;

    // Cancel any pending requests
    requestQueue.current.forEach((source) => source.cancel());
    requestQueue.current = [];

    if (showLoading) {
      setIsInitialLoading(true);
      isFirstRun.current = false;
    }

    const collectionDuration = 5000; // 5 seconds
    const startTime = Date.now();
    const allData = {};

    // Initialize data structure for all selected rovers
    selectedRovers.forEach((roverId) => {
      allData[roverId] = {
        moisture: [],
        ph: [],
        temperature: [],
        battery: [],
        timestamps: [],
      };
    });

    // Optimized for maximum requests - adjust based on your API rate limits
    const requestsPerSecond = 10; // Max requests per second (adjust as needed)
    const requestInterval = 1000 / requestsPerSecond;

    try {
      while (Date.now() - startTime < collectionDuration) {
        const batchStartTime = Date.now();

        // Process all rovers in parallel
        const promises = selectedRovers.map(async (roverId) => {
          const sensorData = await fetchSensorData(roverId);
          if (sensorData) {
            // Only process if not canceled
            const timestamp = new Date(
              sensorData.timestamp * 1000
            ).toLocaleTimeString();

            allData[roverId].moisture.push(sensorData.soil_moisture);
            allData[roverId].ph.push(sensorData.soil_pH);
            allData[roverId].temperature.push(sensorData.temperature);
            allData[roverId].battery.push(sensorData.battery_level);
            allData[roverId].timestamps.push(timestamp);
          }
        });

        await Promise.all(promises);

        // Calculate remaining time for this batch
        const batchTime = Date.now() - batchStartTime;
        const waitTime = Math.max(0, requestInterval - batchTime);
        await new Promise((resolve) => setTimeout(resolve, waitTime));
      }
    } catch (error) {
      if (!axios.isCancel(error)) {
        console.error("Error during data collection:", error);
      }
    } finally {
      setAnalyticsData(allData);
      if (showLoading) {
        setIsInitialLoading(false);
      }
    }
  };

  const createChartData = (roverId, metric) => {
    const data = analyticsData[roverId];
    if (!data || data.timestamps.length === 0) {
      return {
        labels: ["No data available"],
        datasets: [
          {
            label: `${metric.toUpperCase()} over time`,
            data: [0],
            borderColor: getColorForMetric(metric),
            backgroundColor: `${getColorForMetric(metric)}20`,
            tension: 0.1,
            fill: true,
          },
        ],
      };
    }

    return {
      labels: data.timestamps,
      datasets: [
        {
          label: `${metric.toUpperCase()} over time`,
          data: data[metric],
          borderColor: getColorForMetric(metric),
          backgroundColor: `${getColorForMetric(metric)}20`,
          tension: 0.1,
          fill: true,
          pointRadius: 1, // Smaller points for dense data
        },
      ],
    };
  };

  const getColorForMetric = (metric) => {
    switch (metric) {
      case "moisture":
        return "#3B82F6";
      case "ph":
        return "#10B981";
      case "temperature":
        return "#EF4444";
      case "battery":
        return "#F59E0B";
      default:
        return "#6B7280";
    }
  };

  const toggleAutoRefresh = () => {
    setAutoRefresh((prev) => !prev);
  };

  const handleRunAnalytics = () => {
    fetchAnalyticsData(true); // Show loading only for manual runs
  };

  return (
    <div className="analytics-container">
      <h1 className="fleet-analytics">Fleet Analytics</h1>

      <div className="rover-selection">
        <h3>Select Rovers for Analytics</h3>
        <div className="checkbox-group">
          {roverStatus.map((rover) => (
            <label key={rover.id} className="checkbox-label">
              <input
                type="checkbox"
                checked={selectedRovers.includes(rover.id)}
                onChange={() => handleCheckboxChange(rover.id)}
                disabled={isInitialLoading}
              />
              {rover.id}
            </label>
          ))}
        </div>
        <div className="action-buttons">
          <button
            className="run-button"
            onClick={handleRunAnalytics}
            disabled={selectedRovers.length === 0 || isInitialLoading}
          >
            {isInitialLoading ? "Collecting Data..." : "Run Analytics"}
          </button>
          <button
            className={`auto-refresh-button  ${autoRefresh ? "active" : ""}`}
            onClick={toggleAutoRefresh}
            disabled={selectedRovers.length === 0}
          >
            {autoRefresh ? "Auto Refresh ON" : "Auto Refresh OFF"}
          </button>
        </div>
      </div>

      {isInitialLoading && isFirstRun.current && (
        <div className="loading-overlay">
          <div className="loading-spinner"></div>
          <p>Collecting initial sensor data... Please wait</p>
        </div>
      )}

      <div className="analytics-results">
        {Object.keys(analyticsData).length > 0 &&
          selectedRovers.map(
            (roverId) =>
              analyticsData[roverId] && (
                <div key={roverId} className="rover-analytics-card">
                  <h2 className="rover-graph-title">
                    {roverId} Analytics{" "}
                    {autoRefresh && (
                      <span className="auto-refresh-indicator">(Auto)</span>
                    )}
                  </h2>
                  <div className="metrics-grid">
                    <div className="metric-chart-container">
                      <h3>Soil Moisture (%)</h3>
                      <div className="chart-wrapper">
                        <Line
                          data={createChartData(roverId, "moisture")}
                          options={chartOptions}
                        />
                      </div>
                    </div>
                    <div className="metric-chart-container">
                      <h3>Soil pH</h3>
                      <div className="chart-wrapper">
                        <Line
                          data={createChartData(roverId, "ph")}
                          options={chartOptions}
                        />
                      </div>
                    </div>
                    <div className="metric-chart-container">
                      <h3>Temperature (°C)</h3>
                      <div className="chart-wrapper">
                        <Line
                          data={createChartData(roverId, "temperature")}
                          options={chartOptions}
                        />
                      </div>
                    </div>
                    <div className="metric-chart-container">
                      <h3>Battery Level (%)</h3>
                      <div className="chart-wrapper">
                        <Line
                          data={createChartData(roverId, "battery")}
                          options={chartOptions}
                        />
                      </div>
                    </div>
                  </div>
                </div>
              )
          )}
      </div>
      <NavigationButtons currentPage="/analytics" />
    </div>
  );
};

const chartOptions = {
  responsive: true,
  maintainAspectRatio: false,
  plugins: {
    legend: {
      display: false,
    },
    tooltip: {
      callbacks: {
        label: (context) =>
          `${context.dataset.label}: ${context.parsed.y.toFixed(2)}`,
        // Only show the nearest point's tooltip for dense data
        mode: "nearest",
        intersect: false,
      },
    },
  },
  scales: {
    x: {
      ticks: {
        maxRotation: 45,
        minRotation: 45,
        autoSkip: true,
        maxTicksLimit: 10,
      },
      grid: {
        display: false,
      },
    },
    y: {
      beginAtZero: false,
    },
  },
  animation: {
    duration: 0,
  },
  elements: {
    point: {
      radius: 1, // Smaller points for dense data
      hoverRadius: 3, // Slightly larger on hover
    },
    line: {
      borderWidth: 1.5, // Thinner line for dense data
    },
  },
  interaction: {
    mode: "nearest",
    axis: "x",
    intersect: false,
  },
};

export default AnalyticsPage;

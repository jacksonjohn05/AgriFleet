import React, { useState, useEffect, useRef } from "react";
import "./styles/RoverSimulation.css";
import DiseasePrediction from "./DiseasePrediction";
import InsectClassification from "./InsectClassification";
import NavigationButtons from "./NavigationButtons";

const RoverSimulation = () => {
  // Grid dimensions
  const gridSize = 100;
  const cellSize = 5; // px

  // Rover colors with border effects
  const roverColors = [
    { main: "#4285F4", border: "#3367D6" }, // Blue
    { main: "#34A853", border: "#2D8E49" }, // Green
    { main: "#EA4335", border: "#D33426" }, // Red
    { main: "#FBBC05", border: "#E9AB04" }, // Yellow
    { main: "#673AB7", border: "#5E35B1" }, // Purple
  ];

  // Initial fleet positions
  const initialFleet = [
    {
      id: 1,
      x: 10,
      y: 10,
      color: roverColors[0],
      task: null,
      path: [],
      status: "idle",
      battery: 100,
    },
    {
      id: 2,
      x: 90,
      y: 10,
      color: roverColors[1],
      task: null,
      path: [],
      status: "idle",
      battery: 100,
    },
    {
      id: 3,
      x: 10,
      y: 90,
      color: roverColors[2],
      task: null,
      path: [],
      status: "idle",
      battery: 100,
    },
    {
      id: 4,
      x: 90,
      y: 90,
      color: roverColors[3],
      task: null,
      path: [],
      status: "idle",
      battery: 100,
    },
    {
      id: 5,
      x: 50,
      y: 50,
      color: roverColors[4],
      task: null,
      path: [],
      status: "idle",
      battery: 100,
    },
  ];

  // Task types with specific colors (all with 2 second duration)
  const taskTypes = [
    { name: "Soil Analysis", color: "#795548", textColor: "#FFFFFF" }, // Brown
    { name: "Irrigation", color: "#2196F3", textColor: "#FFFFFF" }, // Blue
    { name: "Weeding", color: "#4CAF50", textColor: "#FFFFFF" }, // Green
    { name: "Crop Monitoring", color: "#FF9800", textColor: "#000000" }, // Orange
  ];

  // Generate non-colliding random tasks
  const generateTasks = () => {
    const newTasks = [];
    const minDistance = 15; // Minimum distance between tasks

    for (let i = 1; i <= 5; i++) {
      const taskType = taskTypes[(i - 1) % taskTypes.length];
      let x, y, isValidPosition;
      let attempts = 0;
      const maxAttempts = 100;

      do {
        x = Math.floor(Math.random() * (gridSize - 20)) + 10;
        y = Math.floor(Math.random() * (gridSize - 20)) + 10;
        isValidPosition = true;

        for (const task of newTasks) {
          const distance = Math.sqrt(
            Math.pow(x - task.x, 2) + Math.pow(y - task.y, 2)
          );
          if (distance < minDistance) {
            isValidPosition = false;
            break;
          }
        }

        for (const rover of initialFleet) {
          const distance = Math.sqrt(
            Math.pow(x - rover.x, 2) + Math.pow(y - rover.y, 2)
          );
          if (distance < minDistance) {
            isValidPosition = false;
            break;
          }
        }

        attempts++;
        if (attempts >= maxAttempts) {
          minDistance = Math.max(5, minDistance - 2);
        }
      } while (!isValidPosition && attempts < maxAttempts * 2);

      newTasks.push({
        id: i,
        x,
        y,
        priority: Math.floor(Math.random() * 3) + 1,
        type: taskType.name,
        color: taskType.color,
        textColor: taskType.textColor,
        progress: 0,
        assignedTo: null,
        completed: false,
        startTime: null,
        completionTime: null,
      });
    }

    return newTasks;
  };

  // State management
  const [rovers, setRovers] = useState(initialFleet);
  const [tasks, setTasks] = useState(generateTasks());
  const [isSimulating, setIsSimulating] = useState(false);
  const [simulationSpeed, setSimulationSpeed] = useState(100);
  const [selectedEntity, setSelectedEntity] = useState(null);
  const [logs, setLogs] = useState([]);
  const [allTasksCompleted, setAllTasksCompleted] = useState(false);
  const simulationRef = useRef(null);
  const logsEndRef = useRef(null);
  const taskTimersRef = useRef({});

  // Add log entry
  const addLog = (message) => {
    const timestamp = new Date().toLocaleTimeString();
    setLogs((prev) => [...prev.slice(-99), { timestamp, message }]);
  };

  // Scroll logs to bottom
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs]);

  // Check if all tasks are completed
  useEffect(() => {
    if (
      tasks.length > 0 &&
      tasks.every((t) => t.completed) &&
      !allTasksCompleted
    ) {
      setAllTasksCompleted(true);
      setIsSimulating(false);
      addLog("All tasks completed! Simulation stopped.");
    }
  }, [tasks, allTasksCompleted]);

  // Clean up timers on unmount
  useEffect(() => {
    return () => {
      Object.values(taskTimersRef.current).forEach((timer) =>
        clearTimeout(timer)
      );
    };
  }, []);

  // Pathfinding function
  const calculatePath = (startX, startY, endX, endY) => {
    const path = [];
    let currentX = startX;
    let currentY = startY;

    const dx = Math.abs(endX - startX);
    const dy = Math.abs(endY - startY);
    const sx = startX < endX ? 1 : -1;
    const sy = startY < endY ? 1 : -1;
    let err = dx - dy;

    while (true) {
      if (currentX !== startX || currentY !== startY) {
        path.push({ x: currentX, y: currentY });
      }

      if (currentX === endX && currentY === endY) break;

      const e2 = 2 * err;
      if (e2 > -dy) {
        err -= dy;
        currentX += sx;
      }
      if (e2 < dx) {
        err += dx;
        currentY += sy;
      }
    }

    return path;
  };

  // Assign tasks to rovers
  const assignTasks = () => {
    const unassignedTasks = tasks
      .filter((t) => !t.completed && !t.assignedTo)
      .sort((a, b) => b.priority - a.priority);

    const availableRovers = rovers.filter(
      (r) => r.status === "idle" && r.battery > 10
    );

    unassignedTasks.forEach((task) => {
      if (availableRovers.length === 0) return;

      let bestScore = Infinity;
      let bestRover = null;

      availableRovers.forEach((rover) => {
        const distance = Math.sqrt(
          Math.pow(task.x - rover.x, 2) + Math.pow(task.y - rover.y, 2)
        );
        const score = distance * (1 + (100 - rover.battery) / 100);

        if (score < bestScore) {
          bestScore = score;
          bestRover = rover;
        }
      });

      if (bestRover) {
        const path = calculatePath(bestRover.x, bestRover.y, task.x, task.y);
        const batteryUsage = path.length * 0.1;

        if (bestRover.battery - batteryUsage > 10) {
          setRovers((prev) =>
            prev.map((r) =>
              r.id === bestRover.id
                ? {
                    ...r,
                    task,
                    path,
                    status: "moving",
                    battery: r.battery - batteryUsage,
                  }
                : r
            )
          );

          setTasks((prev) =>
            prev.map((t) =>
              t.id === task.id ? { ...t, assignedTo: bestRover.id } : t
            )
          );

          addLog(
            `Rover R${bestRover.id} assigned to Task T${task.id} (${task.type})`
          );

          availableRovers.splice(availableRovers.indexOf(bestRover), 1);
        }
      }
    });
  };

  // Complete a task after 2 seconds
  const completeTask = (roverId, taskId) => {
    // Clear any existing timer for this rover
    if (taskTimersRef.current[roverId]) {
      clearTimeout(taskTimersRef.current[roverId]);
    }

    // Set new timer
    taskTimersRef.current[roverId] = setTimeout(() => {
      const completionTime = new Date();

      setTasks((prev) =>
        prev.map((t) =>
          t.id === taskId
            ? {
                ...t,
                completed: true,
                progress: 100,
                assignedTo: null,
                completionTime,
              }
            : t
        )
      );

      setRovers((prev) =>
        prev.map((r) =>
          r.id === roverId
            ? {
                ...r,
                task: null,
                status: "idle",
                battery: Math.max(0, r.battery - 1),
              }
            : r
        )
      );

      addLog(`Rover R${roverId} completed Task T${taskId}`);

      // Remove the timer reference
      delete taskTimersRef.current[roverId];

      // Immediately assign new tasks
      assignTasks();
    }, 2000); // Fixed 2 second completion time
  };

  // Simulation step - handles movement
  const simulationStep = () => {
    // Process rovers that are moving
    setRovers((prev) =>
      prev.map((rover) => {
        if (rover.path.length > 0 && rover.status === "moving") {
          const nextStep = rover.path[0];
          const newPath = rover.path.slice(1);

          // If reached destination, start working on task
          if (newPath.length === 0) {
            addLog(
              `Rover R${rover.id} reached Task T${rover.task.id} and started working`
            );

            // Start task completion timer
            completeTask(rover.id, rover.task.id);

            return {
              ...rover,
              x: nextStep.x,
              y: nextStep.y,
              path: [],
              status: "working",
              battery: Math.max(0, rover.battery - 0.5),
            };
          }

          // Move to next step
          return {
            ...rover,
            x: nextStep.x,
            y: nextStep.y,
            path: newPath,
            battery: Math.max(0, rover.battery - 0.1),
          };
        }
        return rover;
      })
    );

    // Assign new tasks
    assignTasks();

    // Recharge idle rovers
    setRovers((prev) =>
      prev.map((rover) => {
        if (rover.status === "idle") {
          return {
            ...rover,
            battery: Math.min(100, rover.battery + 0.2),
          };
        }
        return rover;
      })
    );
  };

  // Start/stop simulation
  useEffect(() => {
    if (isSimulating && !allTasksCompleted) {
      simulationRef.current = setInterval(simulationStep, simulationSpeed);
      addLog("Simulation started");
    } else {
      clearInterval(simulationRef.current);
      if (simulationRef.current) {
        addLog("Simulation paused");
      }
    }

    return () => clearInterval(simulationRef.current);
  }, [isSimulating, simulationSpeed, allTasksCompleted]);

  const handleStartSimulation = () => {
    if (allTasksCompleted) {
      addLog("All tasks already completed. Reset to start new simulation.");
      return;
    }
    setIsSimulating(true);
  };

  const handleStopSimulation = () => {
    setIsSimulating(false);
  };

  const handleReset = () => {
    // Clear all task timers
    Object.values(taskTimersRef.current).forEach((timer) =>
      clearTimeout(timer)
    );
    taskTimersRef.current = {};

    setIsSimulating(false);
    setRovers(initialFleet);
    setTasks(generateTasks());
    setLogs([]);
    setAllTasksCompleted(false);
    addLog("Simulation reset with 10 new random tasks");
  };

  const handleEntityClick = (entity, type) => {
    setSelectedEntity({ ...entity, type });
  };

  const getTaskCompletionTime = (task) => {
    if (!task.completed) return null;
    if (!task.completionTime) return "Just completed";

    const seconds = Math.round(
      (new Date() - new Date(task.completionTime)) / 1000
    );
    return `${seconds} second${seconds !== 1 ? "s" : ""} ago`;
  };

  return (
    <div className="simulation-container">
      <div className="simulation-header">
        <h2>Agricultural Rover Fleet Simulation</h2>
        <div className="control-panel">
          <button
            onClick={handleStartSimulation}
            disabled={isSimulating || allTasksCompleted}
            className="control-btn start-btn"
          >
            ▶ Start
          </button>
          <button
            onClick={handleStopSimulation}
            disabled={!isSimulating}
            className="control-btn stop-btn"
          >
            ⏸ Pause
          </button>
          <button onClick={handleReset} className="control-btn reset-btn">
            ↻ Reset
          </button>

          <div className="speed-control">
            <span>Speed:</span>
            <input
              type="range"
              min="50"
              max="500"
              step="50"
              value={simulationSpeed}
              onChange={(e) => setSimulationSpeed(Number(e.target.value))}
            />
            <span className="speed-value">{1000 / simulationSpeed}x</span>
          </div>

          {allTasksCompleted && (
            <div className="completion-message">All tasks completed!</div>
          )}
        </div>
      </div>

      <div className="main-content">
        <div className="left-panel">
          <div className="tasks-section">
            <h3>
              Task Queue ({tasks.filter((t) => !t.completed).length} remaining)
            </h3>
            <div className="task-list">
              {tasks.map((task) => (
                <div
                  key={`task-${task.id}`}
                  className={`task-item ${task.assignedTo ? "assigned" : ""} ${
                    task.completed ? "completed" : ""
                  }`}
                  onClick={() => handleEntityClick(task, "task")}
                >
                  <div
                    className="task-color"
                    style={{ backgroundColor: task.color }}
                  />
                  <div className="task-info">
                    <div className="task-header">
                      <span className="task-id">T{task.id}</span>
                      <span className="task-type">{task.type}</span>
                      <span className="task-priority">P{task.priority}</span>
                    </div>
                    <div className="task-location">
                      ({task.x}, {task.y})
                    </div>
                    {task.completed ? (
                      <div className="task-status completed">Completed</div>
                    ) : task.assignedTo ? (
                      <div className="task-status in-progress">
                        In Progress (Rover R{task.assignedTo})
                      </div>
                    ) : (
                      <div className="task-status pending">Pending</div>
                    )}
                    {task.assignedTo && !task.completed && (
                      <div className="task-progress">
                        <div
                          className="progress-bar"
                          style={{
                            width: `${task.progress}%`,
                            backgroundColor: task.color,
                          }}
                        />
                      </div>
                    )}
                  </div>
                </div>
              ))}
            </div>
          </div>

          {selectedEntity && (
            <div className="details-section">
              <h3>
                {selectedEntity.type === "rover"
                  ? `Rover R${selectedEntity.id} Details`
                  : `Task T${selectedEntity.id} Details`}
              </h3>
              <div className="details-content">
                {selectedEntity.type === "rover" ? (
                  <>
                    <div className="detail-row">
                      <span className="detail-label">Status:</span>
                      <span
                        className={`detail-value status-${selectedEntity.status}`}
                      >
                        {selectedEntity.status}
                      </span>
                    </div>
                    <div className="detail-row">
                      <span className="detail-label">Position:</span>
                      <span className="detail-value">
                        ({selectedEntity.x}, {selectedEntity.y})
                      </span>
                    </div>
                    <div className="detail-row">
                      <span className="detail-label">Battery:</span>
                      <div className="battery-meter">
                        <div
                          className="battery-level"
                          style={{ width: `${selectedEntity.battery}%` }}
                        />
                        <span className="battery-text">
                          {Math.round(selectedEntity.battery)}%
                        </span>
                      </div>
                    </div>
                    {selectedEntity.task && (
                      <>
                        <div className="detail-row">
                          <span className="detail-label">Current Task:</span>
                          <span className="detail-value">
                            {selectedEntity.task.type} (T
                            {selectedEntity.task.id})
                          </span>
                        </div>
                        <div className="detail-row">
                          <span className="detail-label">Task Progress:</span>
                          <div className="progress-container">
                            <div
                              className="progress-bar"
                              style={{
                                width: `${selectedEntity.task.progress}%`,
                                backgroundColor: selectedEntity.task.color,
                              }}
                            />
                          </div>
                        </div>
                      </>
                    )}
                  </>
                ) : (
                  <>
                    <div className="detail-row">
                      <span className="detail-label">Type:</span>
                      <span className="detail-value">
                        {selectedEntity.type}
                      </span>
                    </div>
                    <div className="detail-row">
                      <span className="detail-label">Priority:</span>
                      <span className="detail-value">
                        {selectedEntity.priority}
                      </span>
                    </div>
                    <div className="detail-row">
                      <span className="detail-label">Location:</span>
                      <span className="detail-value">
                        ({selectedEntity.x}, {selectedEntity.y})
                      </span>
                    </div>
                    <div className="detail-row">
                      <span className="detail-label">Duration:</span>
                      <span className="detail-value">2 seconds</span>
                    </div>
                    <div className="detail-row">
                      <span className="detail-label">Status:</span>
                      <span
                        className={`detail-value ${
                          selectedEntity.completed
                            ? "status-completed"
                            : selectedEntity.assignedTo
                            ? "status-in-progress"
                            : "status-pending"
                        }`}
                      >
                        {selectedEntity.completed
                          ? `Completed ${getTaskCompletionTime(selectedEntity)}`
                          : selectedEntity.assignedTo
                          ? `In Progress (Rover R${selectedEntity.assignedTo})`
                          : "Pending"}
                      </span>
                    </div>
                  </>
                )}
              </div>
            </div>
          )}
        </div>

        <div className="center-panel">
          <div className="grid-map-container">
            <div
              className="grid-map"
              style={{
                "--grid-size": gridSize,
                "--cell-size": `${cellSize}px`,
              }}
            >
              {Array.from({ length: gridSize }).map((_, y) =>
                Array.from({ length: gridSize }).map((_, x) => (
                  <div
                    key={`${x}-${y}`}
                    className="grid-cell"
                    data-x={x}
                    data-y={y}
                  />
                ))
              )}

              {tasks.map((task) => (
                <div
                  key={`task-${task.id}`}
                  className={`task-marker ${
                    task.assignedTo ? "assigned" : ""
                  } ${task.completed ? "completed" : ""}`}
                  style={{
                    left: `${task.x * cellSize}px`,
                    top: `${task.y * cellSize}px`,
                    backgroundColor: task.color,
                    borderColor: task.color,
                    opacity: task.completed ? 0.7 : 1,
                  }}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleEntityClick(task, "task");
                  }}
                >
                  <div className="task-id" style={{ color: task.textColor }}>
                    T{task.id}
                  </div>
                  {task.assignedTo && !task.completed && (
                    <div className="assigned-rover">R{task.assignedTo}</div>
                  )}
                  {task.completed ? (
                    <div className="task-completed">✓</div>
                  ) : (
                    task.assignedTo && (
                      <div className="task-progress-ring">
                        <div
                          className="progress-fill"
                          style={{
                            transform: `rotate(${task.progress * 3.6}deg)`,
                            backgroundColor: task.color,
                          }}
                        />
                      </div>
                    )
                  )}
                </div>
              ))}

              {rovers.map((rover) => (
                <div
                  key={`rover-${rover.id}`}
                  className={`rover ${rover.status}`}
                  style={{
                    left: `${rover.x * cellSize}px`,
                    top: `${rover.y * cellSize}px`,
                    backgroundColor: rover.color.main,
                    borderColor: rover.color.border,
                    boxShadow: `0 0 0 2px ${rover.color.border}`,
                  }}
                  onClick={(e) => {
                    e.stopPropagation();
                    handleEntityClick(rover, "rover");
                  }}
                >
                  <div className="rover-id">R{rover.id}</div>

                  <div className={`rover-status ${rover.status}`}>
                    {rover.status === "idle"
                      ? "I"
                      : rover.status === "moving"
                      ? "M"
                      : "W"}
                  </div>

                  <div className="rover-battery">
                    <div
                      className="battery-level"
                      style={{
                        width: `${rover.battery}%`,
                        backgroundColor:
                          rover.battery > 30 ? "#4CAF50" : "#F44336",
                      }}
                    />
                  </div>

                  {rover.path.length > 0 && (
                    <div className="rover-path">
                      {rover.path.map((step, i) => (
                        <div
                          key={`path-${rover.id}-${i}`}
                          className="path-step"
                          style={{
                            left: `${step.x * cellSize}px`,
                            top: `${step.y * cellSize}px`,
                            backgroundColor: rover.color.main,
                          }}
                        />
                      ))}
                    </div>
                  )}

                  {rover.task && (
                    <div className="rover-task-indicator">T{rover.task.id}</div>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>

        <div className="right-panel">
          <div className="event-log">
            <h3>Event Log</h3>
            <div className="log-entries">
              {logs.map((log, index) => (
                <div key={index} className="log-entry">
                  <span className="log-timestamp">[{log.timestamp}]</span>
                  <span className="log-message">{log.message}</span>
                </div>
              ))}
              <div ref={logsEndRef} />
            </div>
          </div>
        </div>
      </div>
      <NavigationButtons currentPage="/simulation" />
    </div>
  );
};

export default RoverSimulation;

class TrajectoryVisualizer {
    constructor() {
        // Wait for DOM to be ready
        const canvas = document.getElementById('canvas');
        if (!canvas) {
            console.error('Canvas element not found');
            return;
        }
        
        this.canvas = canvas;
        this.ctx = this.canvas.getContext('2d');
        this.sceneData = null;
        this.currentScene = null;
        this.viewMode = 'overview';
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        this.isDragging = false;
        this.lastMouseX = 0;
        this.lastMouseY = 0;
        
        this.initializeEventListeners();
        this.resizeCanvas();
    }
    
    initializeEventListeners() {
        // File input
        const fileInput = document.getElementById('fileInput');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.loadSceneFile(e.target.files[0]));
        }
        
        // Scene selection
        const sceneSelect = document.getElementById('sceneSelect');
        if (sceneSelect) {
            sceneSelect.addEventListener('change', (e) => this.selectScene(e.target.value));
        }
        
        // View options - updated to match new checkbox structure
        const showTrajectories = document.getElementById('showTrajectories');
        const showLaneGraph = document.getElementById('showLaneGraph');
        const showCandidates = document.getElementById('showCandidates');
        
        if (showTrajectories) {
            showTrajectories.addEventListener('change', () => this.render());
        }
        if (showLaneGraph) {
            showLaneGraph.addEventListener('change', () => this.render());
        }
        if (showCandidates) {
            showCandidates.addEventListener('change', () => this.render());
        }
        
        // Canvas mouse events
        this.canvas.addEventListener('mousedown', (e) => this.onMouseDown(e));
        this.canvas.addEventListener('mousemove', (e) => this.onMouseMove(e));
        this.canvas.addEventListener('mouseup', (e) => this.onMouseUp(e));
        this.canvas.addEventListener('wheel', (e) => this.onWheel(e));
        
        // Window resize
        window.addEventListener('resize', () => this.resizeCanvas());
    }
    
    resizeCanvas() {
        const container = this.canvas.parentElement;
        const rect = container.getBoundingClientRect();
        this.canvas.width = rect.width;
        this.canvas.height = rect.height;
        this.render();
    }
    
    async loadSceneFile(file) {
        if (!file) return;
        
        try {
            const text = await file.text();
            const data = JSON.parse(text);
            
            // Check if it's a single scene or multiple scenes
            if (data.scene_id) {
                // Single scene
                this.sceneData = { [data.scene_id]: data };
                this.populateSceneSelect([data.scene_id]);
                this.selectScene(data.scene_id);
            } else if (typeof data === 'object') {
                // Multiple scenes or metadata
                this.sceneData = data;
                const sceneIds = Object.keys(data).filter(key => 
                    data[key].scene_id || key.startsWith('scene_')
                );
                this.populateSceneSelect(sceneIds);
            }
            
            this.showStatus('Scene data loaded successfully', 'success');
        } catch (error) {
            this.showStatus(`Error loading file: ${error.message}`, 'error');
        }
    }
    
    loadScenes(scenes, metadata) {
        this.sceneData = scenes;
        this.metadata = metadata;
        this.populateSceneSelect(Object.keys(scenes));
        this.showStatus('Scenes loaded successfully!', 'success');
    }
    
    populateSceneSelect(sceneIds) {
        const select = document.getElementById('sceneSelect');
        select.innerHTML = '<option value="">Select a scene...</option>';
        
        sceneIds.forEach(id => {
            const option = document.createElement('option');
            option.value = id;
            option.textContent = id;
            select.appendChild(option);
        });
        
        select.disabled = false;
    }
    
    selectScene(sceneId) {
        if (!sceneId || !this.sceneData) return;
        
        this.currentScene = this.sceneData[sceneId];
        if (!this.currentScene) {
            this.showStatus(`Scene ${sceneId} not found`, 'error');
            return;
        }
        
        this.updateSceneInfo();
        this.resetView();
        this.render();
    }
    
    updateSceneInfo() {
        if (!this.currentScene) return;
        
        // Update individual info elements
        document.getElementById('sceneId').textContent = this.currentScene.scene_id || '-';
        document.getElementById('timestamp').textContent = this.currentScene.timestamp || '-';
        document.getElementById('scenarioType').textContent = this.currentScene.scenario_type || '-';
        
        // Calculate agent count
        const numAgents = this.currentScene.agent_trajectories ? 
            this.currentScene.agent_trajectories.length : 0;
        document.getElementById('agentCount').textContent = numAgents;
    }
    
    resetView() {
        this.scale = 1;
        this.offsetX = 0;
        this.offsetY = 0;
        
        if (this.currentScene && this.viewMode === 'ego-centered') {
            // Center on ego vehicle if available
            const egoState = this.currentScene.state_features;
            if (egoState && egoState.length >= 2) {
                this.offsetX = -egoState[0] * this.scale + this.canvas.width / 2;
                this.offsetY = -egoState[1] * this.scale + this.canvas.height / 2;
            }
        }
        
        this.render();
    }
    
    render() {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        
        if (!this.currentScene) {
            this.drawPlaceholder();
            return;
        }
        
        this.ctx.save();
        this.ctx.translate(this.offsetX, this.offsetY);
        this.ctx.scale(this.scale, this.scale);
        
        // Draw lane graph if enabled
        if (document.getElementById('showLaneGraph').checked) {
            this.drawLaneGraph();
        }
        
        // Draw agent trajectories if enabled
        if (document.getElementById('showTrajectories').checked) {
            this.drawAgentTrajectories();
        }
        
        // Draw candidate trajectories if enabled
        if (document.getElementById('showCandidates').checked) {
            this.drawCandidateTrajectories();
        }
        
        // Always draw ego vehicle
        this.drawEgoVehicle();
        
        this.ctx.restore();
        
        // Draw UI elements
        this.drawCoordinateSystem();
    }
    
    drawPlaceholder() {
        this.ctx.fillStyle = '#666';
        this.ctx.font = '18px Arial';
        this.ctx.textAlign = 'center';
        this.ctx.fillText(
            'Load a scene file to begin visualization',
            this.canvas.width / 2,
            this.canvas.height / 2
        );
    }
    
    drawLaneGraph() {
        if (!this.currentScene.lane_graph) return;
        
        this.ctx.strokeStyle = '#666666';
        this.ctx.lineWidth = 2;
        
        // Parse lane graph data
        let laneGraph = null;
        if (this.currentScene.lane_graph.raw_data) {
            try {
                let rawData = this.currentScene.lane_graph.raw_data;
                if (typeof rawData === 'string') {
                    // Convert Python dict format to JSON format
                    rawData = rawData
                        .replace(/'/g, '"')  // Replace single quotes with double quotes
                        .replace(/True/g, 'true')  // Replace Python True with JSON true
                        .replace(/False/g, 'false')  // Replace Python False with JSON false
                        .replace(/None/g, 'null');  // Replace Python None with JSON null
                }
                laneGraph = JSON.parse(rawData);
            } catch (e) {
                console.warn('Failed to parse lane graph:', e);
                return;
            }
        } else {
            laneGraph = this.currentScene.lane_graph;
        }
        
        if (laneGraph && laneGraph.lanes) {
            laneGraph.lanes.forEach(lane => {
                if (lane.centerline && lane.centerline.length > 1) {
                    this.ctx.beginPath();
                    const firstPoint = this.worldToCanvas(lane.centerline[0][0], lane.centerline[0][1]);
                    this.ctx.moveTo(firstPoint.x, firstPoint.y);
                    
                    for (let i = 1; i < lane.centerline.length; i++) {
                        const point = this.worldToCanvas(lane.centerline[i][0], lane.centerline[i][1]);
                        this.ctx.lineTo(point.x, point.y);
                    }
                    this.ctx.stroke();
                }
            });
        }
    }
    
    drawAgentTrajectories() {
        if (!this.currentScene.agent_trajectories) return;
        
        this.ctx.fillStyle = '#4444ff';
        this.ctx.strokeStyle = '#4444ff';
        
        this.currentScene.agent_trajectories.forEach((agent, index) => {
            if (agent.length >= 2) {
                const x = agent[0];
                const y = -agent[1]; // Flip Y coordinate
                
                // Draw agent as a rectangle
                this.ctx.fillRect(x - 1, y - 2, 2, 4);
                
                // Draw heading if available
                if (agent.length >= 7) {
                    const heading = agent[6];
                    this.ctx.save();
                    this.ctx.translate(x, y);
                    this.ctx.rotate(heading);
                    this.ctx.beginPath();
                    this.ctx.moveTo(0, 0);
                    this.ctx.lineTo(3, 0);
                    this.ctx.stroke();
                    this.ctx.restore();
                }
            }
        });
    }
    
    drawCandidateTrajectories() {
        if (!this.currentScene.candidate_trajectories) return;
        
        this.ctx.strokeStyle = '#44ff44';
        this.ctx.lineWidth = 2;
        
        this.currentScene.candidate_trajectories.forEach(candidate => {
            if (candidate.waypoints && candidate.waypoints.length > 1) {
                this.ctx.beginPath();
                this.ctx.moveTo(candidate.waypoints[0][0], -candidate.waypoints[0][1]);
                
                for (let i = 1; i < candidate.waypoints.length; i++) {
                    this.ctx.lineTo(candidate.waypoints[i][0], -candidate.waypoints[i][1]);
                }
                
                this.ctx.stroke();
                
                // Draw waypoints
                this.ctx.fillStyle = '#44ff44';
                candidate.waypoints.forEach(waypoint => {
                    this.ctx.beginPath();
                    this.ctx.arc(waypoint[0], -waypoint[1], 1, 0, 2 * Math.PI);
                    this.ctx.fill();
                });
            }
        });
    }
    
    drawEgoVehicle() {
        const egoState = this.currentScene.state_features;
        if (!egoState || egoState.length < 2) return;
        
        const x = egoState[0];
        const y = -egoState[1]; // Flip Y coordinate
        const heading = egoState.length >= 7 ? egoState[6] : 0;
        
        this.ctx.save();
        this.ctx.translate(x, y);
        this.ctx.rotate(heading);
        
        // Draw ego vehicle as a larger rectangle
        this.ctx.fillStyle = '#ff4444';
        this.ctx.fillRect(-2, -1, 4, 2);
        
        // Draw direction arrow
        this.ctx.strokeStyle = '#ff4444';
        this.ctx.lineWidth = 2;
        this.ctx.beginPath();
        this.ctx.moveTo(2, 0);
        this.ctx.lineTo(4, 0);
        this.ctx.moveTo(3, -1);
        this.ctx.lineTo(4, 0);
        this.ctx.lineTo(3, 1);
        this.ctx.stroke();
        
        this.ctx.restore();
    }
    
    drawCoordinateSystem() {
        // Draw scale indicator
        this.ctx.fillStyle = '#333';
        this.ctx.font = '12px Arial';
        this.ctx.textAlign = 'left';
        this.ctx.fillText(`Scale: ${this.scale.toFixed(2)}x`, 10, 20);
        
        // Draw coordinate grid (optional)
        if (this.scale > 0.5) {
            this.ctx.strokeStyle = '#eee';
            this.ctx.lineWidth = 1;
            
            const gridSize = 10;
            const startX = Math.floor(-this.offsetX / this.scale / gridSize) * gridSize;
            const startY = Math.floor(-this.offsetY / this.scale / gridSize) * gridSize;
            
            for (let x = startX; x < startX + this.canvas.width / this.scale + gridSize; x += gridSize) {
                this.ctx.beginPath();
                this.ctx.moveTo(x * this.scale + this.offsetX, 0);
                this.ctx.lineTo(x * this.scale + this.offsetX, this.canvas.height);
                this.ctx.stroke();
            }
            
            for (let y = startY; y < startY + this.canvas.height / this.scale + gridSize; y += gridSize) {
                this.ctx.beginPath();
                this.ctx.moveTo(0, y * this.scale + this.offsetY);
                this.ctx.lineTo(this.canvas.width, y * this.scale + this.offsetY);
                this.ctx.stroke();
            }
        }
    }
    
    // Mouse event handlers
    onMouseDown(e) {
        this.isDragging = true;
        this.lastMouseX = e.clientX;
        this.lastMouseY = e.clientY;
        this.canvas.style.cursor = 'grabbing';
    }
    
    onMouseMove(e) {
        if (this.isDragging) {
            const deltaX = e.clientX - this.lastMouseX;
            const deltaY = e.clientY - this.lastMouseY;
            
            this.offsetX += deltaX;
            this.offsetY += deltaY;
            
            this.lastMouseX = e.clientX;
            this.lastMouseY = e.clientY;
            
            this.render();
        }
    }
    
    onMouseUp(e) {
        this.isDragging = false;
        this.canvas.style.cursor = 'crosshair';
    }
    
    onWheel(e) {
        e.preventDefault();
        
        const rect = this.canvas.getBoundingClientRect();
        const mouseX = e.clientX - rect.left;
        const mouseY = e.clientY - rect.top;
        
        const scaleFactor = e.deltaY > 0 ? 0.9 : 1.1;
        const newScale = Math.max(0.1, Math.min(10, this.scale * scaleFactor));
        
        // Zoom towards mouse position
        this.offsetX = mouseX - (mouseX - this.offsetX) * (newScale / this.scale);
        this.offsetY = mouseY - (mouseY - this.offsetY) * (newScale / this.scale);
        this.scale = newScale;
        
        this.render();
    }
    
    showStatus(message, type = 'success') {
        const statusDiv = document.getElementById('status');
        if (statusDiv) {
            // Remove existing classes
            statusDiv.className = 'status';
            
            // Add new class and animation
            statusDiv.classList.add(type);
            statusDiv.textContent = message;
            
            // Add loading state for async operations
            if (message.includes('Loading') || message.includes('loading')) {
                statusDiv.classList.add('loading');
            }
            
            // Auto-hide success messages after 3 seconds
            if (type === 'success') {
                setTimeout(() => {
                    statusDiv.style.opacity = '0';
                    setTimeout(() => {
                        statusDiv.textContent = 'Ready to load trajectory data';
                        statusDiv.className = 'status';
                        statusDiv.style.opacity = '1';
                    }, 300);
                }, 3000);
            }
        }
    }
}

// Initialize the visualizer when the page loads
document.addEventListener('DOMContentLoaded', () => {
    // Add a small delay to ensure DOM is fully rendered
    setTimeout(() => {
        // Initialize visualizer
        window.visualizer = new TrajectoryVisualizer();
        
        // Load test data function
        window.loadTestData = async function() {
            try {
                // Load metadata
                const metadataResponse = await fetch('./metadata.json');
                const metadata = await metadataResponse.json();
                
                // Load scene data
                const sceneResponse = await fetch('./scene_training_001.json');
                const sceneData = await sceneResponse.json();
                
                // Create scenes object
                const scenes = {
                    [sceneData.scene_id]: sceneData
                };
                
                visualizer.loadScenes(scenes, metadata);
                visualizer.showStatus('Test data loaded successfully!', 'success');
            } catch (error) {
                console.error('Error loading test data:', error);
                visualizer.showStatus('Failed to load test data: ' + error.message, 'error');
            }
        };
    }, 100);
});
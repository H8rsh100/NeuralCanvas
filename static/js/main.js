/**
 * NeuralCanvas — Frontend Logic
 * Handles tab switching, API calls, Cytoscape.js rendering,
 * node interactions, legend building, and PNG export.
 */

document.addEventListener('DOMContentLoaded', function () {

    // ─── State ─────────────────────────────────────────────
    let cy = null;
    let currentMode = 'topic';

    // ─── DOM Elements ──────────────────────────────────────
    const tabs = document.querySelectorAll('.tab');
    const topicInput = document.getElementById('topic-input');
    const textInput = document.getElementById('text-input');
    const topicField = document.getElementById('topic-field');
    const textField = document.getElementById('text-field');
    const generateTopicBtn = document.getElementById('generate-topic-btn');
    const generateTextBtn = document.getElementById('generate-text-btn');
    const statusMessage = document.getElementById('status-message');
    const statusText = document.getElementById('status-text');
    const statusSpinner = document.getElementById('status-spinner');
    const emptyState = document.getElementById('empty-state');
    const nodeInfo = document.getElementById('node-info');
    const infoTitle = document.getElementById('info-title');
    const infoCategory = document.getElementById('info-category');
    const infoCluster = document.getElementById('info-cluster');
    const infoScore = document.getElementById('info-score');
    const infoClose = document.getElementById('info-close');
    const exportBtn = document.getElementById('export-btn');
    const legendPanel = document.getElementById('legend-panel');
    const clusterLegend = document.getElementById('cluster-legend');
    const statsBar = document.getElementById('stats-bar');
    const statNodes = document.getElementById('stat-nodes');
    const statEdges = document.getElementById('stat-edges');
    const statClusters = document.getElementById('stat-clusters');

    // ─── Category Colors ───────────────────────────────────
    const categoryColors = {
        'Definition': '#38bdf8',
        'Process': '#a78bfa',
        'Example': '#facc15',
        'Theory': '#f472b6',
        'Application': '#34d399'
    };

    const categoryIcons = {
        'Definition': '📖',
        'Process': '⚙️',
        'Example': '💡',
        'Theory': '🔬',
        'Application': '🚀'
    };

    // ─── Tab Switching ─────────────────────────────────────
    tabs.forEach(tab => {
        tab.addEventListener('click', function () {
            tabs.forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            currentMode = this.dataset.mode;

            if (currentMode === 'topic') {
                topicInput.classList.add('active');
                textInput.classList.remove('active');
            } else {
                textInput.classList.add('active');
                topicInput.classList.remove('active');
            }
        });
    });

    // ─── Status Helpers ────────────────────────────────────
    function showStatus(message, type = 'loading') {
        statusMessage.classList.add('visible');
        statusMessage.classList.remove('error', 'success');
        statusText.textContent = message;

        if (type === 'loading') {
            statusSpinner.style.display = 'block';
        } else {
            statusSpinner.style.display = 'none';
            statusMessage.classList.add(type);
        }
    }

    function hideStatus() {
        statusMessage.classList.remove('visible');
    }

    function setButtonsDisabled(disabled) {
        generateTopicBtn.disabled = disabled;
        generateTextBtn.disabled = disabled;
    }

    // ─── Generate Handlers ─────────────────────────────────
    generateTopicBtn.addEventListener('click', function () {
        const topic = topicField.value.trim();
        if (!topic) {
            showStatus('Please enter a topic name.', 'error');
            setTimeout(hideStatus, 3000);
            return;
        }
        generateMap('/api/generate-from-topic', { topic: topic });
    });

    generateTextBtn.addEventListener('click', function () {
        const text = textField.value.trim();
        if (!text) {
            showStatus('Please paste some text or notes.', 'error');
            setTimeout(hideStatus, 3000);
            return;
        }
        if (text.length < 20) {
            showStatus('Text is too short. Please provide more detailed content.', 'error');
            setTimeout(hideStatus, 3000);
            return;
        }
        generateMap('/api/generate-from-text', { text: text });
    });

    // Allow Enter key on topic field
    topicField.addEventListener('keydown', function (e) {
        if (e.key === 'Enter') {
            generateTopicBtn.click();
        }
    });

    // ─── API Call & Pipeline ───────────────────────────────
    async function generateMap(endpoint, body) {
        setButtonsDisabled(true);
        hideNodeInfo();

        // Show progressive status messages
        const steps = endpoint.includes('topic')
            ? ['Calling Gemini API...', 'Extracting concepts...', 'Building relationships...', 'Clustering concepts...', 'Rendering map...']
            : ['Extracting concepts...', 'Building relationships...', 'Clustering concepts...', 'Rendering map...'];

        let stepIndex = 0;
        showStatus(steps[stepIndex], 'loading');

        const stepInterval = setInterval(() => {
            stepIndex++;
            if (stepIndex < steps.length) {
                showStatus(steps[stepIndex], 'loading');
            }
        }, 2000);

        try {
            const response = await fetch(endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(body)
            });

            clearInterval(stepInterval);

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.error || 'Server error');
            }

            const data = await response.json();

            if (data.error) {
                showStatus(data.error, 'error');
                setTimeout(hideStatus, 4000);
                setButtonsDisabled(false);
                return;
            }

            if (!data.nodes || data.nodes.length === 0) {
                showStatus('No concepts could be extracted. Try different input.', 'error');
                setTimeout(hideStatus, 4000);
                setButtonsDisabled(false);
                return;
            }

            // Render the graph
            renderGraph(data);
            showStatus(`Map generated — ${data.nodes.length} concepts, ${data.edges.length} relations`, 'success');
            setTimeout(hideStatus, 4000);

        } catch (error) {
            clearInterval(stepInterval);
            console.error('Generation error:', error);
            showStatus(`Error: ${error.message}`, 'error');
            setTimeout(hideStatus, 5000);
        }

        setButtonsDisabled(false);
    }

    // ─── Cytoscape.js Rendering ────────────────────────────
    function renderGraph(data) {
        // Hide empty state
        emptyState.style.display = 'none';

        // Build Cytoscape elements
        const elements = [];

        // Add nodes
        data.nodes.forEach(node => {
            const score = node.score || 0.5;
            const size = Math.max(30, Math.min(70, score * 120 + 25));

            elements.push({
                group: 'nodes',
                data: {
                    id: node.id,
                    label: node.label,
                    cluster: node.cluster,
                    color: node.color,
                    category: node.category,
                    score: score,
                    size: size
                },
                position: {
                    x: node.x,
                    y: node.y
                }
            });
        });

        // Add edges
        data.edges.forEach((edge, idx) => {
            elements.push({
                group: 'edges',
                data: {
                    id: 'e' + idx,
                    source: edge.source,
                    target: edge.target,
                    weight: edge.weight || 0.5,
                    label: edge.label || ''
                }
            });
        });

        // Destroy existing instance
        if (cy) {
            cy.destroy();
        }

        // Initialize Cytoscape
        cy = cytoscape({
            container: document.getElementById('cy'),
            elements: elements,
            layout: { name: 'preset' },  // Use PCA x/y positions
            style: [
                // Node style
                {
                    selector: 'node',
                    style: {
                        'label': 'data(label)',
                        'width': 'data(size)',
                        'height': 'data(size)',
                        'background-color': 'data(color)',
                        'border-width': 2,
                        'border-color': 'data(color)',
                        'border-opacity': 0.4,
                        'color': '#f1f5f9',
                        'text-valign': 'bottom',
                        'text-halign': 'center',
                        'font-size': '11px',
                        'font-family': 'Inter, sans-serif',
                        'font-weight': 500,
                        'text-margin-y': 8,
                        'text-wrap': 'wrap',
                        'text-max-width': '100px',
                        'overlay-opacity': 0,
                        'shadow-blur': 15,
                        'shadow-color': 'data(color)',
                        'shadow-opacity': 0.3,
                        'shadow-offset-x': 0,
                        'shadow-offset-y': 0,
                        'transition-property': 'width, height, border-width, shadow-blur',
                        'transition-duration': '0.2s'
                    }
                },
                // Node hover
                {
                    selector: 'node:active',
                    style: {
                        'overlay-opacity': 0.08,
                        'overlay-color': '#38bdf8'
                    }
                },
                // Selected / highlighted node
                {
                    selector: 'node.highlighted',
                    style: {
                        'border-width': 4,
                        'border-color': '#ffffff',
                        'shadow-blur': 25,
                        'shadow-opacity': 0.6,
                        'z-index': 999
                    }
                },
                // Dimmed nodes (not connected to selected)
                {
                    selector: 'node.dimmed',
                    style: {
                        'opacity': 0.2,
                        'transition-property': 'opacity',
                        'transition-duration': '0.3s'
                    }
                },
                // Edge style
                {
                    selector: 'edge',
                    style: {
                        'width': function (ele) {
                            return Math.max(1, Math.min(5, ele.data('weight') * 4));
                        },
                        'line-color': 'rgba(148, 163, 184, 0.25)',
                        'target-arrow-color': 'rgba(148, 163, 184, 0.25)',
                        'curve-style': 'bezier',
                        'label': 'data(label)',
                        'font-size': '9px',
                        'font-family': 'Inter, sans-serif',
                        'color': 'rgba(148, 163, 184, 0.6)',
                        'text-rotation': 'autorotate',
                        'text-margin-y': -8,
                        'text-background-color': '#0a0e1a',
                        'text-background-opacity': 0.8,
                        'text-background-padding': '3px',
                        'overlay-opacity': 0,
                        'transition-property': 'line-color, opacity',
                        'transition-duration': '0.3s'
                    }
                },
                // Highlighted edge (connected to selected node)
                {
                    selector: 'edge.highlighted',
                    style: {
                        'line-color': '#38bdf8',
                        'width': function (ele) {
                            return Math.max(2, Math.min(6, ele.data('weight') * 5));
                        },
                        'color': '#f1f5f9',
                        'z-index': 999
                    }
                },
                // Dimmed edge
                {
                    selector: 'edge.dimmed',
                    style: {
                        'opacity': 0.08,
                        'transition-property': 'opacity',
                        'transition-duration': '0.3s'
                    }
                }
            ],
            // Interaction options
            minZoom: 0.3,
            maxZoom: 3,
            wheelSensitivity: 0.3,
            boxSelectionEnabled: false,
            autounselectify: true
        });

        // ─── Node Click Handler ────────────────────────────
        cy.on('tap', 'node', function (evt) {
            const node = evt.target;
            highlightNode(node);
            showNodeInfo(node.data());
        });

        // Click on background to reset
        cy.on('tap', function (evt) {
            if (evt.target === cy) {
                resetHighlights();
                hideNodeInfo();
            }
        });

        // ─── Build Legend & Stats ──────────────────────────
        buildClusterLegend(data.nodes);
        updateStats(data);

        // Enable export
        exportBtn.disabled = false;
        legendPanel.style.display = 'block';

        // Fit view with padding
        cy.fit(undefined, 50);
    }

    // ─── Highlight Connected Nodes ─────────────────────────
    function highlightNode(node) {
        resetHighlights();

        const connectedEdges = node.connectedEdges();
        const connectedNodes = connectedEdges.connectedNodes();

        // Dim everything
        cy.elements().addClass('dimmed');

        // Highlight selected node + connected
        node.removeClass('dimmed').addClass('highlighted');
        connectedNodes.removeClass('dimmed');
        connectedEdges.removeClass('dimmed').addClass('highlighted');
    }

    function resetHighlights() {
        if (!cy) return;
        cy.elements().removeClass('dimmed highlighted');
    }

    // ─── Node Info Panel ───────────────────────────────────
    function showNodeInfo(data) {
        infoTitle.textContent = data.label;
        infoCategory.textContent = data.category || 'Unknown';
        infoCategory.style.backgroundColor = (categoryColors[data.category] || '#64748b') + '22';
        infoCategory.style.color = categoryColors[data.category] || '#64748b';
        infoCluster.textContent = 'Cluster ' + (data.cluster !== undefined ? data.cluster : '—');
        infoCluster.style.color = data.color || '#38bdf8';
        infoScore.textContent = (data.score !== undefined ? data.score.toFixed(3) : '—');
        nodeInfo.classList.add('visible');
    }

    function hideNodeInfo() {
        nodeInfo.classList.remove('visible');
    }

    infoClose.addEventListener('click', function () {
        hideNodeInfo();
        resetHighlights();
    });

    // ─── Cluster Legend ────────────────────────────────────
    function buildClusterLegend(nodes) {
        const clusters = {};
        nodes.forEach(node => {
            if (!(node.cluster in clusters)) {
                clusters[node.cluster] = node.color;
            }
        });

        clusterLegend.innerHTML = '';
        Object.entries(clusters).forEach(([id, color]) => {
            const item = document.createElement('div');
            item.className = 'legend-item';
            item.innerHTML = `<span class="legend-dot" style="background:${color}"></span> Cluster ${id}`;
            clusterLegend.appendChild(item);
        });
    }

    // ─── Stats Bar ─────────────────────────────────────────
    function updateStats(data) {
        const clusterSet = new Set(data.nodes.map(n => n.cluster));
        statNodes.textContent = data.nodes.length;
        statEdges.textContent = data.edges.length;
        statClusters.textContent = clusterSet.size;
        statsBar.classList.add('visible');
    }

    // ─── Export PNG ────────────────────────────────────────
    exportBtn.addEventListener('click', function () {
        if (!cy) return;

        const png = cy.png({
            bg: '#0a0e1a',
            full: true,
            scale: 2,
            maxWidth: 4000,
            maxHeight: 3000
        });

        const link = document.createElement('a');
        link.href = png;
        link.download = 'neuralcanvas-concept-map.png';
        link.click();
    });
});

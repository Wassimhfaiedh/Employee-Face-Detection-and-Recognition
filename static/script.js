const socket = io.connect(window.location.origin); // Connect to Flask server
let frameDimensions = { width: 1920, height: 1080 };
let isProcessorReady = false;
let activeCameraId = null;
let isPaused = false;
let lockedDetections = [];
let known_count1 = 0;
let visitor_count1 = 0;
let known_count2 = 0;
let visitor_count2 = 0;
let selectedPersonId = null;
let selectedCameraId = null;

function updateAnalytics(knownPercent, visitorPercent) {
    if (isPaused) return;
    const elements = [
        { id: 'known-percent', value: knownPercent },
        { id: 'visitor-percent', value: visitorPercent }
    ];

    elements.forEach(({ id, value }) => {
        const element = document.getElementById(id);
        element.textContent = value + '%';
        element.setAttribute('data-value', value);
    });
}

function calculateAnalytics() {
    let known_c = 0;
    let visitor_c = 0;
    if (activeCameraId === '1') {
        known_c = known_count1;
        visitor_c = visitor_count1;
    } else if (activeCameraId === '2') {
        known_c = known_count2;
        visitor_c = visitor_count2;
    } else if (activeCameraId === 'both') {
        known_c = known_count1 + known_count2;
        visitor_c = visitor_count1 + visitor_count2;
    }
    const total = known_c + visitor_c;
    const known_p = total > 0 ? Math.round((known_c / total) * 100) : 0;
    const visitor_p = 100 - known_p;
    updateAnalytics(known_p, visitor_p);
}

function updateDetections(detections) {
    if (isPaused) return;
    const employeeList = document.getElementById('employee-detections-list');
    const visitorList = document.getElementById('visitor-detections-list');

    detections.forEach(detection => {
        if (!lockedDetections.some(locked => locked.name === detection.name && locked.face_id === detection.face_id)) {
            lockedDetections.push(detection);
        }
    });

    lockedDetections = lockedDetections.slice(-10);

    const employees = lockedDetections.filter(d => d.name !== "Visitor");
    const visitors = lockedDetections.filter(d => d.name === "Visitor");

    employeeList.innerHTML = '';
    if (employees.length === 0) {
        employeeList.innerHTML = '<div class="detection-placeholder">No recognized employees yet</div>';
    } else {
        employees.reverse().forEach(detection => {
            const detectionCard = document.createElement('div');
            detectionCard.className = `detection-card known ${detection.face_id === selectedPersonId ? 'selected' : ''}`;
            detectionCard.dataset.faceId = detection.face_id;
            detectionCard.dataset.cameraId = detection.camera_id || '1';
            detectionCard.onclick = () => showPersonModal(detection);
            detectionCard.innerHTML = `
                <img src="${detection.face_image}" class="detection-image">
                <div class="detection-info">
                    <div class="detection-name">${detection.name}</div>
                    <div class="detection-timestamp">${detection.timestamp}</div>
                </div>
            `;
            employeeList.appendChild(detectionCard);
        });
    }

    visitorList.innerHTML = '';
    if (visitors.length === 0) {
        visitorList.innerHTML = '<div class="detection-placeholder">No detected visitors yet</div>';
    } else {
        visitors.reverse().forEach(detection => {
            const detectionCard = document.createElement('div');
            detectionCard.className = `detection-card visitor ${detection.face_id === selectedPersonId ? 'selected' : ''}`;
            detectionCard.dataset.faceId = detection.face_id;
            detectionCard.dataset.cameraId = detection.camera_id || '1';
            detectionCard.onclick = () => showPersonModal(detection);
            detectionCard.innerHTML = `
                <img src="${detection.face_image}" class="detection-image">
                <div class="detection-info">
                    <div class="detection-name">${detection.name}</div>
                    <div class="detection-timestamp">${detection.timestamp}</div>
                </div>
            `;
            visitorList.appendChild(detectionCard);
        });
    }
}

function showPersonModal(detection) {
    selectedPersonId = detection.face_id;
    selectedCameraId = detection.camera_id || '1';
    document.getElementById('modal-person-name').textContent = detection.name;
    document.getElementById('modal-person-id').textContent = detection.face_id;
    document.getElementById('modal-person-timestamp').textContent = detection.timestamp;
    document.getElementById('modal-person-camera').textContent = `CCTV - Camera ${detection.camera_id === '1' ? 'A' : 'B'}`;
    document.getElementById('modal-person-image').src = detection.face_image;
    document.getElementById('person-modal').style.display = 'flex';
    socket.emit('request_person_frame', { face_id: detection.face_id, camera_id: selectedCameraId });
    socket.emit('highlight_person', { face_id: detection.face_id, camera_id: selectedCameraId });

    document.querySelectorAll('.detection-card').forEach(card => {
        card.classList.remove('selected');
        if (card.dataset.faceId === detection.face_id) {
            card.classList.add('selected');
        }
    });
}

function closeModal() {
    document.getElementById('person-modal').style.display = 'none';
    selectedPersonId = null;
    selectedCameraId = null;
    socket.emit('highlight_person', { face_id: null, camera_id: null });
    document.querySelectorAll('.detection-card').forEach(card => card.classList.remove('selected'));
    const modalCanvas = document.getElementById('modal-canvas');
    const context = modalCanvas.getContext('2d');
    context.fillStyle = '#000';
    context.fillRect(0, 0, modalCanvas.width, modalCanvas.height);
}

function updateStatus(pid, message, isError = false) {
    const statusElement = document.getElementById(`status${pid}`);
    statusElement.textContent = `Status: ${message}`;
    statusElement.className = isError ? 'status-message status-error' : 'status-message';
}

function setActiveCamera(cameraId) {
    document.querySelectorAll('.camera-tile').forEach(tile => {
        tile.classList.remove('active');
        if (tile.getAttribute('data-camera-id') === cameraId) {
            tile.classList.add('active');
        }
    });
}

function selectCamera(cameraId) {
    if (isPaused) return;
    if (isProcessorReady && activeCameraId === cameraId) return;
    console.log(`Selecting CCTV ${cameraId}`);
    isProcessorReady = true;
    activeCameraId = cameraId;
    socket.emit('select_camera', { camera_id: cameraId });
    setActiveCamera(cameraId);
    document.getElementById('pause-resume-btn').disabled = false;
    isPaused = false;
    document.getElementById('pause-resume-btn').className = 'btn btn-pause';
    document.getElementById('pause-resume-btn').innerHTML = '<i class="fas fa-pause"></i> Pause Detection';

    const canvasContainer = document.getElementById('canvas-container');
    const wrapper1 = document.getElementById('canvas-wrapper-1');
    const wrapper2 = document.getElementById('canvas-wrapper-2');
    const status1 = document.getElementById('status1');
    const status2 = document.getElementById('status2');

    canvasContainer.classList.remove('both-cameras');
    wrapper1.style.display = 'none';
    wrapper2.style.display = 'none';
    status1.style.display = 'none';
    status2.style.display = 'none';

    if (cameraId === '1') {
        wrapper1.style.display = 'block';
        status1.style.display = 'block';
        canvasContainer.style.height = '720px';
    } else if (cameraId === '2') {
        wrapper2.style.display = 'block';
        status2.style.display = 'block';
        canvasContainer.style.height = '720px';
    } else if (cameraId === 'both') {
        canvasContainer.classList.add('both-cameras');
        wrapper1.style.display = 'block';
        wrapper2.style.display = 'block';
        status1.style.display = 'block';
        status2.style.display = 'block';
        canvasContainer.style.height = '720px';
    }
}

function drawPaused(canvas) {
    const context = canvas.getContext('2d');
    context.fillStyle = '#000';
    context.fillRect(0, 0, canvas.width, canvas.height);
    context.fillStyle = '#FFF';
    context.font = '24px Roboto Mono';
    context.textAlign = 'center';
    context.textBaseline = 'middle';
    context.fillText('System Paused', canvas.width / 2, canvas.height / 2);
}

function togglePauseResume() {
    if (!isProcessorReady) return;
    isPaused = !isPaused;
    socket.emit('toggle_pause_resume', {});
    const button = document.getElementById('pause-resume-btn');

    if (isPaused) {
        button.className = 'btn btn-resume';
        button.innerHTML = '<i class="fas fa-play"></i> Resume Detection';
        if (document.getElementById('canvas-wrapper-1').style.display === 'block') {
            drawPaused(canvas1);
        }
        if (document.getElementById('canvas-wrapper-2').style.display === 'block') {
            drawPaused(canvas2);
        }
        updateDetections([]);
        updateAnalytics(0, 0);
        lockedDetections = [];
        closeModal();
        updateStatus(1, 'System fully paused');
        updateStatus(2, 'System fully paused');
    } else {
        button.className = 'btn btn-pause';
        button.innerHTML = '<i class="fas fa-pause"></i> Pause Detection';
        updateStatus(1, 'System fully resumed');
        updateStatus(2, 'System fully resumed');
        if (!activeCameraId) {
            const context1 = canvas1.getContext('2d');
            context1.fillStyle = '#000';
            context1.fillRect(0, 0, canvas1.width, canvas1.height);
            context1.fillStyle = '#FFF';
            context1.font = '24px Roboto Mono';
            context1.textAlign = 'center';
            context1.textBaseline = 'middle';
            context1.fillText('Select a CCTV feed below', canvas1.width / 2, canvas1.height / 2);
            const context2 = canvas2.getContext('2d');
            context2.fillStyle = '#000';
            context2.fillRect(0, 0, canvas2.width, canvas2.height);
            context2.fillStyle = '#FFF';
            context2.font = '24px Roboto Mono';
            context2.textAlign = 'center';
            context2.textBaseline = 'middle';
            context2.fillText('Select a CCTV feed below', canvas2.width / 2, canvas2.height / 2);
        }
    }
}

socket.on('connect', () => {
    console.log('Connected to SocketIO server');
    updateStatus(1, 'Connected to face recognition server');
    updateStatus(2, 'Connected to face recognition server');
    document.querySelector('.status-indicator .status-dot:not(.recording)').style.background = 'var(--success)';
    document.querySelector('.status-indicator .status-dot:not(.recording)').style.boxShadow = '0 0 10px var(--success)';
});

socket.on('disconnect', () => {
    console.log('Disconnected from SocketIO server');
    updateStatus(1, 'Disconnected from server', true);
    updateStatus(2, 'Disconnected from server', true);
    updateAnalytics(0, 0);
    updateDetections([]);
    lockedDetections = [];
    document.querySelectorAll('.camera-tile').forEach(tile => tile.classList.remove('active'));
    document.getElementById('pause-resume-btn').disabled = true;
    isPaused = false;
    isProcessorReady = false;
    activeCameraId = null;
    document.getElementById('pause-resume-btn').className = 'btn btn-pause';
    document.getElementById('pause-resume-btn').innerHTML = '<i class="fas fa-pause"></i> Pause Detection';
    const context1 = canvas1.getContext('2d');
    context1.fillStyle = '#000';
    context1.fillRect(0, 0, canvas1.width, canvas1.height);
    context1.fillStyle = '#FFF';
    context1.font = '24px Roboto Mono';
    context1.textAlign = 'center';
    context1.textBaseline = 'middle';
    context1.fillText('Select a CCTV feed below', canvas1.width / 2, canvas1.height / 2);
    const context2 = canvas2.getContext('2d');
    context2.fillStyle = '#000';
    context2.fillRect(0, 0, canvas2.width, canvas2.height);
    context2.fillStyle = '#FFF';
    context2.font = '24px Roboto Mono';
    context2.textAlign = 'center';
    context2.textBaseline = 'middle';
    context2.fillText('Select a CCTV feed below', canvas2.width / 2, canvas2.height / 2);
    closeModal();
    document.querySelector('.status-indicator .status-dot:not(.recording)').style.background = 'var(--gray)';
    document.querySelector('.status-indicator .status-dot:not(.recording)').style.boxShadow = '0 0 8px rgba(0, 0, 0, 0.5)';
});

socket.on('video_frame_1', (data) => {
    if (isPaused) return;
    const img = new Image();
    img.src = `data:image/jpeg;base64,${data.frame}`;
    img.onload = () => {
        context1.drawImage(img, 0, 0, canvas1.width, canvas1.height);
        if (selectedCameraId === '1' && selectedPersonId) {
            const modalCanvas = document.getElementById('modal-canvas');
            const modalContext = modalCanvas.getContext('2d');
            modalContext.drawImage(img, 0, 0, modalCanvas.width, modalCanvas.height);
        }
    };
});

socket.on('video_frame_2', (data) => {
    if (isPaused) return;
    const img = new Image();
    img.src = `data:image/jpeg;base64,${data.frame}`;
    img.onload = () => {
        context2.drawImage(img, 0, 0, canvas2.width, canvas2.height);
        if (selectedCameraId === '2' && selectedPersonId) {
            const modalCanvas = document.getElementById('modal-canvas');
            const modalContext = modalCanvas.getContext('2d');
            modalContext.drawImage(img, 0, 0, modalCanvas.width, modalCanvas.height);
        }
    };
});

socket.on('update_status_1', (data) => {
    if (isPaused && !data.message.includes('System fully paused') && !data.message.includes('System fully resumed')) return;
    updateStatus(1, data.message, data.message.includes('Error') || data.message.includes('not found'));
    if (data.message.includes('Error') || data.message.includes('not found')) {
        if (activeCameraId === '1' || activeCameraId === 'both') {
            document.querySelectorAll('.camera-tile[data-camera-id="1"]').forEach(tile => tile.classList.remove('active'));
            if (activeCameraId === '1') {
                document.getElementById('pause-resume-btn').disabled = true;
                isProcessorReady = false;
                activeCameraId = null;
                isPaused = false;
                document.getElementById('pause-resume-btn').className = 'btn btn-pause';
                document.getElementById('pause-resume-btn').innerHTML = '<i class="fas fa-pause"></i> Pause Detection';
                const context = canvas1.getContext('2d');
                context.fillStyle = '#000';
                context.fillRect(0, 0, canvas1.width, canvas1.height);
                context.fillStyle = '#FFF';
                context.font = '24px Roboto Mono';
                context.textAlign = 'center';
                context.textBaseline = 'middle';
                context.fillText('Select a CCTV feed below', canvas1.width / 2, canvas1.height / 2);
                updateDetections([]);
                lockedDetections = [];
                closeModal();
            }
        }
    }
});

socket.on('update_status_2', (data) => {
    if (isPaused && !data.message.includes('System fully paused') && !data.message.includes('System fully resumed')) return;
    updateStatus(2, data.message, data.message.includes('Error') || data.message.includes('not found'));
    if (data.message.includes('Error') || data.message.includes('not found')) {
        if (activeCameraId === '2' || activeCameraId === 'both') {
            document.querySelectorAll('.camera-tile[data-camera-id="2"]').forEach(tile => tile.classList.remove('active'));
            if (activeCameraId === '2') {
                document.getElementById('pause-resume-btn').disabled = true;
                isProcessorReady = false;
                activeCameraId = null;
                isPaused = false;
                document.getElementById('pause-resume-btn').className = 'btn btn-pause';
                document.getElementById('pause-resume-btn').innerHTML = '<i class="fas fa-pause"></i> Pause Detection';
                const context = canvas2.getContext('2d');
                context.fillStyle = '#000';
                context.fillRect(0, 0, canvas2.width, canvas2.height);
                context.fillStyle = '#FFF';
                context.font = '24px Roboto Mono';
                context.textAlign = 'center';
                context.textBaseline = 'middle';
                context.fillText('Select a CCTV feed below', canvas2.width / 2, canvas2.height / 2);
                updateDetections([]);
                lockedDetections = [];
                closeModal();
            }
        }
    }
});

socket.on('update_detections_1', (data) => {
    if (isPaused) return;
    const detections = data.detections.map(d => ({ ...d, camera_id: '1' }));
    updateDetections(detections);
});

socket.on('update_detections_2', (data) => {
    if (isPaused) return;
    const detections = data.detections.map(d => ({ ...d, camera_id: '2' }));
    updateDetections(detections);
});

socket.on('update_analytics_1', (data) => {
    if (isPaused) return;
    known_count1 = data.known_count || 0;
    visitor_count1 = data.visitor_count || 0;
    calculateAnalytics();
});

socket.on('update_analytics_2', (data) => {
    if (isPaused) return;
    known_count2 = data.known_count || 0;
    visitor_count2 = data.visitor_count || 0;
    calculateAnalytics();
});

socket.on('person_frame', (data) => {
    if (isPaused || !selectedPersonId || selectedCameraId !== data.camera_id) return;
    const img = new Image();
    img.src = `data:image/jpeg;base64,${data.frame}`;
    img.onload = () => {
        const modalCanvas = document.getElementById('modal-canvas');
        const modalContext = modalCanvas.getContext('2d');
        modalContext.drawImage(img, 0, 0, modalCanvas.width, modalCanvas.height);
        if (data.camera_id === '1' && (activeCameraId === '1' || activeCameraId === 'both')) {
            context1.drawImage(img, 0, 0, canvas1.width, canvas1.height);
        } else if (data.camera_id === '2' && (activeCameraId === '2' || activeCameraId === 'both')) {
            context2.drawImage(img, 0, 0, canvas2.width, canvas2.height);
        }
    };
});

const canvas1 = document.getElementById('canvas1');
const context1 = canvas1.getContext('2d');
const canvas2 = document.getElementById('canvas2');
const context2 = canvas2.getContext('2d');

context1.fillStyle = '#000';
context1.fillRect(0, 0, canvas1.width, canvas1.height);
context1.fillStyle = '#FFF';
context1.font = '24px Roboto Mono';
context1.textAlign = 'center';
context1.textBaseline = 'middle';
context1.fillText('Select a CCTV feed below', canvas1.width / 2, canvas1.height / 2);

context2.fillStyle = '#000';
context2.fillRect(0, 0, canvas2.width, canvas2.height);
context2.fillStyle = '#FFF';
context2.font = '24px Roboto Mono';
context2.textAlign = 'center';
context2.textBaseline = 'middle';
context2.fillText('Select a CCTV feed below', canvas2.width / 2, canvas2.height / 2);

updateAnalytics(0, 0);
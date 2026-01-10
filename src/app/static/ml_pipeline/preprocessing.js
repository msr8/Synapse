const preprocessingChartsDiv = document.getElementById('preprocessing-charts');
const correlationMatrixDiv   = document.getElementById('correlation-matrix-div');
const mutualInfoDiv          = document.getElementById('mutual-info-div');
const pairplotDiv            = document.getElementById('pairplot-div');

let todoEDA;


function preprocess(toDoEDA_inp=false) {
    setTarget().then((targetCol) => {
        targetChooserDiv.style.display = "none";
        edaSkipChooser.style.display   = "none";
        todoEDA = toDoEDA_inp;
    
        args = {
            "scaler":         getSelectedRadioValue("scaler"),
            "fs_method":      getSelectedRadioValue("fs-method"),
            "mi_threshold":   document.getElementsByName("mi-threshold")[0].value,
            "corr_threshold": document.getElementsByName("corr-threshold")[0].value,
            "n_features":     document.getElementsByName("n-features")[0].value,
        }
        socket.emit('preprocess', {task_id: taskID, args: args});
    });
}


socket.on('preprocess_error', (data) => {
    console.log("Preprocessing Error:", data.message);
    alert(data.message);
});

socket.on('preprocess_charts_error', (data) => {
    console.log("Preprocessing Charts Error:", data.message);
    alert(data.message);
});


socket.on('preprocess_done', (data) => {
    // Run this function only if EDA has to be done
    if (!todoEDA) {
        postPreprocessing()
        return;
    }

    socket.emit('preprocess_charts', {task_id: taskID});
    // Show the charts div
    preprocessingChartsDiv.style.display = "flex";
});

function postPreprocessing() {
    bayesianChoiceDiv.style.display = "block";
    downloadChartsDiv.style.display = "block";
}



socket.on('correlation_matrix_chart', (data) => {
    let image = document.createElement('img');
    image.src = `data:image/png;base64,${data.base64}`;
    image.classList.add('matplotlib-chart');
    correlationMatrixDiv.appendChild(image);
    zip.file("Correlation Matrix.png", data.base64, {base64: true});
});

socket.on('mutual_info_chart', (data) => {
    let image = document.createElement('img');
    image.src = `data:image/png;base64,${data.base64}`;
    image.classList.add('matplotlib-chart');
    mutualInfoDiv.appendChild(image);
    zip.file("mutual Information.png", data.base64, {base64: true});
});

socket.on('pairplot_chart', (data) => {
    let image = document.createElement('img');
    image.src = `data:image/png;base64,${data.base64}`;
    image.classList.add('matplotlib-chart');
    pairplotDiv.appendChild(image);
    zip.file("Pairplot.png", data.base64, {base64: true});
});

socket.on('preprocess_charts_done', (data) => {
    postPreprocessing();
});



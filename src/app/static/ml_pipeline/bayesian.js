const bayesianChoiceDiv  = document.getElementById("bayesian-choice-div");
const bayesianResultsDiv = document.getElementById("bayesian-results-div");


function startBayesian() {
    bayesianChoiceDiv.style.display = "none";
    downloadChartsDiv.style.display = "none";
    args = {
        "cv":              document.getElementsByName("cv")[0].value,
        "n_iter":          document.getElementsByName("n-iter")[0].value,
        "test_size_ratio": document.getElementsByName("test-size-ratio")[0].value,
        "scorer":          getSelectedRadioValue("scorer"),
        "stopper":         getSelectedRadioValue("stopper"),
        "deadline_time":   document.getElementsByName("deadline-time")[0].value,
        "threshold_value": "-" + document.getElementsByName("threshold-value")[0].value,
        "delta_value":     document.getElementsByName("delta-value")[0].value,
        "cm_normalize":    getSelectedRadioValue("cm-normalize"),
    }
    socket.emit('start_bayesian', {task_id: taskID, args: args});
    document.getElementById("charts").style.display = "none";
    document.getElementById("preprocessing-charts").style.display = "none";
    bayesianResultsDiv.style.display = "block";
}



socket.on('bayesian_error', (data) => {
    console.log("Bayesian Error:", data.message);
    alert(data.message);
});


socket.on('bayesian_model_error', (data) => {
    statusDiv = document.getElementById(`${data.model}-status`);
    statusDiv.innerHTML = `Error: ${data.message}`;
    statusDiv.style.color = "red";
    statusDiv.style.fontWeight = "bold";
});



socket.on('bayesian_callback', (data) => {
    console.log("Bayesian Callback:", data);
    
    document.getElementById(`${data.model}-n-iter`).innerText = data.n_iter;
    document.getElementById(`${data.model}-best-score`).innerText = data.best_score.toFixed(2);
});


socket.on('model_result', (data) => {
    console.log(data);


    let paramTableHTML = "<table class='table table-bordered table-striped'><thead><tr>"
    // Go through the keys of data.best_params and create a table row for each key
    Object.keys(data.best_params).forEach(key => { paramTableHTML += `<th>${key}</th>`; });
    paramTableHTML += "</tr></thead><tbody>";
    // Go through the values of data.best_params
    paramTableHTML += "<tr>";
    Object.values(data.best_params).forEach(value => { paramTableHTML += `<td>${value}</td>`; });
    paramTableHTML += "</tr></tbody></table>";

    document.getElementById(`${data.model}-status`).innerHTML = `Iterations Done: ${data.n_iter}<br>Testing score: ${data.testing_score.toFixed(2)}<br>Best Parameters: ${paramTableHTML}<br>Time Taken: ${data.time_taken.toFixed(2)} seconds`;
    
    let confMatrix = document.getElementById(`${data.model}-confusion-matrix`);
    confMatrix.src = `data:image/png;base64,${data.confusion_matrix_base64}`;
    confMatrix.style.display = "block";
});





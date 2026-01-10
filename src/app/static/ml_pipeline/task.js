// import { animate } from "motion"

const socket = io();
const zip    = JSZip();

const taskID             = document.getElementById("task-id").innerHTML;
const targetChooserDiv   = document.getElementById("target-chooser");
const edaSkipChooser     = document.getElementById("eda-skip-chooser");
const targetRadios       = document.getElementsByName("target");
const EdaResultsDiv      = document.getElementById("eda-results");
const chartsDiv          = document.getElementById("charts");
const downloadChartsDiv  = document.getElementById("download-charts-div");
const tasknameSpan       = document.getElementById("task-name-span");
const settingsDiv        = document.getElementById("settings-div");
const dataToolsTabButton = document.getElementById("tab1");

// AOS.init({
//     offset: 0
// });
gsap.registerPlugin(ScrollTrigger);
console.log("Task ID:", taskID);
dataToolsTabButton.checked = true;

initialiseSettings();
// For every radio button in #settings-div, add an event listener to run initialiseSettings
const radioButtons = settingsDiv.querySelectorAll('input[type="radio"]');
radioButtons.forEach((radioButton) => {
    radioButton.addEventListener('click', (event) => {
        console.log("Radio button clicked:", event.target.value);
        initialiseSettings();
    });
});





function getSelectedRadioValue(name) {
    const radios = document.getElementsByName(name);
    for (let i = 0; i < radios.length; i++) {
        if (radios[i].checked) {return radios[i].value;}
    }
    return null;
}



function showChatbot() {
    chatbotContent.style.display    = "block";
    nonChatbotContent.style.display = "none";
}

function showNonChatbot() {
    chatbotContent.style.display    = "none";
    nonChatbotContent.style.display = "block";
}

function changeTaskname() {
    // Prompt user for new username
    let newTaskname = prompt("Enter new taskname:");

    fetch('/api/task/change-taskname', {
        method: 'POST',
        body: JSON.stringify({new_taskname: newTaskname, task_id:taskID}),
        headers: {'Content-Type': 'application/json'}
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            tasknameSpan.innerText = newTaskname;
        }
        else {alert(data.message);}
    });
}

function deleteTask() {
    // Confirm with the user before deleting the task
    if (!confirm("Are you sure you want to delete this task? This action cannot be undone.")) {return;}

    fetch('/api/task/delete-task', {
        method: 'POST',
        body: JSON.stringify({task_id: taskID}),
        headers: {'Content-Type': 'application/json'}
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            // Redirect to dashbord
            window.location.href = "/dashboard";
        }
        else {alert(data.message);}
    });
}




function setTarget() {
    return new Promise((resolve, reject) => {
        // Get the name of the target column
        let targetCol = getSelectedRadioValue("target");
        console.log("Target column:", targetCol);

        // If no target column is selected, show an alert and reject the promise
        if (targetCol == null) { alert("Please select a target column"); return reject(); }

        fetch('/api/task/set-target', {
            method: 'POST',
            body: JSON.stringify({task_id: taskID, target: targetCol}),
            headers: {'Content-Type': 'application/json'}
        })
        .then(response => response.json())
        .then(data => {
            if      (data.status == "error")   { alert(data.message); return reject(); }
            else if (data.status != "success") { alert("An error occurred while setting the target column"); return reject(); }

            console.log("Target column set successfully");
            resolve(targetCol);
        });
    });
}


function toggleSettings() {
    settingsDiv.style.display = settingsDiv.style.display == "none" ? "block" : "none";
    // if (settingsDiv.style.display == "block") {
    //     settingsDiv.style.setProperty("--animate-duration", "0.5s");
    //     gsap.fromTo(settingsDiv, { opacity: 0 }, { opacity: 1 });
    // }
    // else {
    //     settingsDiv.style.setProperty("--animate-duration", "0.5s");
    //     gsap.fromTo(settingsDiv, { opacity: 1 }, { opacity: 0 });
    // }
}


function initialiseSettings() {
    // Preprocessing
    ["mi-corr-settings", "n-features-span"].forEach((id) => {document.getElementById(id).style.display = "none";});
    let fsMethod = getSelectedRadioValue("fs-method");
    console.log("FS Method:", fsMethod);
    if (fsMethod === "corr+mi") {document.getElementById("mi-corr-settings").style.display = "block";}
    else                        {document.getElementById("n-features-span").style.display  = "block";}

    ["n-iter-span", "deadline-time-span", "threshold-value-span", "delta-value-span"].forEach((id) => {document.getElementById(id).style.display = "none";});
    let bayesianStopper = getSelectedRadioValue("stopper");
    console.log("Bayesian Stopper:", bayesianStopper);
    if      (bayesianStopper === "n_iter")    {document.getElementById("n-iter-span").style.display          = "block";}
    else if (bayesianStopper === "deadline")  {document.getElementById("deadline-time-span").style.display   = "block";}
    else if (bayesianStopper === "threshold") {document.getElementById("threshold-value-span").style.display = "block";}
    else if (bayesianStopper === "delta")     {document.getElementById("delta-value-span").style.display     = "block";}
}



function eda() {
    setTarget().then((targetCol) => {
        args = {"n_unique_threshold": document.getElementById("n-unique-threshold-input").value};

        let sock_data = {task_id: taskID, target: targetCol, args: args};
        socket.emit('eda', sock_data);

        chartsDiv.style.display        = "flex";
        // EdaResultsDiv.style.display    = "block";
        targetChooserDiv.style.display = "none";
        edaSkipChooser.style.display   = "none";

        preprocess(true);
    });
}





socket.on('eda_error', (data) => {
    console.log("EDA Error:", data.error);
    alert(data.message);
});


socket.on('eda_chart', (data) => {
    let columnName = data.col;
    let idx        = data.idx;

    const chartContainer = document.getElementById(`chart-${idx}-container`);
    let element;
    console.log(data);

    if (data.type == "pygal") {
        // <embed type="image/svg+xml" id="chart-{{i}}-embed" src= {{ chart|safe }} />
        element = document.createElement("embed");
        element.setAttribute("type", "image/svg+xml");
        element.setAttribute("id", `chart-${idx}-embed`);
        element.setAttribute("src", data.data);
        chartContainer.classList.add("pygal-chart-container");

        zip.file(`${columnName}.svg`, data.data.split(",")[1], {base64: true});
    }
    else if (data.type == "matplotlib") {
        // <img id="chart-{{i}}-img" src= {{ chart|safe }} />
        element = document.createElement("img");
        element.setAttribute("id", `chart-${idx}-img`);
        element.setAttribute("src", `data:image/png;base64,${data.data}`);
        element.classList.add("matplotlib-chart");

        zip.file(`${columnName}.png`, data.data, {base64: true});
    }

    chartContainer.appendChild(element);
    chartContainer.style.display = "flex";
});




function downloadCharts() {
    zip.generateAsync({type: "blob"})
    .then(function(content) {
        saveAs(content, "charts.zip");
    });
}














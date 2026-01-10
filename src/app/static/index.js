// const socket = io();
// const messageDiv = document.getElementById("messages");

// socket.emit('start_task', {task: 'task1'});

// // Listen for messages from the server
// socket.on('message', (data) => {
//     // console.log("Message from server:", data);
//     messageDiv.innerHTML += `<p>${data}</p>`;
// });

// // socket.on('response', (data) => {
// //     console.log("Custom event response:", data);
// // });

// socket.on('update', (data) => {
//     console.log("Custom event response:", data);
// });

// // // Send a message to the server
// // document.getElementById("sendBtn").addEventListener("click", () => {
// //     const message = document.getElementById("messageInput").value;
// //     socket.send(message);
// // });

const fileInput = document.getElementById("file-input");

function uploadFile() {
    if (fileInput.files.length === 0) {
        alert("Please select a file.");
        return;
    }

    const formData = new FormData();
    formData.append("file", fileInput.files[0]);

    fetch("/api/upload", {
        method: "POST",
        body: formData
    })
        .then(response => response.json())
        .then(data => {
            if      (data.status === "redirect") { window.location.href = data.url;      }
            else if (data.status === "error")    { alert(data.message);                  }
            else                                 { alert("An unknown error occurred."); }
        })
}



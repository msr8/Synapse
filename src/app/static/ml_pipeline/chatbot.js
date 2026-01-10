const chatbotContent      = document.getElementById("chatbot-content");
const nonChatbotContent   = document.getElementById("non-chatbot-content");
const chatbotButton       = document.getElementById("tab2");
const chatbotInitialising = document.getElementById("chatbot-initialising");
const chatsContainer      = document.getElementById("chats-container");
const messageInput        = document.getElementById("message-input");
const chatbotTabButton    = document.getElementById("tab2");
const chatbotPlaceholder  = document.getElementById("chatbot-placeholder")






function addChatMessage(message, role) {
    chatbotPlaceholder.style.display = "none";
    // <div class="chat user-message">
    //     <span class="icon user-icon"><i class="bi-person-circle"></i></span>
    //     <div class="message-content">hello</div>
    // </div>
    // In case of user message, adding the icon before works cause .user-icon has `flex-direction: row-reverse`
    let chatDiv = document.createElement("div");
    chatDiv.className = `chat ${role}-message`;
    
    let iconSpan = document.createElement("span");
    iconSpan.className = `icon ${role}-icon`;
    iconSpan.innerHTML = `<i class="bi-${role == "user" ? "person-circle" : "robot"}"></i>`;

    let messageDiv = document.createElement("div");
    messageDiv.className = "message-content";
    messageDiv.innerHTML = message;

    chatDiv.appendChild(iconSpan);
    chatDiv.appendChild(messageDiv);
    chatsContainer.appendChild(chatDiv);

    return messageDiv;
}




function initialiseChatbot() {
    setTarget().then((targetCol) => {
        console.log("Promise was accepted: ", targetCol);
        
        // Initialise the chatbot
        fetch('/api/task/chatbot/initialise', {
            method: 'POST',
            body: JSON.stringify({task_id: taskID}),
            headers: {'Content-Type': 'application/json'}
        })
        .then(response => response.json())
        .then(data => {
            if      (data.status == "error")   { alert(data.message); return; }
            else if (data.status != "success") { alert("An error occurred while initialising the chatbot"); return; }

            chatbotContent.style.display    = "block";
            nonChatbotContent.style.display = "none";
            chatbotButton.onclick           = showChatbot;

            // Display the already existing chat messages
            for (let msg of data.messages) {
                addChatMessage(msg.message, msg.role);
            }

            console.log("Chatbot initialised successfully");
            console.log(data.messages);
        });
    })
    .catch((error) => {
        // Stop the chatbot tab button from being selected
        chatbotTabButton.checked = false;
    });
}




function chat() {
    let text = messageInput.value;
    messageInput.value = "";

    if (text == "") { return; }

    addChatMessage(text, "user");
    let botResponse = addChatMessage("Loading...", "bot");

    fetch('/api/task/chatbot/chat', {
        method: 'POST',
        body: JSON.stringify({task_id: taskID, text: text}),
        headers: {'Content-Type': 'application/json'}
    })
    .then(response => response.json())
    .then(data => {
        if      (data.status == "error")   { alert(data.message); return; }
        else if (data.status != "success") { alert("An error occurred while sending the message to the chatbot"); return; }

        // Replace "Loading..." with the bot's response
        botResponse.innerHTML = data.response;
    });
}




function chatReset() {
    fetch('/api/task/chatbot/reset', {
        method: 'POST',
        body: JSON.stringify({task_id: taskID}),
        headers: {'Content-Type': 'application/json'}
    })
    .then(response => response.json())
    .then(data => {
        if      (data.status == "error")   { alert(data.message); return; }
        else if (data.status != "success") { alert("An error occurred while resetting the chatbot"); return; }

        // Clear the chat messages
        chatsContainer.innerHTML = "";
        // Make the placeholder reappear
        chatbotPlaceholder.style.display = "flex";
        console.log("Chatbot reset successfully");
    });
}